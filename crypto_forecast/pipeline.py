"""Train, freeze, evaluate, and reconstruct multi-horizon forecasts."""

import copy
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from . import HORIZONS, __version__
from .data import load_data, load_multiframe_data
from .features import FEATURE_NAMES, build_samples, inference_sample, split_samples
from .model import MODEL_SPECS, Normalizer, build_model
from .multiframe import MTF_FEATURE_NAMES, build_multiframe_samples, inference_multiframe_sample
from .timeframes import TIMEFRAMES, interval_delta
from .report import write_report
from .learning_rate import LearningRateSchedule, validate_schedule


@dataclass
class TrainingConfig:
    coins: list[str]
    start: str = "2021-01-01"
    end: str | None = None
    data_dir: str = "data"
    run_dir: str = "runs/multiframe-coins"
    input_mode: str = "multiframe"
    model_type: str = "gru"
    origin_interval: str = "4h"
    lookback: int = 60
    hidden_size: int = 48
    epochs: int = 60
    patience: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    lr_scheduler: str = "warmup_cosine"
    lr_min_ratio: float = 0.1
    lr_warmup_fraction: float = 0.05
    lr_factor: float = 0.5
    lr_step_epochs: int = 5
    lr_patience: int = 2
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    seed: int = 42
    device: str = "auto"
    kronos_timeframe: str = "4h"
    kronos_prediction_length: int | None = 24
    kronos_model_path: str = "NeoQuasar/Kronos-small"
    kronos_tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_sample_count: int = 1
    kronos_eval_stride: int = 6
    kronos_train_stride: int = 6
    kronos_preparation_workers: int = 0
    kronos_finetune_tokenizer: bool = True
    kronos_training_stage: str = "sequential"
    kronos_tokenizer_run_dir: str | None = None
    kronos_tokenizer_batch_size: int | None = None
    kronos_tokenizer_epochs: int = 5
    kronos_tokenizer_patience: int = 2
    kronos_tokenizer_learning_rate: float = 2e-4
    kronos_tokenizer_lr_scheduler: str = "onecycle"
    kronos_tokenizer_lr_min_ratio: float = 0.1
    kronos_tokenizer_lr_warmup_fraction: float = 0.03
    kronos_tokenizer_lr_factor: float = 0.5
    kronos_tokenizer_lr_step_epochs: int = 5
    kronos_tokenizer_lr_patience: int = 1

    def validate(self):
        if not isinstance(self.coins, list) or not all(isinstance(c, str) for c in self.coins):
            raise ValueError("coins must be a JSON list of coin symbols.")
        self.coins = [c.upper() for c in self.coins]
        if not self.coins or len(set(self.coins)) != len(self.coins):
            raise ValueError("Select a nonempty group of unique coins.")
        if any(not c.isascii() or not c.isalnum() or c == "USDT" for c in self.coins):
            raise ValueError("Coin symbols must be alphanumeric; USDT is the quote unit.")
        integer_values = (self.lookback, self.hidden_size, self.epochs, self.patience, self.batch_size, self.seed)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise ValueError("Lookback, hidden size, epochs, patience, batch size and seed must be integers.")
        if min(self.lookback, self.hidden_size, self.epochs, self.patience, self.batch_size) < 1:
            raise ValueError("Lookback, hidden size, epochs, patience and batch size must be positive.")
        if type(self.learning_rate) not in (int, float) or not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        validate_schedule(self)
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu, or cuda.")
        if self.input_mode not in {"daily", "multiframe"}:
            raise ValueError("input_mode must be daily or multiframe.")
        if not isinstance(self.model_type, str) or self.model_type not in MODEL_SPECS:
            raise ValueError("model_type must be gru, lstm, kronos-small, or kronos-base.")
        if self.model_type in {"kronos-small", "kronos-base"}:
            if self.kronos_training_stage not in {"tokenizer", "predictor", "sequential"}:
                raise ValueError("kronos_training_stage must be tokenizer, predictor, or sequential.")
            if self.kronos_tokenizer_batch_size is not None and (
                    type(self.kronos_tokenizer_batch_size) is not int or not 1 <= self.kronos_tokenizer_batch_size <= 8192):
                raise ValueError("kronos_tokenizer_batch_size must be an integer from 1 to 8192.")
            if self.kronos_tokenizer_run_dir is not None:
                if not isinstance(self.kronos_tokenizer_run_dir, str) or not self.kronos_tokenizer_run_dir.strip():
                    raise ValueError("kronos_tokenizer_run_dir must be a nonempty directory path.")
                if self.kronos_training_stage != "predictor":
                    raise ValueError("A saved tokenizer can only be selected for predictor-only training.")
            if type(self.kronos_preparation_workers) is not int or not 0 <= self.kronos_preparation_workers <= 4:
                raise ValueError("kronos_preparation_workers must be an integer from 0 (automatic) to 4.")
            validate_schedule(self, "kronos_tokenizer_")
            if self.model_type == "kronos-base" and self.kronos_model_path == "NeoQuasar/Kronos-small":
                self.kronos_model_path = "NeoQuasar/Kronos-base"
            if not isinstance(self.kronos_finetune_tokenizer, bool):
                raise ValueError("kronos_finetune_tokenizer must be a boolean.")
            for name in ("kronos_tokenizer_epochs", "kronos_tokenizer_patience"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer.")
            value = self.kronos_tokenizer_learning_rate
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError("kronos_tokenizer_learning_rate must be a finite positive number.")
            if self.kronos_timeframe not in TIMEFRAMES:
                raise ValueError("kronos_timeframe must be 15m, 1h, 4h or 1d.")
            if self.kronos_prediction_length is not None and (
                    type(self.kronos_prediction_length) is not int or not 1 <= self.kronos_prediction_length <= 512):
                raise ValueError("kronos_prediction_length must be an integer between 1 and 512.")
            # None preserves the original target definition for legacy runs.
            steps = (self.kronos_prediction_length if self.kronos_prediction_length is not None else
                     int(pd.Timedelta(days=max(HORIZONS)) / interval_delta(self.kronos_timeframe)))
            if self.lookback + steps - 1 > 512:
                raise ValueError(f"{self.model_type.capitalize()}'s 512-token training context allows lookback at most {513-steps} for {self.kronos_timeframe}.")
            for name in ("kronos_sample_count", "kronos_eval_stride", "kronos_train_stride"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer.")
            for name in ("kronos_model_path", "kronos_tokenizer_path"):
                if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                    raise ValueError(f"{name} must be a local pretrained directory or Hugging Face model ID.")
        interval_delta(self.origin_interval)
        if not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be between 0 and 2**32-1.")
        start = _utc(self.start)
        end = _utc(self.end) if self.end is not None else pd.Timestamp.now(tz="UTC").floor("D")
        if start != start.floor("D") or (self.end is not None and end != end.floor("D")):
            raise ValueError("Training start/end must be UTC midnight dates.")
        if start >= end:
            raise ValueError("Training start must precede end.")


def _utc(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Invalid UTC date or timestamp.")
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else
                        "cpu" if requested == "auto" else requested)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _arrays(samples, mapping, x_scaler, y_scaler=None):
    x = np.empty((len(samples), *samples[0].x.shape), dtype=np.float32)
    for start in range(0, len(samples), 256):
        block = samples[start:start+256]
        x[start:start+len(block)] = x_scaler.transform(np.stack([s.x for s in block]))
    coins = np.asarray([mapping[s.coin] for s in samples], dtype=np.int64)
    y = np.stack([s.y for s in samples])
    if y_scaler is not None:
        y = y_scaler.transform(y)
    return torch.from_numpy(x), torch.from_numpy(coins), torch.from_numpy(y.astype(np.float32))


def _predict(model, samples, mapping, x_scaler, y_scaler, device, batch_size=256):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            x, coins, _ = _arrays(samples[start:start+batch_size], mapping, x_scaler)
            outputs.append(model(x.to(device), coins.to(device)).cpu().numpy())
    result = y_scaler.inverse(np.concatenate(outputs))
    if not np.isfinite(result).all() or not np.isfinite(np.exp(result)).all():
        raise ValueError("The model produced non-finite price estimates.")
    return result


def _sample_for(coin, frame, config):
    if config.input_mode == "multiframe":
        return inference_multiframe_sample(coin, frame, config.lookback, config.origin_interval)
    return inference_sample(coin, frame, config.lookback)


def _manifest(frame):
    return {"csv_sha256": frame.attrs["data_sha256"],
            "used_rows_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
            "rows": len(frame), "first_open": frame.timestamp.min().isoformat(),
            "last_open": frame.timestamp.max().isoformat(),
            "quality": frame.attrs.get("quality", {}),
            "provenance": copy.deepcopy(frame.attrs["provenance"])}


def _macro_loss(prediction, target, samples):
    per_sample = ((prediction - target) ** 2).mean(axis=1)
    return float(np.mean([per_sample[[s.coin == c for s in samples]].mean()
                          for c in sorted({s.coin for s in samples})]))


def _metrics(samples, prediction, residual_bounds, non_overlapping=False, model_type="gru", *, horizons=HORIZONS):
    rows = []
    for coin in sorted({s.coin for s in samples}):
        indices = sorted([i for i, s in enumerate(samples) if s.coin == coin],
                         key=lambda i: samples[i].origin)
        for j, horizon in enumerate(horizons):
            selected = indices
            if non_overlapping:
                selected, next_origin = [], None
                for i in indices:
                    if next_origin is None or samples[i].origin >= next_origin:
                        selected.append(i)
                        next_origin = samples[i].origin + pd.Timedelta(days=horizon)
            truth = np.asarray([samples[i].y[j] for i in selected])
            for name, estimates in (
                (model_type, prediction[selected, j]),
                ("persistence", np.zeros(len(selected))),
                ("drift", np.asarray([samples[i].drift[j] for i in selected])),
            ):
                actual_return, estimate_return = np.expm1(truth), np.expm1(estimates)
                error = 100 * (estimate_return - actual_return)
                direction_mask = (np.abs(truth) > 1e-10) & (np.abs(estimates) > 1e-10)
                direction = (float((np.sign(truth[direction_mask]) ==
                                    np.sign(estimates[direction_mask])).mean())
                             if direction_mask.any() else None)
                coverage, width = None, None
                if name == model_type:
                    low, high = estimates + residual_bounds[0, j], estimates + residual_bounds[1, j]
                    coverage = float(((truth >= low) & (truth <= high)).mean())
                    width = float(100 * (np.exp(high) - np.exp(low)).mean())
                rows.append({"coin": coin, "horizon_days": horizon, "model": name,
                             "samples": len(selected), "return_mae_pct": float(np.abs(error).mean()),
                             "return_rmse_pct": float(np.sqrt((error ** 2).mean())),
                             "log_return_mae": float(np.abs(estimates-truth).mean()),
                             "direction_accuracy": direction, "interval_coverage": coverage,
                             "interval_width_pct": width})
    for horizon in horizons:
        for name in (model_type, "persistence", "drift"):
            group = [r for r in rows if r["horizon_days"] == horizon and r["model"] == name]
            aggregate = {"coin": "ALL", "horizon_days": horizon, "model": name,
                         "samples": sum(r["samples"] for r in group)}
            for field in ("return_mae_pct", "return_rmse_pct", "log_return_mae",
                          "direction_accuracy", "interval_coverage", "interval_width_pct"):
                values = [r[field] for r in group if r[field] is not None]
                aggregate[field] = float(np.mean(values)) if values else None
            rows.append(aggregate)
    return rows


def _prediction_rows(samples, predictions, bounds, *, horizons=HORIZONS):
    rows = []
    for sample, estimates in zip(samples, predictions):
        for j, horizon in enumerate(horizons):
            rows.append({"coin": sample.coin, "as_of": sample.origin.isoformat(),
                         "target_date": (sample.origin + pd.Timedelta(days=horizon)).isoformat(),
                         "horizon_days": horizon, "last_price_usdt": sample.price,
                         "actual_price_usdt": float(sample.price*np.exp(sample.y[j])),
                         "forecast_price_usdt": float(sample.price*np.exp(estimates[j])),
                         "actual_change_pct": float(100*np.expm1(sample.y[j])),
                         "change_pct": float(100*np.expm1(estimates[j])),
                         "lower_price_usdt": float(sample.price*np.exp(estimates[j]+bounds[0, j])),
                         "upper_price_usdt": float(sample.price*np.exp(estimates[j]+bounds[1, j]))})
    return rows


def train(config: TrainingConfig, *, overwrite=False, data_kind="historical", progress=print,
          on_event=None, on_training_decision=None):
    """Train a frozen run, optionally reporting structured live progress.

    ``on_event`` receives independent dictionaries suitable for a dashboard.
    Callback exceptions propagate so a caller can interrupt its own training job.
    The existing text ``progress`` callback retains its original timing.
    ``on_training_decision`` can wait for a browser choice when patience expires.
    Returning True grants exactly ten more epochs, then requests another choice.
    Returning False finishes with the best validation weights across the whole run.
    Without this callback, early stopping retains its automatic CLI behavior.
    """
    started_at = time.monotonic()

    def emit(kind, **values):
        if on_event is not None:
            on_event({"kind": kind, **values, "elapsed_seconds": time.monotonic() - started_at})

    config.validate()
    if config.model_type in {"kronos-small", "kronos-base"}:
        from .kronos_backend import train_kronos
        return train_kronos(config, overwrite=overwrite, data_kind=data_kind, progress=progress,
                            on_event=on_event, on_training_decision=on_training_decision)
    if on_training_decision is not None and not callable(on_training_decision):
        raise ValueError("on_training_decision must be a callable or None.")
    if data_kind not in {"historical", "synthetic", "synthetic_demo"}:
        raise ValueError("data_kind must be historical or synthetic_demo.")
    if data_kind == "synthetic":
        data_kind = "synthetic_demo"
    run_dir = Path(config.run_dir)
    if (run_dir / "model.pt").exists() and not overwrite:
        raise ValueError(f"{run_dir} already contains a trained run. Select another --run-dir or use --overwrite.")
    config_end = _utc(config.end) if config.end else pd.Timestamp.now(tz="UTC")
    loader_function = load_multiframe_data if config.input_mode == "multiframe" else load_data
    emit("stage", stage="loading")
    frames = loader_function(config.data_dir, config.coins, as_of=min(config_end, pd.Timestamp.now(tz="UTC")))
    progress(f"Loaded {config.input_mode} candles for {','.join(config.coins)}")
    start = _utc(config.start)
    if config.input_mode == "multiframe":
        frames = {coin: {tf: frame[(frame.timestamp >= start) & (frame.timestamp < config_end)].copy()
                         for tf, frame in branches.items()} for coin, branches in frames.items()}
        flat_frames = [frame for branches in frames.values() for frame in branches.values()]
    else:
        frames = {coin: frame[(frame.timestamp >= start) & (frame.timestamp < config_end)].copy()
                  for coin, frame in frames.items()}
        flat_frames = list(frames.values())
    if any("synthetic" in str(frame.attrs.get("provenance", {}).get("source", "")).lower()
           for frame in flat_frames):
        data_kind = "synthetic_demo"
    progress("Building aligned historical inputs and exact future targets")
    emit("stage", stage="preparing")
    samples = (build_multiframe_samples(frames, config.lookback, config.origin_interval)
               if config.input_mode == "multiframe" else build_samples(frames, config.lookback))
    partitions, split_info = split_samples(samples, config.train_fraction, config.validation_fraction)
    excluded = {}
    latest_samples = {}
    for coin in config.coins:
        counts = {part: sum(s.coin == coin for s in values) for part, values in partitions.items()}
        if counts.get("train", 0) < 32 or counts.get("validation", 0) < 5 or counts.get("test", 0) < 5:
            excluded[coin] = f"Insufficient contiguous samples in shared calendar splits: {counts}; need 32/5/5."
        else:
            try:
                latest_samples[coin] = _sample_for(coin, frames[coin], config)
            except ValueError as error:
                excluded[coin] = f"Latest forecasting window unavailable: {error}"
    coins = [c for c in config.coins if c not in excluded]
    if not coins:
        raise ValueError("No coin has enough history for purged training/validation/test. " + str(excluded))
    for part in partitions:
        partitions[part] = [s for s in partitions[part] if s.coin in coins]
    split_info["eligible_counts"] = {p: {c: sum(s.coin == c for s in values) for c in coins}
                                      for p, values in partitions.items()}
    mapping = {coin: i for i, coin in enumerate(coins)}
    train_samples, validation_samples, test_samples = (partitions[p] for p in ("train", "validation", "test"))
    x_scaler = (Normalizer.fit_multiframe(train_samples) if config.input_mode == "multiframe"
                else Normalizer.fit(np.concatenate([s.x for s in train_samples])))
    y_scaler = Normalizer.fit(np.stack([s.y for s in train_samples]))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = _device(config.device)
    feature_names = MTF_FEATURE_NAMES if config.input_mode == "multiframe" else FEATURE_NAMES
    model = build_model(len(feature_names), len(coins), config.hidden_size,
                        input_mode=config.input_mode, model_type=config.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    x, coin_ids, y = _arrays(train_samples, mapping, x_scaler, y_scaler)
    counts = {c: sum(s.coin == c for s in train_samples) for c in coins}
    sampler = WeightedRandomSampler([1/counts[s.coin] for s in train_samples], len(train_samples),
                                    generator=torch.Generator().manual_seed(config.seed))
    loader = DataLoader(TensorDataset(x, coin_ids, y), batch_size=config.batch_size, sampler=sampler)
    scheduler = LearningRateSchedule(optimizer, config, config.epochs, len(loader))
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    validation_y = y_scaler.transform(np.stack([s.y for s in validation_samples]))
    history, best_loss, best_state, best_epoch = [], float("inf"), None, 0
    epoch, epoch_limit, extension_end = 0, config.epochs, None
    decisions = []
    progress(f"Training on {device}: {len(train_samples)} train / {len(validation_samples)} validation / "
             f"{len(test_samples)} test samples; coins={','.join(coins)}")
    emit("stage", stage="training", counts={"train": len(train_samples),
         "validation": len(validation_samples), "test": len(test_samples)}, device=str(device), learning_rate=scheduler.rate)
    while epoch < epoch_limit:
        epoch += 1
        model.train()
        losses = []
        running_loss, running_samples, rate_sum = 0.0, 0, 0.0
        last_batch_event = time.monotonic()
        for batch_number, (batch_x, batch_coins, batch_y) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch_x.to(device), batch_coins.to(device)), batch_y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; inspect data and learning rate.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.batch_finished()
            rate_sum += scheduler.last_used
            losses.append((float(loss.detach().cpu()), len(batch_y)))
            running_loss += losses[-1][0] * len(batch_y)
            running_samples += len(batch_y)
            batch_time = time.monotonic()
            if on_event is not None and (batch_number == 1 or batch_number == len(loader)
                                         or batch_time - last_batch_event >= 1.0):
                emit("batch", stage="training", epoch=epoch, max_epochs=epoch_limit,
                     batch=batch_number, total_batches=len(loader),
                     train_loss=running_loss / running_samples, learning_rate=scheduler.last_used)
                last_batch_event = batch_time
        validation_prediction = _predict(model, validation_samples, mapping, x_scaler, y_scaler, device)
        validation_loss = _macro_loss(y_scaler.transform(validation_prediction), validation_y, validation_samples)
        training_loss = sum(loss*size for loss, size in losses)/sum(size for _, size in losses)
        scheduler.epoch_finished(validation_loss)
        history.append({"epoch": epoch, "train_loss": training_loss, "validation_loss": validation_loss,
                        "learning_rate": rate_sum / len(loader), "lr_end": scheduler.last_used, "lr_next": scheduler.rate})
        progress(f"Epoch {epoch:3}: train={training_loss:.5f} validation={validation_loss:.5f} lr={scheduler.last_used:.6g}")
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        emit("epoch", stage="training", epoch=epoch, max_epochs=epoch_limit,
             train_loss=training_loss, validation_loss=validation_loss,
             learning_rate=history[-1]["learning_rate"], lr_end=scheduler.last_used, lr_next=scheduler.rate,
             best_epoch=best_epoch, best_validation_loss=best_loss)
        if extension_end is not None:
            if epoch < extension_end:
                continue
            reason = "extension_complete"
        elif epoch - best_epoch >= config.patience:
            reason = "early_stop"
        else:
            continue
        if on_training_decision is None:
            break
        decision = {"reason": reason, "decision_id": len(decisions) + 1,
                    "epoch": epoch, "max_epochs": epoch_limit,
                    "best_epoch": best_epoch, "best_validation_loss": best_loss,
                    "train_loss": training_loss, "validation_loss": validation_loss,
                    "extension_epochs": 10}
        emit("decision", stage="awaiting_decision", **decision)
        continue_training = on_training_decision(dict(decision))
        if not isinstance(continue_training, bool):
            raise ValueError("The training decision must return True to continue or False to finish.")
        decisions.append({**decision, "action": "continue" if continue_training else "finish"})
        if not continue_training:
            break
        epoch_limit = extension_end = epoch + 10
        progress(f"Continuation approved: training 10 more epochs, through epoch {epoch_limit}.")
        emit("stage", stage="training", epoch=epoch, max_epochs=epoch_limit,
             best_epoch=best_epoch, best_validation_loss=best_loss)
    model.load_state_dict(best_state)
    emit("stage", stage="evaluating")
    changed = any(not torch.equal(initial_state[k], v.detach().cpu()) for k, v in model.state_dict().items())
    validation_prediction = _predict(model, validation_samples, mapping, x_scaler, y_scaler, device)
    validation_truth = np.stack([s.y for s in validation_samples])
    # These bands are fitted to validation only. Their honest coverage check is on the final test.
    residual_bounds = np.quantile(validation_truth-validation_prediction, [0.1, 0.9], axis=0)
    test_prediction = _predict(model, test_samples, mapping, x_scaler, y_scaler, device)
    metrics = {"validation": _metrics(validation_samples, validation_prediction, residual_bounds,
                                      model_type=config.model_type),
               "test": _metrics(test_samples, test_prediction, residual_bounds, model_type=config.model_type),
               "test_non_overlapping": _metrics(test_samples, test_prediction, residual_bounds, True,
                                                 model_type=config.model_type)}
    data_manifest = {}
    for coin, frame in frames.items():
        if coin in coins:
            data_manifest[coin] = ({"timeframes": {tf: _manifest(branch) for tf, branch in frame.items()}}
                                   if config.input_mode == "multiframe" else _manifest(frame))
    metadata = {"format_version": 2 if config.input_mode == "multiframe" else 1,
                "package_version": __version__, "config": asdict(config),
                "model_type": config.model_type,
                "input_mode": config.input_mode, "timeframes": list(TIMEFRAMES) if config.input_mode == "multiframe" else ["1d"],
                "origin_interval": config.origin_interval if config.input_mode == "multiframe" else "1d",
                "coins": coins, "features": list(feature_names), "horizons": list(HORIZONS),
                "x_scaler": x_scaler.to_dict(), "y_scaler": y_scaler.to_dict(),
                "residual_bounds": residual_bounds.tolist(), "data_kind": data_kind}
    interval_note = ("Empirical 80% validation-residual ranges, pooled across eligible coins per horizon. "
                     "The validation block also selects the epoch, so these are not independent calibration "
                     "or guaranteed probabilities. Final-test coverage measures their out-of-sample behavior. "
                     "Overlapping labels are dependent; non-overlapping test origins are reported separately.")
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "package_version": __version__,
               "data_kind": data_kind, "coins": coins, "config": asdict(config), "splits": split_info,
               "model_type": config.model_type,
               "excluded_coins": excluded, "data_manifest": data_manifest, "best_epoch": best_epoch,
               "history": history, "metrics": metrics, "interval_note": interval_note,
               "parameters_changed": changed, "torch_version": str(torch.__version__),
               "device": str(device), "deterministic_algorithms": True,
               "training_control": {"initial_epochs": config.epochs, "epoch_limit": epoch_limit,
                                    "extension_epochs": 10, "epochs_completed": len(history),
                                    "decisions": decisions}}
    latest = [latest_samples[c] for c in coins]
    latest_prediction = _predict(model, latest, mapping, x_scaler, y_scaler, device)
    now = pd.Timestamp.now(tz="UTC")
    freshness = 8/24 if config.input_mode == "multiframe" else 2
    staleness = {s.coin: (now-s.origin).total_seconds()/86400 > freshness for s in latest}
    forecasts = _forecast_rows(latest, latest_prediction, residual_bounds, staleness, data_kind)
    # Prepare every artifact before replacing any existing run files. Inference or
    # report errors must not mix a new checkpoint with an old report/forecast.
    emit("stage", stage="saving")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".forecast-stage-", dir=run_dir.parent) as temporary:
        stage = Path(temporary)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "metadata": metadata}, stage / "model.pt")
        _write_json(stage / "summary.json", summary)
        _write_csv(stage / "validation_predictions.csv", _prediction_rows(validation_samples, validation_prediction, residual_bounds))
        _write_csv(stage / "test_predictions.csv", _prediction_rows(test_samples, test_prediction, residual_bounds))
        _write_csv(stage / "forecasts.csv", forecasts)
        _write_json(stage / "forecasts.json", forecasts)
        write_report(stage, summary, forecasts)
        run_dir.mkdir(parents=True, exist_ok=True)
        for artifact in stage.iterdir():
            os.replace(artifact, run_dir / artifact.name)
    progress(f"Saved frozen model, evaluation, and report to {run_dir.resolve()}")
    return summary


def predict(run_dir, data_dir, coins=None, *, allow_stale=False, max_age_days=None,
            as_of=None, update_report=True):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata.get("format_version") in (3, 4):
        from .kronos_backend import predict_kronos
        return predict_kronos(run_dir, data_dir, coins, allow_stale=allow_stale,
                              max_age_days=max_age_days, as_of=as_of,
                              update_report=update_report, checkpoint=checkpoint)
    version = metadata.get("format_version")
    multi = version == 2
    feature_names = MTF_FEATURE_NAMES if multi else FEATURE_NAMES
    if (version not in (1, 2) or metadata["features"] != list(feature_names)
            or metadata["horizons"] != list(HORIZONS)
            or (multi and (metadata.get("input_mode") != "multiframe" or metadata.get("timeframes") != list(TIMEFRAMES)))):
        raise ValueError("Model artifact format or feature/horizon definition is incompatible.")
    max_age_days = (8/24 if multi else 2) if max_age_days is None else max_age_days
    if not np.isfinite(max_age_days) or max_age_days < 0:
        raise ValueError("max_age_days must be finite and nonnegative.")
    trained_coins = metadata["coins"]
    coins = [c.upper() for c in (coins if coins is not None else trained_coins)]
    if not coins or len(set(coins)) != len(coins) or not set(coins).issubset(trained_coins):
        raise ValueError(f"Select unique trained coins. This model supports {trained_coins}.")
    # Older frozen GRU artifacts predate the model selector and omit this field.
    model_type = metadata.get("model_type", metadata["config"].get("model_type", "gru"))
    if not isinstance(model_type, str) or model_type not in MODEL_SPECS:
        raise ValueError("Model artifact contains an unsupported model_type.")
    config_model_type = metadata["config"].get("model_type", model_type)
    if config_model_type != model_type:
        raise ValueError("Model artifact and configuration disagree on model_type.")
    model = build_model(len(feature_names), len(trained_coins), metadata["config"]["hidden_size"],
                        input_mode="multiframe" if multi else "daily", model_type=model_type)
    model.load_state_dict(checkpoint["state_dict"])
    x_scaler, y_scaler = (Normalizer.from_dict(metadata[k]) for k in ("x_scaler", "y_scaler"))
    bounds = np.asarray(metadata["residual_bounds"])
    now = pd.Timestamp.now(tz="UTC")
    effective_time = min(_utc(as_of), now) if as_of is not None else now
    frames = (load_multiframe_data if multi else load_data)(data_dir, coins, as_of=effective_time)
    if multi:
        cadence = metadata["origin_interval"]
        samples = [inference_multiframe_sample(c, frames[c], metadata["config"]["lookback"], cadence) for c in coins]
    else:
        samples = [inference_sample(c, frames[c], metadata["config"]["lookback"]) for c in coins]
    staleness = {s.coin: (now-s.origin).total_seconds()/86400 > max_age_days for s in samples}
    if any(staleness.values()) and not allow_stale:
        stale = [c for c, is_stale in staleness.items() if is_stale]
        label = "multiframe" if multi else "daily"
        raise ValueError(f"Stale {label} data for {stale}; refresh candles or use --allow-stale for historical research.")
    estimates = _predict(model, samples, {c: i for i, c in enumerate(trained_coins)},
                         x_scaler, y_scaler, torch.device("cpu"))
    forecasts = _forecast_rows(samples, estimates, bounds, staleness, metadata["data_kind"])
    _write_csv(run_dir / "forecasts.csv", forecasts)
    _write_json(run_dir / "forecasts.json", forecasts)
    if update_report:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        write_report(run_dir, summary, forecasts)
    return forecasts


def _forecast_rows(samples, estimates, bounds, staleness, data_kind, *, horizons=HORIZONS):
    forecasts = []
    for sample, values in zip(samples, estimates):
        for j, horizon in enumerate(horizons):
            forecasts.append({"coin": sample.coin, "as_of": sample.origin.isoformat(),
                              "target_date": (sample.origin+pd.Timedelta(days=horizon)).isoformat(),
                              "horizon_days": horizon, "last_price_usdt": sample.price,
                              "forecast_price_usdt": float(sample.price*np.exp(values[j])),
                              "change_pct": float(100*np.expm1(values[j])),
                              "lower_price_usdt": float(sample.price*np.exp(values[j]+bounds[0, j])),
                              "upper_price_usdt": float(sample.price*np.exp(values[j]+bounds[1, j])),
                              "is_stale": staleness[sample.coin], "data_kind": data_kind})
    return forecasts

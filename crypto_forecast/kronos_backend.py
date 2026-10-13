"""Independent or sequential Kronos tuning on local OHLCV candles.

Kronos reads one candle resolution. Its tokenizer is trained first, then frozen
for predictor training; either stage uses the same purged historical splits.
Predictor checkpoints include their matched tokenizer; tokenizer jobs save independently.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from .learning_rate import LearningRateSchedule

from . import HORIZONS, __version__
from .data import _coins, _multiframe_provenance, _normalize, load_data
from .timeframes import interval_delta
from .kronos_preparation import KronosSample, plan_windows, materialize, materialize_task, watch_parent


RAW_FEATURES = ("open", "high", "low", "close", "volume", "amount")
MAX_CONTEXT = 512
CLIP = 5.0
MODEL_FIELDS = ("s1_bits", "s2_bits", "n_layers", "d_model", "n_heads", "ff_dim",
                "ffn_dropout_p", "attn_dropout_p", "resid_dropout_p", "token_dropout_p", "learn_te")
TOKENIZER_FIELDS = ("d_in", "d_model", "n_heads", "ff_dim", "n_enc_layers", "n_dec_layers",
                    "ffn_dropout_p", "attn_dropout_p", "resid_dropout_p", "s1_bits", "s2_bits",
                    "beta", "gamma0", "gamma", "zeta", "group_size")


def _upstream():
    try:
        from ._vendor.kronos import Kronos, KronosTokenizer
        from ._vendor.kronos.kronos import auto_regressive_inference
    except ImportError as error:
        raise ValueError("Kronos dependencies are unavailable. Install the project's Kronos dependencies "
                         "before selecting Kronos.") from error
    return Kronos, KronosTokenizer, auto_regressive_inference


def load_pretrained(config, device):
    """Load official pretrained modules or user-selected local Hub folders."""
    Kronos, KronosTokenizer, _ = _upstream()
    try:
        model = Kronos.from_pretrained(config.kronos_model_path).to(device)
        tokenizer = KronosTokenizer.from_pretrained(config.kronos_tokenizer_path).to(device)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Cannot load Kronos pretrained weights. Download the selected Kronos predictor and its base tokenizer "
                         "first, or supply valid local model/tokenizer folders. " + str(error)) from error
    _freeze_tokenizer(model, tokenizer)
    return model, tokenizer


def load_tokenizer_pretrained(config, device):
    """Tokenizer jobs do not allocate a predictor or require its weight files."""
    _, Tokenizer, _ = _upstream()
    try:
        return Tokenizer.from_pretrained(config.kronos_tokenizer_path).to(device)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Cannot load Kronos pretrained tokenizer: " + str(error)) from error


def load_predictor_pretrained(config, device):
    """A self-contained saved tokenizer needs only the original predictor weights."""
    Model, _, _ = _upstream()
    try:
        return Model.from_pretrained(config.kronos_model_path).to(device)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Cannot load Kronos pretrained predictor: " + str(error)) from error


def _freeze_tokenizer(model, tokenizer):
    if tokenizer.d_in != len(RAW_FEATURES):
        raise ValueError("Kronos requires a six-channel OHLCV/amount tokenizer.")
    if (model.s1_bits, model.s2_bits) != (tokenizer.s1_bits, tokenizer.s2_bits):
        raise ValueError("Kronos predictor and tokenizer token vocabularies do not match.")
    tokenizer.eval()
    tokenizer.requires_grad_(False)


def _constructor_configs(model, tokenizer):
    model_config = {name: getattr(model, name) for name in MODEL_FIELDS}
    return model_config, _tokenizer_constructor_config(tokenizer)


def _tokenizer_constructor_config(tokenizer):
    tokenizer_config = {name: getattr(tokenizer, name) for name in TOKENIZER_FIELDS
                        if name not in {"n_enc_layers", "n_dec_layers", "beta", "gamma0", "gamma", "zeta", "group_size"}}
    tokenizer_config.update(n_enc_layers=tokenizer.enc_layers, n_dec_layers=tokenizer.dec_layers)
    quantizer = tokenizer.tokenizer.bsq
    tokenizer_config.update({name: getattr(quantizer, name)
                             for name in ("beta", "gamma0", "gamma", "zeta", "group_size")})
    return tokenizer_config


def load_candles(data_dir, coins, timeframe, *, as_of=None):
    """Load a selected stream, including existing daily compatibility CSVs.

    Supplemental CSVs may omit metadata; those are identified as user supplied.
    Supplied metadata is checked using the project's usual source/unit rules.
    """
    from .pipeline import _utc
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else _utc(as_of)
    directory = Path(data_dir)
    frames = {}
    for coin in _coins(coins):
        path = directory / timeframe / f"{coin}.csv"
        if timeframe == "1d" and not path.exists():
            # Legacy daily runs accept minimal source metadata; retain those
            # loader rules when using their original root-level CSV layout.
            frame = load_data(directory, [coin], as_of=cutoff)[coin]
            frame.attrs["interval"] = "1d"
            frames[coin] = frame
            continue
        if not path.exists():
            raise ValueError(f"{coin}: missing {timeframe} candles at {path}. Download candles or supply this CSV.")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        frame = _normalize(pd.read_csv(io.BytesIO(payload)), coin, cutoff, timeframe)
        if frame.empty:
            raise ValueError(f"{coin}: no completed {timeframe} candles available")
        metadata = (_multiframe_provenance(path, coin, timeframe, digest)
                    if path.with_suffix(".metadata.json").exists() else
                    {"source": "local_csv_user_supplied", "coin": coin, "quote_currency": "USDT",
                     "interval": timeframe, "timestamp_semantics": "UTC candle open"})
        frame.attrs.update(coin=coin, interval=timeframe, data_sha256=digest, provenance=metadata)
        frames[coin] = frame
    return frames


def _bars(timeframe, prediction_length=None):
    if prediction_length is not None:
        return np.arange(1, prediction_length + 1, dtype=np.int64)
    step = interval_delta(timeframe)
    return np.asarray([int(pd.Timedelta(days=h) / step) for h in HORIZONS], dtype=np.int64)


def _horizons(config):
    if config.kronos_prediction_length is None:
        return list(HORIZONS)
    days_per_bar = interval_delta(config.kronos_timeframe) / pd.Timedelta(days=1)
    return (np.arange(1, config.kronos_prediction_length + 1) * days_per_bar).tolist()


def _candle_rows(rows, config):
    """Keep day-based fields compatible while labeling each candle explicitly."""
    if config.kronos_prediction_length is not None:
        days_per_bar = interval_delta(config.kronos_timeframe) / pd.Timedelta(days=1)
        for row in rows:
            row.update(horizon_candles=round(row["horizon_days"] / days_per_bar),
                       timeframe=config.kronos_timeframe)
    return rows


def _raw(frame):
    values = frame.loc[:, RAW_FEATURES[:-1]].to_numpy(dtype=np.float64)
    # Candle files contain base volume, not actual traded quote turnover.
    return np.column_stack((values, np.zeros(len(values), dtype=np.float64)))


def _make_sample(coin, frame, values, index, lookback, timeframe, *, with_targets, prediction_length=None):
    bars = _bars(timeframe, prediction_length)
    step = interval_delta(timeframe)
    origin = pd.Timestamp(frame.timestamp.iloc[index]) + step
    price = float(values[index, 3])
    drift_bars = int(pd.Timedelta(days=7) / step) if prediction_length is None else lookback - 1
    drift = (np.log(price / values[index - drift_bars, 3]) / max(1, drift_bars) * bars).astype(np.float32)
    future = values[index + 1:index + 1 + int(bars[-1])].copy() if with_targets else np.empty((0, 6))
    timestamps = pd.DatetimeIndex(frame.timestamp.iloc[index - lookback + 1:index + 1])
    future_timestamps = pd.date_range(origin, periods=int(bars[-1]), freq=step)
    y = (np.log(values[index + bars, 3] / price).astype(np.float32) if with_targets
         else np.full(len(bars), np.nan, dtype=np.float32))
    return KronosSample(values[index - lookback + 1:index + 1].copy(), future, timestamps,
                        future_timestamps, y, coin, origin, origin + step * int(bars[-1]), price, drift)


def build_samples(frames, lookback, timeframe, *, prediction_length=None, emit=None):
    """Build raw, contiguous inputs with targets at exact closing boundaries."""
    bars = _bars(timeframe, prediction_length)
    step = interval_delta(timeframe)
    week = int(pd.Timedelta(days=7) / step) if prediction_length is None else lookback - 1
    first_index = max(lookback - 1, week)
    total = sum(max(0, len(frame) - int(bars[-1]) - first_index) for frame in frames.values())
    completed, last_update = 0, time.monotonic()
    samples = []
    for coin, frame in frames.items():
        if emit:
            emit("preparation", stage="preparing", preparation_phase="windows",
                 preparation_completed=completed, preparation_total=total, coin=coin,
                 usable_samples=len(samples))
        values = _raw(frame)
        gaps = np.concatenate(([0], np.cumsum(frame.timestamp.diff().iloc[1:] != step)))
        stop = len(frame) - int(bars[-1])
        for index in range(first_index, stop):
            first = min(index - lookback + 1, index - week)
            if gaps[index + int(bars[-1])] == gaps[first]:
                samples.append(_make_sample(coin, frame, values, index, lookback, timeframe,
                                            with_targets=True, prediction_length=prediction_length))
            completed += 1
            if emit:
                now = time.monotonic()
                if index == stop - 1 or now - last_update >= 1:
                    emit("preparation", stage="preparing", preparation_phase="windows",
                         preparation_completed=completed, preparation_total=total, coin=coin,
                         usable_samples=len(samples))
                    last_update = now
    if emit:
        emit("preparation", stage="preparing", preparation_phase="sorting",
             preparation_completed=0, preparation_total=0)
    return sorted(samples, key=lambda sample: (sample.origin, sample.coin))


def inference_sample(coin, frame, lookback, timeframe, *, prediction_length=None):
    step = interval_delta(timeframe)
    required = (max(lookback, int(pd.Timedelta(days=7) / step) + 1)
                if prediction_length is None else lookback)
    if len(frame) < required:
        raise ValueError(f"{coin} needs at least {required} completed {timeframe} candles for inference")
    if (frame.timestamp.iloc[-required:].diff().iloc[1:] != step).any():
        raise ValueError(f"{coin} latest {timeframe} inference history has missing candles")
    return _make_sample(coin, frame, _raw(frame), len(frame) - 1, lookback, timeframe,
                        with_targets=False, prediction_length=prediction_length)


def normalize_sample(sample):
    """Fit each normalization solely to this sample's historical input bars."""
    mean = sample.x.mean(axis=0)
    scale = sample.x.std(axis=0) + 1e-5
    context = np.clip((sample.x - mean) / scale, -CLIP, CLIP).astype(np.float32)
    future = np.clip((sample.future - mean) / scale, -CLIP, CLIP).astype(np.float32)
    return context, future, mean, scale


def _stamps(index):
    return np.column_stack((index.minute, index.hour, index.dayofweek, index.day, index.month)).astype(np.float32)


def select_stride(samples, stride):
    """Apply a per-coin origin stride after assigning chronological partitions."""
    counts = {}
    selected = []
    for sample in samples:
        number = counts.get(sample.coin, 0)
        if number % stride == 0:
            selected.append(sample)
        counts[sample.coin] = number + 1
    return selected


def prepare_training_samples(frames, config, emit=None):
    """Plan all origins, then materialize sampled windows with bounded CPU work."""
    bars = _bars(config.kronos_timeframe, config.kronos_prediction_length)
    step = interval_delta(config.kronos_timeframe)
    week = int(pd.Timedelta(days=7) / step) if config.kronos_prediction_length is None else config.lookback - 1
    partitions, info = plan_windows(frames, config.lookback, bars, step, week,
                                    config.train_fraction, config.validation_fraction, emit)
    if emit:
        emit("preparation", stage="preparing", preparation_phase="eligibility")
    excluded, latest = {}, {}
    for coin in config.coins:
        counts = {name: len(values[coin]) for name, values in partitions.items()}
        if counts["train"] < 32 or counts["validation"] < 5 or counts["test"] < 5:
            excluded[coin] = f"Insufficient contiguous samples in purged splits: {counts}; need 32/5/5."
        else:
            try:
                latest[coin] = inference_sample(coin, frames[coin], config.lookback, config.kronos_timeframe,
                                                prediction_length=config.kronos_prediction_length)
            except ValueError as error:
                excluded[coin] = str(error)
    coins = [coin for coin in config.coins if coin not in excluded]
    if not coins:
        raise ValueError("No coin has enough history for purged Kronos training/validation/test. " + str(excluded))
    info["eligible_counts"] = {name: {coin: len(values[coin]) for coin in coins}
                               for name, values in partitions.items()}
    if emit:
        emit("preparation", stage="preparing", preparation_phase="sampling")
    indices = {name: {coin: values[coin][::config.kronos_train_stride if name == "train"
                                      else config.kronos_eval_stride] for coin in coins}
               for name, values in partitions.items()}
    info["sampled_counts"] = {name: sum(map(len, values.values())) for name, values in indices.items()}
    info["sampling"] = {"after_split": True, "per_coin": True,
                        "train_stride": config.kronos_train_stride, "eval_stride": config.kronos_eval_stride}
    total = sum(info["sampled_counts"].values())
    requested = config.kronos_preparation_workers
    workers = min(requested or min(4, os.cpu_count() or 1), len(coins))
    # Benchmarks on 13 coins / 22,769 selected windows favor serial preparation;
    # avoid Windows spawn and result-transfer overhead at that workload size.
    if requested == 0 and total < 50000:
        workers = 1
    selected = {name: [] for name in indices}
    completed, last_progress = 0, 0.0

    def progress(coin=None, *, force=False):
        nonlocal last_progress
        now = time.monotonic()
        if emit and (force or now - last_progress >= 1):
            emit("preparation", stage="preparing", preparation_phase="materializing",
                 preparation_completed=completed, preparation_total=total,
                 preparation_workers=workers, coin=coin)
            last_progress = now

    progress(force=True)
    if workers == 1:
        for coin in coins:
            values = _raw(frames[coin])
            timestamps = pd.DatetimeIndex(frames[coin].timestamp)
            for name, by_coin in indices.items():
                for start in range(0, len(by_coin[coin]), 256):
                    block = materialize(coin, values, timestamps, by_coin[coin][start:start + 256],
                                        config.lookback, bars, step, week)
                    selected[name].extend(block)
                    completed += len(block)
                    progress(coin)
    else:
        with tempfile.TemporaryDirectory(prefix="kronos-preparation-") as directory:
            paths = {}
            for number, coin in enumerate(coins):
                values_path = Path(directory) / f"{number}-values.npy"
                stamps_path = Path(directory) / f"{number}-timestamps.npy"
                np.save(values_path, _raw(frames[coin]))
                np.save(stamps_path, pd.DatetimeIndex(frames[coin].timestamp).as_unit("ns").asi8)
                paths[coin] = values_path, stamps_path

            def tasks():
                for coin in coins:
                    for name, by_coin in indices.items():
                        for start in range(0, len(by_coin[coin]), 256):
                            yield (coin, *paths[coin], name, by_coin[coin][start:start + 256],
                                   config.lookback, bars, step, week)

            pending_tasks = iter(tasks())
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=watch_parent)
            pending = {}
            try:
                for _ in range(workers):
                    task = next(pending_tasks, None)
                    if task is not None:
                        pending[pool.submit(materialize_task, task)] = task[0]
                while pending:
                    done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    if not done:
                        progress()
                    for future in done:
                        coin = pending.pop(future)
                        name, block = future.result()
                        selected[name].extend(block)
                        completed += len(block)
                        progress(coin)
                        task = next(pending_tasks, None)
                        if task is not None:
                            pending[pool.submit(materialize_task, task)] = task[0]
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    for values in selected.values():
        values.sort(key=lambda sample: (sample.origin, sample.coin))
    progress(force=True)
    return selected, info, excluded, latest


class _TokenizerDataset(Dataset):
    """Normalize windows lazily without duplicating the full candle dataset."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        context, future, _, _ = normalize_sample(self.samples[index])
        return torch.from_numpy(np.concatenate((context, future)))


def _tokenizer_loss(tokenizer, candles):
    (coarse, full), bsq_loss, _, _ = tokenizer(candles)
    coarse_mse = F.mse_loss(coarse, candles)
    full_mse = F.mse_loss(full, candles)
    return (coarse_mse + full_mse + bsq_loss) / 2, full_mse


def _tokenizer_validation_loss(tokenizer, samples, device, batch_size):
    """Select the tokenizer by full reconstruction MSE, as in upstream."""
    tokenizer.eval()
    values = []
    with torch.no_grad():
        for candles in DataLoader(_TokenizerDataset(samples), batch_size=batch_size):
            candles = candles.to(device)
            (_, reconstructed), _, _, _ = tokenizer(candles)
            values.extend((reconstructed - candles).square().mean(dim=(1, 2)).cpu().tolist())
    # Match the predictor's equal-coin objective even with unequal history sizes.
    return float(np.mean([np.mean([v for v, s in zip(values, samples) if s.coin == coin])
                          for coin in sorted({s.coin for s in samples})]))


def _train_tokenizer(tokenizer, train_samples, validation_samples, config, device, emit, progress,
                     on_checkpoint=None):
    """Restore the best tokenizer before freezing and preparing predictor tokens."""
    initial_hash = _state_hash(tokenizer)
    tokenizer.requires_grad_(True)
    counts = {coin: sum(s.coin == coin for s in train_samples) for coin in {s.coin for s in train_samples}}
    sampler = WeightedRandomSampler([1 / counts[s.coin] for s in train_samples], len(train_samples),
                                    generator=torch.Generator().manual_seed(config.seed))
    batch_size = config.kronos_tokenizer_batch_size or config.batch_size
    loader = DataLoader(_TokenizerDataset(train_samples), batch_size=batch_size, sampler=sampler)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=config.kronos_tokenizer_learning_rate,
                                 betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = LearningRateSchedule(optimizer, config, config.kronos_tokenizer_epochs, len(loader), "kronos_tokenizer_")
    best_loss, best_epoch, best_state, history = float("inf"), 0, None, []
    emit("stage", stage="training_tokenizer", epoch=0, max_epochs=config.kronos_tokenizer_epochs,
         best_epoch=0, best_validation_loss=None, batch=0, total_batches=len(loader), train_loss=None,
         training_component="tokenizer", device=str(device), learning_rate=scheduler.rate)
    progress("Fine-tuning tokenizer with coarse/full reconstruction and BSQ loss.")
    for epoch in range(1, config.kronos_tokenizer_epochs + 1):
        tokenizer.train()
        total, seen, last_batch_event, rate_sum = 0.0, 0, time.monotonic(), 0.0
        for batch_number, candles in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _tokenizer_loss(tokenizer, candles.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite Kronos tokenizer training loss; inspect candles and weights.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 2.0)
            optimizer.step()
            scheduler.batch_finished()
            rate_sum += scheduler.last_used
            total += float(loss.detach().cpu()) * len(candles)
            seen += len(candles)
            batch_time = time.monotonic()
            if batch_number in (1, len(loader)) or batch_time - last_batch_event >= 1:
                emit("tokenizer_batch", stage="training_tokenizer", epoch=epoch,
                     max_epochs=config.kronos_tokenizer_epochs, batch=batch_number,
                     total_batches=len(loader), train_loss=total / seen, learning_rate=scheduler.last_used)
                last_batch_event = batch_time
        validation_loss = _tokenizer_validation_loss(tokenizer, validation_samples, device, batch_size)
        if not np.isfinite(validation_loss):
            raise ValueError("Non-finite Kronos tokenizer validation loss; inspect candles and weights.")
        scheduler.epoch_finished(validation_loss)
        history.append({"epoch": epoch, "train_loss": total / seen, "validation_loss": validation_loss,
                        "learning_rate": rate_sum / len(loader), "lr_end": scheduler.last_used, "lr_next": scheduler.rate})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in tokenizer.state_dict().items()}
            if on_checkpoint:
                on_checkpoint(best_state, _tokenizer_training_info(best_epoch, best_loss, history,
                              config, batch_size, _state_hash(tokenizer) != initial_hash), False)
        emit("tokenizer_epoch", stage="training_tokenizer", **history[-1],
             max_epochs=config.kronos_tokenizer_epochs, best_epoch=best_epoch, best_validation_loss=best_loss)
        progress(f"Tokenizer epoch {epoch}/{config.kronos_tokenizer_epochs}: train={total / seen:.6f}, "
                 f"validation reconstruction MSE={validation_loss:.6f}; lr={scheduler.last_used:.6g}; best={best_epoch}")
        if epoch - best_epoch >= config.kronos_tokenizer_patience:
            break
    tokenizer.load_state_dict(best_state)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    training = _tokenizer_training_info(best_epoch, best_loss, history, config, batch_size,
                                       _state_hash(tokenizer) != initial_hash)
    if on_checkpoint:
        on_checkpoint(best_state, training, True)
    return training


def _tokenizer_training_info(best_epoch, best_loss, history, config, batch_size, changed):
    return {"enabled": True, "best_epoch": best_epoch, "best_validation_loss": best_loss, "history": list(history),
            "parameters_changed": changed, "batch_size": batch_size,
            "objective": "(Coarse reconstruction MSE + full reconstruction MSE + BSQ loss) / 2",
            "validation_objective": "Full reconstruction MSE, averaged equally across coins.",
            "learning_rate": config.kronos_tokenizer_learning_rate,
            "epochs_completed": len(history)}


def _token_dataset(samples, tokenizer, device, batch_size, emit=None, *, partition=None):
    tokens_one, tokens_two, stamps = [], [], []
    tokenizer.eval()
    if emit:
        emit("tokenizing", stage="preparing", preparation_phase="tokenizing",
             preparation_completed=0, preparation_total=len(samples), preparation_partition=partition,
             prepared_samples=0, total_samples=len(samples))
    for start in range(0, len(samples), batch_size):
        block = samples[start:start + batch_size]
        raw = []
        for sample in block:
            context, future, _, _ = normalize_sample(sample)
            raw.append(np.concatenate((context, future)))
        with torch.no_grad():
            one, two = tokenizer.encode(torch.tensor(np.stack(raw), device=device), half=True)
        tokens_one.append(one.cpu())
        tokens_two.append(two.cpu())
        stamps.append(torch.tensor(np.stack([_stamps(s.x_timestamps.append(s.future_timestamps)) for s in block])))
        if emit:
            emit("tokenizing", stage="preparing", prepared_samples=min(start + len(block), len(samples)),
                 total_samples=len(samples), preparation_phase="tokenizing",
                 preparation_completed=min(start + len(block), len(samples)),
                 preparation_total=len(samples), preparation_partition=partition)
    return TensorDataset(torch.cat(tokens_one), torch.cat(tokens_two), torch.cat(stamps))


@contextmanager
def _causal_validation(model):
    """Prevent upstream eval cross-attention from seeing later teacher tokens.

    Upstream dependency cross-attention toggles its causal mask with ``training``.
    Its default dropout is zero, so enable that mask while keeping the rest of
    the model in evaluation mode. Temporarily zero its dropout for local models.
    """
    attention = model.dep_layer.cross_attn
    previous_training, previous_dropout = attention.training, attention.attn_dropout_p
    attention.training, attention.attn_dropout_p = True, 0.0
    try:
        yield
    finally:
        attention.training, attention.attn_dropout_p = previous_training, previous_dropout


def _batch_loss(model, batch, device, lookback):
    one, two, stamps = (value.to(device) for value in batch)
    logits_one, logits_two = model(one[:, :-1], two[:, :-1], stamps[:, :-1],
                                   use_teacher_forcing=True, s1_targets=one[:, 1:])
    # The last context token predicts the first future bar; earlier history is
    # context, never a target contributing to this forecasting objective.
    first = lookback - 1
    losses = []
    for logits, target in ((logits_one, one[:, 1:]), (logits_two, two[:, 1:])):
        logits, target = logits[:, first:], target[:, first:]
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                      reduction="none").reshape(target.shape).mean(dim=1))
    return (losses[0] + losses[1]) / 2


def _validation_loss(model, dataset, samples, device, lookback, batch_size):
    model.eval()
    values = []
    with torch.no_grad(), _causal_validation(model):
        for batch in DataLoader(dataset, batch_size=batch_size):
            values.extend(_batch_loss(model, batch, device, lookback).cpu().tolist())
    per_coin = [np.mean([value for value, sample in zip(values, samples) if sample.coin == coin])
                for coin in sorted({s.coin for s in samples})]
    result = float(np.mean(per_coin))
    if not np.isfinite(result):
        raise ValueError("Non-finite Kronos validation loss; inspect candle data and weights.")
    return result


def _sample_seed(seed, sample):
    # Forecasts reconstruct exactly even if the caller chooses a coin subset.
    identity = f"{seed}:{sample.coin}:{sample.origin.isoformat()}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def forecast_samples(model, tokenizer, samples, config, device, emit=None):
    _, _, generate = _upstream()
    model.eval()
    tokenizer.eval()
    bars = _bars(config.kronos_timeframe, config.kronos_prediction_length)
    estimates = []
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    # A separate RNG per origin makes report refreshes and subsets reproducible.
    for index, sample in enumerate(samples, 1):
        context, _, mean, scale = normalize_sample(sample)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(_sample_seed(config.seed, sample))
            generated = generate(tokenizer, model,
                                 torch.tensor(context[None], device=device),
                                 torch.tensor(_stamps(sample.x_timestamps)[None], device=device),
                                 torch.tensor(_stamps(sample.future_timestamps)[None], device=device),
                                 MAX_CONTEXT, int(bars[-1]), clip=CLIP, T=1.0, top_k=0, top_p=0.9,
                                 sample_count=config.kronos_sample_count, verbose=False)
        future = generated[0, -int(bars[-1]):] * scale + mean
        closes = future[bars - 1, 3]
        if not np.isfinite(closes).all() or (closes <= 0).any():
            raise ValueError(f"Kronos produced non-positive or non-finite predicted closes for {sample.coin}.")
        estimates.append(np.log(closes / sample.price))
        if emit:
            emit("forecast", completed_samples=index, total_samples=len(samples))
    return np.asarray(estimates)


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_kronos(config, *, overwrite=False, data_kind="historical", progress=print,
                 on_event=None, on_training_decision=None):
    from .pipeline import (_device, _forecast_rows, _manifest, _metrics, _prediction_rows,
                           _utc, _write_csv, _write_json)
    from .report import write_report
    from .kronos_artifacts import (data_signature, load_tokenizer_checkpoint,
                                  save_tokenizer, tokenizer_summary)
    config.validate()
    if on_training_decision is not None and not callable(on_training_decision):
        raise ValueError("on_training_decision must be a callable or None.")
    if data_kind == "synthetic":
        data_kind = "synthetic_demo"
    if data_kind not in {"historical", "synthetic_demo"}:
        raise ValueError("data_kind must be historical or synthetic_demo.")
    run_dir = Path(config.run_dir)
    if any((run_dir / name).exists() for name in ("model.pt", "tokenizer.pt")) and not overwrite:
        raise ValueError(f"{run_dir} already contains a trained run. Select another --run-dir or use --overwrite.")
    if config.kronos_tokenizer_run_dir and Path(config.kronos_tokenizer_run_dir).resolve() == run_dir.resolve():
        raise ValueError("Predictor training must use a new run directory to preserve its source tokenizer.")
    started = time.monotonic()

    def emit(kind, **values):
        if on_event:
            on_event({"kind": kind, **values, "elapsed_seconds": time.monotonic() - started})

    emit("stage", stage="loading")
    now = pd.Timestamp.now(tz="UTC")
    end = min(_utc(config.end), now) if config.end else now
    frames = load_candles(config.data_dir, config.coins, config.kronos_timeframe, as_of=end)
    start = _utc(config.start)
    frames = {coin: frame[(frame.timestamp >= start) & (frame.timestamp < end)].copy()
              for coin, frame in frames.items()}
    if any("synthetic" in str(frame.attrs["provenance"].get("source", "")).lower() for frame in frames.values()):
        data_kind = "synthetic_demo"
    emit("stage", stage="preparing")
    progress(f"Preparing raw {config.kronos_timeframe} candles for official {config.model_type.capitalize()} fine-tuning.")
    horizons = _horizons(config)
    selected, split_info, excluded, latest_samples = prepare_training_samples(
        frames, config, emit=emit if on_event else None)
    coins = [c for c in config.coins if c not in excluded]
    manifest = {coin: {"timeframes": {config.kronos_timeframe: _manifest(frames[coin])}} for coin in coins}
    signature = data_signature(config, manifest, split_info)
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
    emit("stage", stage="loading_model", counts=split_info["sampled_counts"])
    tokenizer_only = config.kronos_training_stage == "tokenizer"
    source = None
    if tokenizer_only:
        progress("Loading pretrained Kronos tokenizer for independent tuning.")
        tokenizer = load_tokenizer_pretrained(config, device)
        tokenizer_config = _tokenizer_constructor_config(tokenizer)
        if tokenizer.d_in != len(RAW_FEATURES):
            raise ValueError("Kronos requires a six-channel OHLCV/amount tokenizer.")
    else:
        if config.kronos_tokenizer_run_dir:
            state, constructor, source = load_tokenizer_checkpoint(config.kronos_tokenizer_run_dir, signature, RAW_FEATURES)
            _, Tokenizer, _ = _upstream()
            tokenizer = Tokenizer(**constructor).to(device)
            tokenizer.load_state_dict(state, strict=True)
            if source.get("tokenizer_sha256") and source["tokenizer_sha256"] != _state_hash(tokenizer):
                raise ValueError("Saved tokenizer weights do not match their checkpoint hash.")
            progress("Loading pretrained Kronos predictor with the selected saved tokenizer.")
            model = load_predictor_pretrained(config, device)
        else:
            progress("Loading pretrained Kronos predictor and base tokenizer.")
            model, tokenizer = load_pretrained(config, device)
        _freeze_tokenizer(model, tokenizer)
        model_config, tokenizer_config = _constructor_configs(model, tokenizer)
        initial_hash = _state_hash(model)
    train_samples, validation_samples, test_samples = (selected[name] for name in ("train", "validation", "test"))
    tokenizer_training = {"enabled": False}
    saved_tokenizer_summary = None

    def checkpoint_tokenizer(state, training, complete):
        nonlocal saved_tokenizer_summary
        saved_tokenizer_summary = tokenizer_summary(
            config, tokenizer_config, training, signature=signature, manifest=manifest, splits=split_info,
            coins=coins, state_hash=_state_hash(tokenizer), complete=complete, data_kind=data_kind)
        save_tokenizer(run_dir, state, tokenizer_config, saved_tokenizer_summary)

    if source is not None:
        tokenizer_training = {**source["tokenizer_training"], "reused": True,
                              "source_run": config.kronos_tokenizer_run_dir}
        progress(f"Loaded saved tokenizer from {config.kronos_tokenizer_run_dir}; its weights remain frozen.")
    elif tokenizer_only or (config.kronos_training_stage == "sequential" and config.kronos_finetune_tokenizer):
        tokenizer_training = _train_tokenizer(tokenizer, train_samples, validation_samples,
                                             config, device, emit, progress, on_checkpoint=checkpoint_tokenizer)
    if tokenizer_only:
        emit("stage", stage="saving", training_component="tokenizer")
        progress(f"Saved best tokenizer independently to {run_dir.resolve()}; predictor training can reuse it.")
        return saved_tokenizer_summary
    _freeze_tokenizer(model, tokenizer)
    emit("stage", stage="preparing", training_component="predictor", epoch=0, max_epochs=config.epochs,
         best_epoch=0, best_validation_loss=None, train_loss=None, validation_loss=None,
         batch=0, total_batches=0, learning_rate=None)
    train_dataset = _token_dataset(train_samples, tokenizer, device, config.batch_size, emit, partition="training")
    validation_dataset = _token_dataset(validation_samples, tokenizer, device, config.batch_size, emit, partition="validation")
    counts = {coin: sum(s.coin == coin for s in train_samples) for coin in coins}
    sampler = WeightedRandomSampler([1 / counts[s.coin] for s in train_samples], len(train_samples),
                                    generator=torch.Generator().manual_seed(config.seed))
    loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = LearningRateSchedule(optimizer, config, config.epochs, len(loader))
    history, best_loss, best_state, best_epoch, decisions = [], float("inf"), None, 0, []
    epoch, epoch_limit, extension_end = 0, config.epochs, None
    emit("stage", stage="training", counts=split_info["sampled_counts"], device=str(device),
         training_component="predictor", epoch=0, max_epochs=config.epochs, best_epoch=0,
         best_validation_loss=None, batch=0, total_batches=len(loader), train_loss=None, validation_loss=None,
         learning_rate=scheduler.rate)
    progress(f"Fine-tuning on {device}: {len(train_samples)} train / {len(validation_samples)} validation / "
             f"{len(test_samples)} test origins after stride sampling. Tokenizer stays frozen.")
    while epoch < epoch_limit:
        epoch += 1
        model.train()
        running_loss, seen, last_batch_event, rate_sum = 0.0, 0, time.monotonic(), 0.0
        for batch_number, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, batch, device, config.lookback).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite Kronos training loss; inspect data and learning rate.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.batch_finished()
            rate_sum += scheduler.last_used
            size = len(batch[0])
            running_loss += float(loss.detach().cpu()) * size
            seen += size
            batch_time = time.monotonic()
            if on_event and (batch_number in (1, len(loader)) or batch_time - last_batch_event >= 1):
                emit("batch", stage="training", epoch=epoch, max_epochs=epoch_limit, batch=batch_number,
                     total_batches=len(loader), train_loss=running_loss / seen, learning_rate=scheduler.last_used)
                last_batch_event = batch_time
        validation_loss = _validation_loss(model, validation_dataset, validation_samples, device,
                                           config.lookback, config.batch_size)
        scheduler.epoch_finished(validation_loss)
        history.append({"epoch": epoch, "train_loss": running_loss / seen, "validation_loss": validation_loss,
                        "learning_rate": rate_sum / len(loader), "lr_end": scheduler.last_used, "lr_next": scheduler.rate})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        emit("epoch", stage="training", **history[-1], max_epochs=epoch_limit, best_epoch=best_epoch,
             best_validation_loss=best_loss)
        progress(f"Epoch {epoch}/{epoch_limit}: token cross entropy train={running_loss / seen:.6f}, "
                 f"validation={validation_loss:.6f}; lr={scheduler.last_used:.6g}; best={best_epoch}")
        reason = ("extension_complete" if extension_end is not None and epoch >= extension_end else
                  "early_stop" if extension_end is None and epoch - best_epoch >= config.patience else None)
        if reason is not None:
            if on_training_decision is None:
                break
            details = {"decision_id": len(decisions) + 1, "reason": reason, "epoch": epoch,
                       "best_epoch": best_epoch, "best_validation_loss": best_loss,
                       "extension_epochs": 10, "max_epochs": epoch_limit,
                       "train_loss": history[-1]["train_loss"], "validation_loss": validation_loss}
            emit("decision", stage="awaiting_decision", **details)
            answer = on_training_decision(dict(details))
            if type(answer) is not bool:
                raise ValueError("Training decision callback must return True or False.")
            decisions.append({**details, "action": "continue" if answer else "finish"})
            if not answer:
                break
            extension_end = epoch + 10
            epoch_limit = extension_end
            emit("stage", stage="training", max_epochs=epoch_limit)
    model.load_state_dict(best_state)
    changed = _state_hash(model) != initial_hash
    emit("stage", stage="evaluating")
    progress("Evaluating the frozen best checkpoint on its selected future candle targets.")
    validation_prediction = forecast_samples(model, tokenizer, validation_samples, config, device, emit)
    residuals = np.stack([s.y for s in validation_samples]) - validation_prediction
    bounds = np.quantile(residuals, [0.1, 0.9], axis=0)
    # Final test is first forecast only after validation chooses the checkpoint.
    test_prediction = forecast_samples(model, tokenizer, test_samples, config, device, emit)
    metrics = {"validation": _metrics(validation_samples, validation_prediction, bounds, model_type=config.model_type, horizons=horizons),
               "test": _metrics(test_samples, test_prediction, bounds, model_type=config.model_type, horizons=horizons),
               "test_non_overlapping": _metrics(test_samples, test_prediction, bounds, True, config.model_type, horizons=horizons)}
    for rows in metrics.values():
        _candle_rows(rows, config)
    input_note = "The amount channel is zero because source CSVs do not contain observed quote turnover."
    kronos = {"timeframe": config.kronos_timeframe, "train_stride": config.kronos_train_stride,
              "eval_stride": config.kronos_eval_stride, "model_path": config.kronos_model_path,
              "tokenizer_path": config.kronos_tokenizer_path, "tokenizer_frozen": True,
               "tokenizer_finetuned": tokenizer_training.get("enabled", False),
              "tokenizer_source_run": config.kronos_tokenizer_run_dir,
              "max_context": MAX_CONTEXT, "sample_count": config.kronos_sample_count,
              "prediction_bars": _bars(config.kronos_timeframe, config.kronos_prediction_length).tolist(), "amount_note": input_note,
              "normalization": "Each input lookback only; future bars use the same historical mean/std.",
              "objective": "Mean next-token cross entropy of s1/s2 over future positions only.",
              "sampled_counts": split_info["sampled_counts"], "validation_causal": True,
              "forecast_device": str(device),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    interval_note = ("Empirical 80% validation-residual ranges, pooled across eligible coins per horizon. "
                     "Validation also selects the epoch, so these are not independent calibration or guaranteed "
                     "probabilities. Final-test coverage measures held-out behavior. Metrics use the saved "
                     "per-coin evaluation stride; non-overlapping test origins are reported separately.")
    metadata = {"format_version": 3 if config.kronos_prediction_length is None else 4, "package_version": __version__, "config": asdict(config),
                "model_type": config.model_type, "input_mode": "single_timeframe",
                "timeframes": [config.kronos_timeframe], "origin_interval": config.kronos_timeframe,
                "coins": coins, "features": list(RAW_FEATURES), "horizons": horizons,
                "residual_bounds": bounds.tolist(), "data_kind": data_kind, "kronos": kronos,
                "model_config": model_config, "tokenizer_config": tokenizer_config}
    metadata["tokenizer_training"] = tokenizer_training
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "package_version": __version__,
               "data_kind": data_kind, "coins": coins, "config": asdict(config), "model_type": config.model_type,
               "splits": split_info, "excluded_coins": excluded,
               "data_manifest": {coin: {"timeframes": {config.kronos_timeframe: _manifest(frames[coin])}} for coin in coins},
               "best_epoch": best_epoch, "history": history, "metrics": metrics, "interval_note": interval_note,
               "parameters_changed": changed, "torch_version": str(torch.__version__), "device": str(device),
               "deterministic_algorithms": True, "kronos": kronos,
               "training_control": {"initial_epochs": config.epochs, "epoch_limit": epoch_limit,
                                    "extension_epochs": 10, "epochs_completed": len(history), "decisions": decisions}}
    summary["tokenizer_training"] = tokenizer_training
    latest = [latest_samples[coin] for coin in coins]
    latest_prediction = forecast_samples(model, tokenizer, latest, config, device, emit)
    now = pd.Timestamp.now(tz="UTC")
    freshness = 2 * interval_delta(config.kronos_timeframe) / pd.Timedelta(days=1)
    staleness = {s.coin: (now - s.origin).total_seconds() / 86400 > freshness for s in latest}
    forecasts = _candle_rows(_forecast_rows(latest, latest_prediction, bounds, staleness, data_kind, horizons=horizons), config)
    emit("stage", stage="saving")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kronos-stage-", dir=run_dir.parent) as temporary:
        stage = Path(temporary)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "tokenizer_state_dict": {k: v.detach().cpu() for k, v in tokenizer.state_dict().items()},
                    "metadata": metadata}, stage / "model.pt")
        _write_json(stage / "summary.json", summary)
        _write_csv(stage / "validation_predictions.csv", _candle_rows(_prediction_rows(validation_samples, validation_prediction, bounds, horizons=horizons), config))
        _write_csv(stage / "test_predictions.csv", _candle_rows(_prediction_rows(test_samples, test_prediction, bounds, horizons=horizons), config))
        _write_csv(stage / "forecasts.csv", forecasts)
        _write_json(stage / "forecasts.json", forecasts)
        write_report(stage, summary, forecasts)
        run_dir.mkdir(parents=True, exist_ok=True)
        for artifact in stage.iterdir():
            os.replace(artifact, run_dir / artifact.name)
    progress(f"Saved self-contained {config.model_type.capitalize()} predictor, tokenizer, evaluation and report to {run_dir.resolve()}")
    return summary


def predict_kronos(run_dir, data_dir, coins=None, *, allow_stale=False, max_age_days=None,
                   as_of=None, update_report=True, checkpoint=None):
    from .pipeline import TrainingConfig, _device, _forecast_rows, _utc, _write_csv, _write_json
    from .report import write_report
    run_dir = Path(run_dir)
    if checkpoint is None:
        checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    saved_config = dict(metadata["config"])
    if metadata.get("format_version") == 3:
        saved_config.setdefault("kronos_prediction_length", None)
    config = TrainingConfig(**saved_config)
    config.validate()
    horizons = _horizons(config)
    if (metadata.get("format_version") not in (3, 4) or metadata.get("model_type") not in {"kronos-small", "kronos-base"}
            or config.model_type != metadata.get("model_type") or metadata.get("features") != list(RAW_FEATURES)
            or metadata.get("horizons") != horizons or metadata.get("input_mode") != "single_timeframe"
            or metadata.get("timeframes") != [config.kronos_timeframe]
            or metadata.get("origin_interval") != config.kronos_timeframe):
        raise ValueError("Kronos artifact format or candle/horizon definition is incompatible.")
    trained_coins = metadata["coins"]
    coins = _coins(coins if coins is not None else trained_coins)
    if not set(coins).issubset(trained_coins):
        raise ValueError(f"Select unique trained coins. This Kronos run supports {trained_coins}.")
    freshness = 2 * interval_delta(config.kronos_timeframe) / pd.Timedelta(days=1)
    max_age_days = freshness if max_age_days is None else max_age_days
    if not np.isfinite(max_age_days) or max_age_days < 0:
        raise ValueError("max_age_days must be finite and nonnegative.")
    now = pd.Timestamp.now(tz="UTC")
    cutoff = min(_utc(as_of), now) if as_of is not None else now
    frames = load_candles(data_dir, coins, config.kronos_timeframe, as_of=cutoff)
    samples = [inference_sample(c, frames[c], config.lookback, config.kronos_timeframe,
                                prediction_length=config.kronos_prediction_length) for c in coins]
    staleness = {s.coin: (now - s.origin).total_seconds() / 86400 > max_age_days for s in samples}
    if any(staleness.values()) and not allow_stale:
        stale = [coin for coin, value in staleness.items() if value]
        raise ValueError(f"Stale {config.kronos_timeframe} Kronos data for {stale}; refresh candles or use --allow-stale.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = _device(config.device)
    Kronos, KronosTokenizer, _ = _upstream()
    # Plain constructors and saved tensors; reconstructing never contacts Hub.
    model = Kronos(**metadata["model_config"]).to(device)
    tokenizer = KronosTokenizer(**metadata["tokenizer_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
    _freeze_tokenizer(model, tokenizer)
    bounds = np.asarray(metadata["residual_bounds"], dtype=np.float64)
    if bounds.shape != (2, len(horizons)) or not np.isfinite(bounds).all():
        raise ValueError("Invalid Kronos validation residual bounds.")
    prediction = forecast_samples(model, tokenizer, samples, config, device)
    forecasts = _candle_rows(_forecast_rows(samples, prediction, bounds, staleness, metadata["data_kind"], horizons=horizons), config)
    _write_csv(run_dir / "forecasts.csv", forecasts)
    _write_json(run_dir / "forecasts.json", forecasts)
    if update_report:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        write_report(run_dir, summary, forecasts)
    return forecasts

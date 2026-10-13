"""Read-only Kronos candle forecasting, independent of optional fine-tuning."""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .data import _coins, _normalize
from .kronos_backend import RAW_FEATURES, load_candles, load_pretrained, _upstream
from .timeframes import TIMEFRAMES, interval_delta
from .kronos_setup import PRETRAINED_MODELS

ORIGINAL_MODELS = {"original": "kronos-small", "original-base": "kronos-base"}


def settings(payload):
    defaults = dict(source="saved", coin="BTC", timeframe="4h", lookback=120,
                    horizon=24, model="original", device="auto", seed=7,
                    sample_count=1, temperature=1.0, top_p=0.9, end=None)
    if not isinstance(payload, dict) or set(payload) - set(defaults) - {"csv"}:
        raise ValueError("Unknown chart forecast setting.")
    value = {**defaults, **payload}
    if value["source"] not in ("saved", "csv") or value["timeframe"] not in TIMEFRAMES:
        raise ValueError("Choose saved candles or CSV and a supported candle interval.")
    value["coin"] = _coins([value["coin"]])[0]
    for key, low, high in (("lookback", 2, 512), ("horizon", 1, 512),
                           ("sample_count", 1, 10), ("seed", 0, 2**32 - 1)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}.")
    for key, low, high in (("temperature", 0.05, 2.0), ("top_p", 0.01, 1.0)):
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    if value["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Choose automatic, CPU or CUDA execution.")
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("Choose original weights or a saved Kronos run.")
    if value["end"] is not None:
        if not isinstance(value["end"], str):
            raise ValueError("Chart cutoff must be a UTC date/time.")
        cutoff = pd.Timestamp(value["end"])
        if pd.isna(cutoff):
            raise ValueError("Chart cutoff must be a UTC date/time.")
        value["end"] = (cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")).isoformat()
    if value["source"] == "csv":
        if not isinstance(value.get("csv"), str) or not 0 < len(value["csv"].encode()) <= 1_000_000:
            raise ValueError("Upload a CSV of at most 1 MB.")
    elif "csv" in value:
        raise ValueError("CSV content is only accepted for the uploaded CSV source.")
    return value


def _inside(workspace, relative):
    path = Path(workspace) / relative
    if not path.resolve().is_relative_to(Path(workspace).resolve()):
        raise ValueError("Chart data and models must stay inside this project.")
    return path


def rows(frame):
    return [dict(timestamp=pd.Timestamp(row.timestamp).isoformat(),
                 **{column: float(getattr(row, column)) for column in RAW_FEATURES[:-1]})
            for row in frame.itertuples(index=False)]


def saved_candles(workspace, value, cutoff):
    relative = Path("data") / value["timeframe"] / f"{value['coin']}.csv"
    if value["timeframe"] == "1d" and not (Path(workspace) / relative).exists():
        relative = Path("data") / f"{value['coin']}.csv"
    path = _inside(workspace, relative)
    _inside(workspace, path.with_suffix(".metadata.json").relative_to(workspace))
    return load_candles(Path(workspace) / "data", [value["coin"]], value["timeframe"], as_of=cutoff)[value["coin"]]


def testing_chart(workspace, payload):
    value = settings(payload)
    if value["source"] != "saved" or value["end"] is not None:
        raise ValueError("Model testing uses the full saved chart without a cutoff.")
    frame = saved_candles(workspace, value, pd.Timestamp.now(tz="UTC"))
    if frame.empty:
        raise ValueError("No completed candles are available for this coin and interval.")
    provenance = frame.attrs.get("provenance", {})
    return {"coin": value["coin"], "timeframe": value["timeframe"], "history": rows(frame),
            "forecast": [], "available_candles": len(frame),
            "source": provenance.get("source", "local_csv_user_supplied"),
            "synthetic": "synthetic" in str(provenance.get("source", "")).lower()}


def prepare_testing(workspace, payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("anchor"), str):
        raise ValueError("Select a completed candle on the price chart.")
    value = settings({key: item for key, item in payload.items() if key != "anchor"})
    if value["source"] != "saved" or value["end"] is not None:
        raise ValueError("Model testing uses saved candles and a clicked chart point.")
    anchor = pd.Timestamp(payload["anchor"])
    if pd.isna(anchor):
        raise ValueError("Select a valid candle opening time.")
    anchor = anchor.tz_localize("UTC") if anchor.tzinfo is None else anchor.tz_convert("UTC")
    frame = saved_candles(workspace, value, pd.Timestamp.now(tz="UTC"))
    if not frame.timestamp.eq(anchor).any():
        raise ValueError("The selected completed candle is not present in this chart. Reload the chart.")
    value["end"] = (anchor + interval_delta(value["timeframe"])).isoformat()
    _, chart = prepare_chart(workspace, value)
    forecast_end = pd.Timestamp(chart["forecast_end"])
    actual = frame.loc[(frame.timestamp >= pd.Timestamp(chart["origin"])) & (frame.timestamp < forecast_end)]
    chart.update(testing=True, selected_candle=anchor.isoformat(), actual=rows(actual))
    return value, chart


def prepare_chart(workspace, payload):
    value = settings(payload)
    now = pd.Timestamp.now(tz="UTC")
    cutoff = min(pd.Timestamp(value["end"]), now) if value["end"] else now
    if value["source"] == "saved":
        frame = saved_candles(workspace, value, cutoff)
        provenance = frame.attrs.get("provenance", {})
    else:
        raw = pd.read_csv(io.StringIO(value["csv"]))
        raw.columns = [str(name).strip().lower() for name in raw.columns]
        raw = raw.rename(columns={"timestamps": "timestamp"})
        if "volume" not in raw:
            raw["volume"] = 0.0
        frame = _normalize(raw, value["coin"], cutoff, value["timeframe"])
        provenance = {"source": "uploaded_csv_user_supplied"}
    incomplete = frame.attrs.get("quality", {}).get("incomplete_dropped", 0)
    available = len(frame)
    if available < value["lookback"]:
        raise ValueError(f"Need {value['lookback']} completed candles; this chart has {available}.")
    frame = frame.tail(value["lookback"]).reset_index(drop=True)
    step = interval_delta(value["timeframe"])
    if not frame.timestamp.diff().iloc[1:].eq(step).all():
        raise ValueError("Selected context contains gaps. Choose a contiguous chart window.")
    origin = frame.timestamp.iloc[-1] + step
    age = max(0, (now - origin).total_seconds())
    return value, {"coin": value["coin"], "timeframe": value["timeframe"],
                   "history": rows(frame), "available_candles": available,
                   "incomplete_dropped": incomplete, "origin": origin.isoformat(),
                   "forecast_end": (origin + step * value["horizon"]).isoformat(),
                   "age_seconds": age, "historical": age > 2 * step.total_seconds(),
                   "source": provenance.get("source", "local_csv_user_supplied"),
                   "synthetic": "synthetic" in str(provenance.get("source", "")).lower(),
                   "amount_note": "Quote turnover is unavailable; its input channel is zero."}


def original_paths(workspace, variant="kronos-small"):
    if variant not in PRETRAINED_MODELS:
        raise ValueError("Choose a supported pretrained Kronos model.")
    paths = [_inside(workspace, Path("models") / variant / part) for part in ("model", "tokenizer")]
    for path in paths:
        files = [_inside(workspace, file.relative_to(workspace)) for file in
                 (path / "config.json", path / "model.safetensors", path / "pytorch_model.bin")]
        if not files[0].is_file() or not any(file.is_file() for file in files[1:]):
            raise ValueError(f"Original {variant} weights are missing. Run python -m crypto_forecast kronos-setup --model {variant} first.")
    return paths


def forecast_chart(workspace, value, chart, *, progress=lambda _: None):
    """Inference only: no optimizer, weight save, or source candle modification."""
    from .pipeline import _device
    from ._vendor.kronos import KronosPredictor
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = _device(value["device"])
    progress("Loading Kronos for read-only inference.")
    if value["model"] in ORIGINAL_MODELS:
        variant = ORIGINAL_MODELS[value["model"]]
        model_path, tokenizer_path = original_paths(workspace, variant)
        model, tokenizer = load_pretrained(SimpleNamespace(kronos_model_path=str(model_path),
                                                          kronos_tokenizer_path=str(tokenizer_path)), device)
        label = f"{variant.capitalize()} · original pretrained weights"
    else:
        # The dashboard validates this run identifier and confines its path.
        path = _inside(workspace, Path("runs") / value["model"] / "model.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint["metadata"]
        if (metadata.get("format_version") not in (3, 4) or metadata.get("model_type") not in PRETRAINED_MODELS
                or metadata.get("features") != list(RAW_FEATURES)
                or metadata.get("timeframes") != [value["timeframe"]]):
            raise ValueError("Saved Kronos run is incompatible with this chart interval.")
        Kronos, Tokenizer, _ = _upstream()
        model, tokenizer = Kronos(**metadata["model_config"]), Tokenizer(**metadata["tokenizer_config"])
        model.load_state_dict(checkpoint["state_dict"])
        tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
        label = f"{metadata['model_type'].capitalize()} · fine-tuned run {value['model']}"
    model.eval().requires_grad_(False)
    tokenizer.eval().requires_grad_(False)
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)
    frame = pd.DataFrame(chart["history"])
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame["amount"] = 0.0
    future = pd.Series(pd.date_range(chart["origin"], periods=value["horizon"], freq=interval_delta(value["timeframe"])))
    progress(f"Forecasting {value['horizon']} future candles on {device}. No weights are changed.")
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(value["seed"])
        with torch.inference_mode():
            prediction = predictor.predict(frame, frame.timestamp, future, value["horizon"],
                                           T=value["temperature"], top_p=value["top_p"],
                                           sample_count=value["sample_count"], verbose=False)
    if len(prediction) != value["horizon"] or not np.isfinite(prediction.to_numpy()).all():
        raise ValueError("Kronos returned invalid candle values. Try a shorter forecast or another chart window.")
    if (prediction[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Kronos returned nonpositive prices. Try a shorter forecast or another chart window.")
    raw_prediction = prediction.copy()
    prediction["high"] = raw_prediction[["open", "high", "low", "close"]].max(axis=1)
    prediction["low"] = raw_prediction[["open", "high", "low", "close"]].min(axis=1)
    prediction["volume"] = prediction.volume.clip(lower=0)
    adjusted = int((raw_prediction[list(RAW_FEATURES[:-1])] != prediction[list(RAW_FEATURES[:-1])]).any(axis=1).sum())
    prediction["timestamp"] = future.to_numpy()
    raw_prediction["timestamp"] = future.to_numpy()
    return {**chart, "forecast": rows(prediction), "raw_forecast": rows(raw_prediction),
            "model_label": label, "model": value["model"], "device": str(device),
            "settings": {key: value[key] for key in ("horizon", "lookback", "seed", "sample_count", "temperature", "top_p")},
            "adjusted_candles": adjusted,
            "adjustment_note": "Predicted high/low ranges include open and close; negative volume is clipped to zero. Raw outputs are retained in JSON."}

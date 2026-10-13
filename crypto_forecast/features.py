"""Causal daily features and purged chronological forecasting samples."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from crypto_forecast import HORIZONS


FEATURE_NAMES = (
    "log_return_1d",
    "log_high_low",
    "log_close_open",
    "log_return_7d",
    "log_return_30d",
    "volatility_7d",
    "volatility_30d",
    "log_volume_change_1d",
    "log_volume_vs_7d",
    "log_volume_vs_30d",
)
_WARMUP = 30
_DAY = pd.Timedelta(days=1)
_REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass
class Sample:
    """A closed-candle sequence and exact calendar-day future log returns.

    ``origin`` is the UTC boundary just after the final input candle closes.
    ``price`` is that candle's close. ``label_end`` is the last target boundary.
    """

    x: np.ndarray
    y: np.ndarray
    coin: str
    origin: pd.Timestamp
    label_end: pd.Timestamp
    price: float
    drift: np.ndarray


def _validate_lookback(lookback: int) -> None:
    if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback < 1:
        raise ValueError("lookback must be a positive integer")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(_REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {', '.join(sorted(missing))}")
    result = frame.loc[:, _REQUIRED].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    if result["timestamp"].isna().any():
        raise ValueError("Candle timestamps cannot be missing")
    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate daily candle timestamps")
    if (result["timestamp"] != result["timestamp"].dt.normalize()).any():
        raise ValueError("Candle timestamps must be daily opens at UTC midnight")
    result = result.sort_values("timestamp").reset_index(drop=True)
    numeric = result.loc[:, _REQUIRED[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Candle prices and volumes must be finite")
    if (numeric[:, :4] <= 0).any() or (numeric[:, 4] < 0).any():
        raise ValueError("Candle prices must be positive and volume nonnegative")
    opening, high, low, close = (numeric[:, column] for column in range(4))
    if ((high < np.maximum(opening, close)) | (low > np.minimum(opening, close))).any():
        raise ValueError("Candle high/low must contain its open and close")
    return result


def _features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each row depends only on itself and the preceding 30 candle rows."""
    log_close = np.log(frame["close"].to_numpy(dtype=np.float64))
    daily = pd.Series(log_close).diff()
    volume = frame["volume"].astype(np.float64)
    log_volume = np.log1p(volume)
    values = np.column_stack(
        (
            daily,
            np.log(frame["high"].to_numpy() / frame["low"].to_numpy()),
            np.log(frame["close"].to_numpy() / frame["open"].to_numpy()),
            pd.Series(log_close).diff(7),
            pd.Series(log_close).diff(30),
            daily.rolling(7, min_periods=7).std(ddof=0),
            daily.rolling(30, min_periods=30).std(ddof=0),
            log_volume.diff(),
            log_volume - np.log1p(volume.rolling(7, min_periods=7).mean()),
            log_volume - np.log1p(volume.rolling(30, min_periods=30).mean()),
        )
    ).astype(np.float32)
    drift = (
        daily.rolling(7, min_periods=7).mean().to_numpy()[:, None]
        * np.asarray(HORIZONS, dtype=np.float64)[None, :]
    ).astype(np.float32)
    return values, drift


def _gap_prefix(frame: pd.DataFrame) -> np.ndarray:
    # prefix[i] counts broken daily transitions at or before row i.
    gaps = (frame["timestamp"].diff().iloc[1:] != _DAY).to_numpy(dtype=np.int64)
    return np.concatenate(([0], np.cumsum(gaps)))


def _sample(
    coin: str,
    frame: pd.DataFrame,
    features: np.ndarray,
    drift: np.ndarray,
    index: int,
    lookback: int,
    with_targets: bool,
) -> Sample:
    price = float(frame["close"].iloc[index])
    origin = pd.Timestamp(frame["timestamp"].iloc[index]) + _DAY
    if with_targets:
        future_prices = frame["close"].iloc[index + np.asarray(HORIZONS)].to_numpy()
        y = np.log(future_prices / price).astype(np.float32)
    else:
        y = np.full(len(HORIZONS), np.nan, dtype=np.float32)
    return Sample(
        x=features[index - lookback + 1 : index + 1].copy(),
        y=y,
        coin=coin,
        origin=origin,
        label_end=origin + max(HORIZONS) * _DAY,
        price=price,
        drift=drift[index].copy(),
    )


def build_samples(frames: dict[str, pd.DataFrame], lookback: int) -> list[Sample]:
    """Build samples only where context, inputs and all future days are contiguous.

    A missing day removes every sample whose 30-day feature warmup, lookback,
    or 30-day future target window crosses it. Missing candles are never filled.
    """
    _validate_lookback(lookback)
    samples: list[Sample] = []
    for coin, raw_frame in frames.items():
        frame = _prepare_frame(raw_frame)
        if len(frame) < lookback + _WARMUP + max(HORIZONS):
            continue
        features, drift = _features(frame)
        gaps = _gap_prefix(frame)
        for index in range(lookback + _WARMUP - 1, len(frame) - max(HORIZONS)):
            context_start = index - lookback + 1 - _WARMUP
            target_end = index + max(HORIZONS)
            if gaps[target_end] != gaps[context_start]:
                continue
            sample = _sample(coin, frame, features, drift, index, lookback, True)
            if not np.isfinite(sample.x).all() or not np.isfinite(sample.y).all():
                raise ValueError(f"Nonfinite feature or target for {coin} at {sample.origin}")
            samples.append(sample)
    return sorted(samples, key=lambda sample: (sample.origin, sample.coin))


def inference_sample(coin: str, frame: pd.DataFrame, lookback: int) -> Sample:
    """Use the latest candle, rejecting inadequate or gapped recent history."""
    _validate_lookback(lookback)
    prepared = _prepare_frame(frame)
    required = lookback + _WARMUP
    if len(prepared) < required:
        raise ValueError(f"{coin} needs at least {required} daily candles for inference")
    start = len(prepared) - required
    gaps = _gap_prefix(prepared)
    if gaps[-1] != gaps[start]:
        raise ValueError(f"{coin} latest inference history has missing calendar days")
    features, drift = _features(prepared)
    sample = _sample(coin, prepared, features, drift, len(prepared) - 1, lookback, False)
    if not np.isfinite(sample.x).all() or not np.isfinite(sample.drift).all():
        raise ValueError(f"{coin} latest inference history produces nonfinite features")
    return sample


def split_samples(
    samples: list[Sample], train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> tuple[dict[str, list[Sample]], dict]:
    """Use shared calendar boundaries and purge labels that touch the next split."""
    if not (
        np.isfinite(train_fraction)
        and np.isfinite(validation_fraction)
        and 0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Split fractions must be positive and leave a test partition")
    ordered = sorted(samples, key=lambda sample: (sample.origin, sample.coin))
    origins = sorted({sample.origin for sample in ordered})
    validation_index = int(len(origins) * train_fraction)
    test_index = int(len(origins) * (train_fraction + validation_fraction))
    if not 0 < validation_index < test_index < len(origins):
        raise ValueError("Need enough unique daily origins for three chronological splits")
    val_start = origins[validation_index]
    test_start = origins[test_index]
    splits: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}
    purged = {"train": 0, "validation": 0}
    for sample in ordered:
        if sample.origin < val_start:
            if sample.label_end < val_start:
                splits["train"].append(sample)
            else:
                purged["train"] += 1
        elif sample.origin < test_start:
            if sample.label_end < test_start:
                splits["validation"].append(sample)
            else:
                purged["validation"] += 1
        else:
            splits["test"].append(sample)
    metadata = {
        "val_start": val_start.isoformat(),
        "test_start": test_start.isoformat(),
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "unique_origins": len(origins),
        "counts": {name: len(items) for name, items in splits.items()},
        "purged": {**purged, "total": sum(purged.values())},
    }
    return splits, metadata

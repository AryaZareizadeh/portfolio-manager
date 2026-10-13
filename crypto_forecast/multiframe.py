"""Causal OHLCV sequences aligned across four completed-candle timeframes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_forecast import HORIZONS
from crypto_forecast.features import Sample, _validate_lookback
from crypto_forecast.timeframes import TIMEFRAMES, interval_delta


MTF_FEATURE_NAMES = (
    "log_return_1bar",
    "log_high_low",
    "log_close_open",
    "log_return_7bar",
    "log_return_30bar",
    "volatility_7bar",
    "volatility_30bar",
    "log_volume_change_1bar",
    "log_volume_vs_7bar",
    "log_volume_vs_30bar",
    "log_volume",
)
_WARMUP = 30
_DAY_NS = pd.Timedelta(days=1).value
_REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass
class MultiSample(Sample):
    """Four input branches; ``input_ends`` records each last candle's close time."""

    input_ends: tuple[pd.Timestamp, ...]


@dataclass
class _Branch:
    ends: np.ndarray
    prices: np.ndarray
    features: np.ndarray
    gaps: np.ndarray
    step: int


def _prepare_frame(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    missing = set(_REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing {interval} candle columns: {', '.join(sorted(missing))}")
    result = frame.loc[:, _REQUIRED].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise").astype(
        "datetime64[ns, UTC]"
    )
    if result["timestamp"].isna().any():
        raise ValueError(f"{interval} candle timestamps cannot be missing")
    if result["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate {interval} candle timestamps")
    if (result["timestamp"].array.asi8 % interval_delta(interval).value != 0).any():
        raise ValueError(f"{interval} candle timestamps must be aligned UTC candle opens")
    result = result.sort_values("timestamp").reset_index(drop=True)
    numeric = result.loc[:, _REQUIRED[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{interval} candle prices and volumes must be finite")
    if (numeric[:, :4] <= 0).any() or (numeric[:, 4] < 0).any():
        raise ValueError(f"{interval} prices must be positive and volume nonnegative")
    opening, high, low, close = (numeric[:, column] for column in range(4))
    if ((high < np.maximum(opening, close)) | (low > np.minimum(opening, close))).any():
        raise ValueError(f"{interval} candle high/low must contain its open and close")
    for index, column in enumerate(_REQUIRED[1:]):
        result[column] = numeric[:, index]
    return result


def _features(frame: pd.DataFrame) -> np.ndarray:
    """All windows count this branch's own bars, including its observed volume."""
    log_close = np.log(frame["close"].astype(np.float64))
    returns = log_close.diff()
    volume = frame["volume"].astype(np.float64)
    log_volume = np.log1p(volume)
    return np.column_stack(
        (
            returns,
            np.log(frame["high"].to_numpy() / frame["low"].to_numpy()),
            np.log(frame["close"].to_numpy() / frame["open"].to_numpy()),
            log_close.diff(7),
            log_close.diff(30),
            returns.rolling(7, min_periods=7).std(ddof=0),
            returns.rolling(30, min_periods=30).std(ddof=0),
            log_volume.diff(),
            log_volume - np.log1p(volume.rolling(7, min_periods=7).mean()),
            log_volume - np.log1p(volume.rolling(30, min_periods=30).mean()),
            log_volume,
        )
    ).astype(np.float32)


def _prepare_branches(frames: dict[str, pd.DataFrame]) -> dict[str, _Branch]:
    missing = set(TIMEFRAMES) - set(frames)
    if missing:
        raise ValueError(f"Missing candle timeframes: {', '.join(sorted(missing))}")
    branches = {}
    for interval in TIMEFRAMES:
        frame = _prepare_frame(frames[interval], interval)
        step = interval_delta(interval).value
        ends = frame["timestamp"].array.asi8 + step
        gaps = np.concatenate(([0], np.cumsum(np.diff(ends) != step)))[: len(ends)]
        branches[interval] = _Branch(
            ends=ends,
            prices=frame["close"].to_numpy(dtype=np.float64),
            features=_features(frame),
            gaps=gaps,
            step=step,
        )
    return branches


def _context_indices(
    branch: _Branch, origins: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized completed-candle lookup, including stale and gapped inputs."""
    indices = np.searchsorted(branch.ends, origins, side="right") - 1
    starts = indices - lookback + 1 - _WARMUP
    valid = starts >= 0
    eligible = np.flatnonzero(valid)
    if len(eligible):
        selected = indices[eligible]
        expected_ends = origins[eligible] // branch.step * branch.step
        valid[eligible] &= branch.ends[selected] == expected_ends
        valid[eligible] &= branch.gaps[selected] == branch.gaps[starts[eligible]]
    return indices, valid


def _exact_indices(branch: _Branch, boundaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(branch.ends, boundaries)
    valid = indices < len(branch.ends)
    eligible = np.flatnonzero(valid)
    if len(eligible):
        valid[eligible] &= branch.ends[indices[eligible]] == boundaries[eligible]
    return indices, valid


def _make_sample(
    coin: str,
    branches: dict[str, _Branch],
    origin: int,
    indices: dict[str, int],
    drift_index: int,
    lookback: int,
    target_indices: np.ndarray | None,
) -> MultiSample:
    reference = branches["15m"]
    price = float(reference.prices[indices["15m"]])
    x = np.stack([
        branches[interval].features[indices[interval] - lookback + 1 : indices[interval] + 1]
        for interval in TIMEFRAMES
    ])
    drift = (
        np.log(price / reference.prices[drift_index]) / 7 * np.asarray(HORIZONS)
    ).astype(np.float32)
    y = (
        np.log(reference.prices[target_indices] / price).astype(np.float32)
        if target_indices is not None
        else np.full(len(HORIZONS), np.nan, dtype=np.float32)
    )
    if not np.isfinite(x).all() or not np.isfinite(drift).all():
        raise ValueError(f"Nonfinite features for {coin} at {pd.Timestamp(origin, tz='UTC')}")
    if target_indices is not None and not np.isfinite(y).all():
        raise ValueError(f"Nonfinite targets for {coin} at {pd.Timestamp(origin, tz='UTC')}")
    boundary = pd.Timestamp(origin, tz="UTC")
    return MultiSample(
        x=x,
        y=y,
        coin=coin,
        origin=boundary,
        label_end=boundary + pd.Timedelta(days=max(HORIZONS)),
        price=price,
        drift=drift,
        input_ends=tuple(
            pd.Timestamp(branches[interval].ends[indices[interval]], tz="UTC")
            for interval in TIMEFRAMES
        ),
    )


def build_multiframe_samples(
    frames: dict[str, dict[str, pd.DataFrame]], lookback: int, origin_interval: str = "4h"
) -> list[MultiSample]:
    """Align closed OHLCV bars and exact 1-, 7-, and 30-calendar-day targets.

    Every branch needs ``lookback + 30`` contiguous bars. The 15m reference
    must also be continuous from seven days before the origin through the final
    target; missing candles are never filled or used to shift a target date.
    """
    _validate_lookback(lookback)
    origin_step = interval_delta(origin_interval).value
    samples = []
    for coin, coin_frames in frames.items():
        branches = _prepare_branches(coin_frames)
        reference = branches["15m"]
        origins = reference.ends[reference.ends % origin_step == 0]
        indices = {}
        valid = np.ones(len(origins), dtype=bool)
        for interval, branch in branches.items():
            indices[interval], context_valid = _context_indices(branch, origins, lookback)
            valid &= context_valid
        drift_indices, drift_valid = _exact_indices(reference, origins - 7 * _DAY_NS)
        valid &= drift_valid
        targets = []
        for horizon in HORIZONS:
            target_indices, target_valid = _exact_indices(reference, origins + horizon * _DAY_NS)
            targets.append(target_indices)
            valid &= target_valid
        targets = np.column_stack(targets)
        eligible = np.flatnonzero(valid)
        if len(eligible):
            valid[eligible] &= (
                reference.gaps[targets[eligible, -1]] == reference.gaps[drift_indices[eligible]]
            )
        for position in np.flatnonzero(valid):
            samples.append(
                _make_sample(
                    coin,
                    branches,
                    int(origins[position]),
                    {interval: int(index[position]) for interval, index in indices.items()},
                    int(drift_indices[position]),
                    lookback,
                    targets[position],
                )
            )
    return sorted(samples, key=lambda sample: (sample.origin, sample.coin))


def inference_multiframe_sample(
    coin: str, frames: dict[str, pd.DataFrame], lookback: int, origin_interval: str = "4h"
) -> MultiSample:
    """Infer at the latest cadence boundary; reject stale or gapped branches."""
    _validate_lookback(lookback)
    origin_step = interval_delta(origin_interval).value
    branches = _prepare_branches(frames)
    reference = branches["15m"]
    if not len(reference.ends):
        raise ValueError(f"{coin} needs 15m candle history for inference")
    origin = int(reference.ends[-1] // origin_step * origin_step)
    origins = np.asarray([origin], dtype=np.int64)
    indices = {}
    for interval, branch in branches.items():
        index, valid = _context_indices(branch, origins, lookback)
        if not valid[0]:
            raise ValueError(
                f"{coin} latest {interval} inference history needs at least {lookback + _WARMUP} "
                "contiguous candles through its latest completed candle; history is insufficient, "
                "stale, or has missing candles"
            )
        indices[interval] = int(index[0])
    drift_indices, drift_valid = _exact_indices(reference, origins - 7 * _DAY_NS)
    if not drift_valid[0] or reference.gaps[indices["15m"]] != reference.gaps[drift_indices[0]]:
        raise ValueError(f"{coin} latest 15m inference history needs seven continuous calendar days")
    return _make_sample(coin, branches, origin, indices, int(drift_indices[0]), lookback, None)

"""CPU-only window planning and materialization without Torch dependencies."""

from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import wait as wait_for_parent
import os
import threading

import numpy as np
import pandas as pd


@dataclass
class KronosSample:
    x: np.ndarray
    future: np.ndarray
    x_timestamps: pd.DatetimeIndex
    future_timestamps: pd.DatetimeIndex
    y: np.ndarray
    coin: str
    origin: pd.Timestamp
    label_end: pd.Timestamp
    price: float
    drift: np.ndarray


def plan_windows(frames, lookback, bars, step, week, train_fraction, validation_fraction, emit=None):
    """Check gaps and purge chronological splits using compact row indices."""
    if not (np.isfinite(train_fraction) and np.isfinite(validation_fraction)
            and 0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and leave a test partition")
    first_index = max(lookback - 1, week)
    horizon = int(bars[-1])
    total = sum(max(0, len(frame) - horizon - first_index) for frame in frames.values())
    completed = usable = 0
    windows, origins = {}, {}
    for coin, frame in frames.items():
        if emit:
            emit("preparation", stage="preparing", preparation_phase="windows",
                 preparation_completed=completed, preparation_total=total, coin=coin, usable_samples=usable)
        stamps = pd.DatetimeIndex(frame.timestamp).as_unit("ns").asi8
        indices = np.arange(first_index, len(frame) - horizon, dtype=np.int64)
        gaps = np.concatenate(([0], np.cumsum(np.diff(stamps) != step.value)))
        first = np.minimum(indices - lookback + 1, indices - week)
        valid = indices[gaps[indices + horizon] == gaps[first]]
        windows[coin] = valid
        origins[coin] = stamps[valid] + step.value
        completed += len(indices)
        usable += len(valid)
    if emit:
        emit("preparation", stage="preparing", preparation_phase="windows",
             preparation_completed=completed, preparation_total=total, usable_samples=usable)
        emit("preparation", stage="preparing", preparation_phase="splitting",
             preparation_completed=0, preparation_total=0)
    unique = np.unique(np.concatenate(list(origins.values()))) if origins else np.empty(0, dtype=np.int64)
    validation_index = int(len(unique) * train_fraction)
    test_index = int(len(unique) * (train_fraction + validation_fraction))
    if not 0 < validation_index < test_index < len(unique):
        raise ValueError("Need enough unique daily origins for three chronological splits")
    val_start, test_start = unique[validation_index], unique[test_index]
    partitions = {name: {} for name in ("train", "validation", "test")}
    purged = {"train": 0, "validation": 0}
    for coin, origin in origins.items():
        ends = origin + step.value * horizon
        before_val = origin < val_start
        before_test = (origin >= val_start) & (origin < test_start)
        masks = {"train": before_val & (ends < val_start),
                 "validation": before_test & (ends < test_start), "test": origin >= test_start}
        purged["train"] += int(np.count_nonzero(before_val & (ends >= val_start)))
        purged["validation"] += int(np.count_nonzero(before_test & (ends >= test_start)))
        for name, mask in masks.items():
            partitions[name][coin] = windows[coin][mask]
    info = {"val_start": pd.Timestamp(val_start, tz="UTC").isoformat(),
            "test_start": pd.Timestamp(test_start, tz="UTC").isoformat(),
            "train_fraction": train_fraction, "validation_fraction": validation_fraction,
            "unique_origins": len(unique),
            "counts": {name: sum(map(len, values.values())) for name, values in partitions.items()},
            "purged": {**purged, "total": sum(purged.values())}}
    return partitions, info


def materialize(coin, values, timestamps, indices, lookback, bars, step, week):
    """Copy only selected input/target windows, preserving original arithmetic."""
    samples = []
    horizon = int(bars[-1])
    for index in indices:
        origin = timestamps[index] + step
        price = float(values[index, 3])
        drift = (np.log(price / values[index - week, 3]) / max(1, week) * bars).astype(np.float32)
        samples.append(KronosSample(
            values[index - lookback + 1:index + 1].copy(), values[index + 1:index + 1 + horizon].copy(),
            timestamps[index - lookback + 1:index + 1], pd.date_range(origin, periods=horizon, freq=step),
            np.log(values[index + bars, 3] / price).astype(np.float32), coin, origin,
            origin + step * horizon, price, drift))
    return samples


def watch_parent():
    """Dashboard cancellation must also stop spawned CPU workers."""
    parent = multiprocessing.parent_process()
    if parent is not None:
        def stop_on_exit():
            wait_for_parent([parent.sentinel])
            os._exit(3)
        threading.Thread(target=stop_on_exit, daemon=True).start()


def materialize_task(task):
    coin, values_path, timestamps_path, partition, indices, lookback, bars, step, week = task
    values = np.load(values_path, mmap_mode="r")
    try:
        timestamps = pd.to_datetime(np.load(timestamps_path), utc=True)
        return partition, materialize(coin, values, timestamps, indices, lookback, bars, step, week)
    finally:
        values._mmap.close()

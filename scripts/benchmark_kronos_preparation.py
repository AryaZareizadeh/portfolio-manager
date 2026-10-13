"""Compare preparation paths in separate processes for meaningful peak RAM.

python -m scripts.benchmark_kronos_preparation --mode legacy
python -m scripts.benchmark_kronos_preparation --mode serial
python -m scripts.benchmark_kronos_preparation --mode parallel --workers 4
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
import time


def peak_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / 2**20
    import resource
    import sys
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)


def main():
    # Imports stay inside main so spawned CPU workers do not load Torch.
    import numpy as np
    import pandas as pd
    from crypto_forecast import kronos_backend as backend
    from crypto_forecast.features import split_samples
    from crypto_forecast.pipeline import TrainingConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "serial", "parallel"), required=True)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--coins", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", help="Optional real local 1h candles; otherwise use synthetic data.")
    args = parser.parse_args()
    coins = [f"C{number}" for number in range(args.coins)]
    if args.data_dir:
        from pathlib import Path
        coins = [path.stem for path in sorted((Path(args.data_dir) / "1h").glob("*.csv"))][:args.coins]
        frames = {coin: frame.tail(args.rows).reset_index(drop=True) for coin, frame in
                  backend.load_candles(args.data_dir, coins, "1h").items()}
    else:
        index = np.arange(args.rows)
        close = 100 + index * 0.03 + np.sin(index / 12)
        frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=args.rows, freq="h", tz="UTC"),
                              "open": close * .999, "high": close * 1.005, "low": close * .995,
                              "close": close, "volume": 1000 + np.sin(index / 9)})
        frames = {coin: frame for coin in coins}
    config = TrainingConfig(coins=coins, model_type="kronos-small", kronos_timeframe="1h",
                            lookback=345, kronos_prediction_length=168,
                            train_fraction=.7, validation_fraction=.2,
                            kronos_train_stride=24, kronos_eval_stride=24)
    start = time.perf_counter()
    events = []
    if args.mode == "legacy":
        samples = backend.build_samples(frames, config.lookback, "1h", prediction_length=168)
        partitions, info = split_samples(samples, .7, .2)
        selected = {name: backend.select_stride(values, 24) for name, values in partitions.items()}
        candidate_bytes = sum(sample.x.nbytes + sample.future.nbytes for sample in samples)
        workers_used = 1
    else:
        selected, info, _, _ = backend.prepare_training_samples(
            frames, replace(config, kronos_preparation_workers=1 if args.mode == "serial" else args.workers),
            emit=lambda kind, **value: events.append(value))
        candidate_bytes = sum(s.x.nbytes + s.future.nbytes for values in selected.values() for s in values)
        workers_used = events[-1]["preparation_workers"]
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256()
    for name, values in selected.items():
        digest.update(name.encode())
        for sample in values:
            digest.update(f"{sample.coin}:{sample.origin.isoformat()}".encode())
            for value in (sample.x, sample.future, sample.y, sample.drift):
                digest.update(value.tobytes())
    print(json.dumps({"mode": args.mode, "rows_per_coin": args.rows, "coins": len(coins),
                      "workers": workers_used, "seconds": round(elapsed, 3),
                      "parent_peak_mib": round(peak_memory(), 1),
                      "copied_candle_mib": round(candidate_bytes / 2**20, 1),
                      "counts": info["counts"], "selected_counts": {name: len(values) for name, values in selected.items()},
                      "sample_sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()

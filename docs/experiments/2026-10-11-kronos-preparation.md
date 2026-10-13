# Kronos preparation optimization — 2026-10-11

Preparation now computes valid origins and chronological boundary purges using
NumPy row indices. Coin eligibility and per-partition strides are applied before
materializing candle arrays. The prior `build_samples` path remains available
as a reference and for callers needing every window.

Parallel preparation uses 2–4 spawned CPU processes, read-only memory-mapped
candle arrays, and batches of at most 256 selected windows. Only as many batches
as workers are in flight. Results are sorted by origin and coin, independently
of worker completion order. Parent progress updates are throttled to once per
second, with initial and final updates always emitted. A parent-exit watcher
stops CPU workers if a dashboard training process is terminated.

## Measurements

These are single runs on the local Windows machine. Each path was invoked in a
separate process with no concurrent benchmark, using the same settings:
1h candles, 345 input bars, 168 future bars, stride 24 for every partition,
70% training and 20% validation. Timing covers preparation, including process
startup and transfer for the parallel path, but excludes CSV loading and model
training. Peak memory is the **parent process only**, including Python and
Torch imports; parallel worker memory is additional.

Synthetic history: four coins, 6,000 candles each.

| Path | Preparation seconds | Peak parent MiB | Copied input/target MiB |
| --- | ---: | ---: | ---: |
| Original build-all path | 5.816 | 1,105.6 | 515.5 |
| Optimized serial | 0.050 | 468.6 | 20.3 |
| Optimized, four processes | 0.898 | 479.2 | 20.3 |

All three paths selected 616 training, 156 validation and 92 test examples.
The SHA-256 covering every selected origin, coin, input array, target array,
return target and drift was identical:
`08133ab9117fabbf51a3d0c27f749ce28502cdbd5ec73abd7a3250b65b6fe0d5`.

Representative local history: 13 alphabetically selected local coins, retaining
the most recent 50,000 rows per coin (shorter histories retain all their rows).
Original preparation was not repeated at this size because it materializes many
gigabytes of overlapping windows.

| Path | Preparation seconds | Peak parent MiB | Copied input/target MiB |
| --- | ---: | ---: | ---: |
| Optimized serial | 3.494 | 1,153.9 | 534.7 |
| Optimized, four processes | 3.844 | 1,248.2 | 534.7 |
| Optimized, two processes | 5.535 | 1,245.6 | 534.7 |

All paths selected 15,307 training, 4,940 validation and 2,522 test examples.
Their selected-sample SHA-256 was identical:
`55c8c7c4fc921f2ad755ab0d96ad251f448ad1066b0ffe25c828f9c043f83e38`.

At these sizes, constructing fewer samples provides the gain; parallel startup
and serialization offset CPU savings. Automatic therefore uses serial
preparation below 50,000 selected windows. At larger sizes it allows up to four
processes; this is a conservative heuristic, not a measured guarantee of a
speedup. Explicit worker choices allow comparison on other machines/workloads.

## Reproduce

Run each command separately from the project root:

```powershell
.\.venv-kronos\Scripts\python.exe -m scripts.benchmark_kronos_preparation --mode legacy
.\.venv-kronos\Scripts\python.exe -m scripts.benchmark_kronos_preparation --mode serial
.\.venv-kronos\Scripts\python.exe -m scripts.benchmark_kronos_preparation --mode parallel --workers 4
```

Append `--data-dir data --rows 50000 --coins 13` to an optimized-path command
to repeat the local-history comparison. Local files and the resulting hash can
change as candle history is refreshed. No benchmark command starts training.

## Correctness checks

Regression tests compare the original path against optimized preparation for
all four candle intervals, configurable candle targets and legacy day targets,
missing candles, short histories, failed latest-history eligibility, unequal
strides, ordering, every sample field, and exact split metadata. A real spawned
process test compares parallel and serial results and checks parent-owned
progress. Additional checks cover abrupt parent termination, worker validation,
CLI/dashboard settings, and materialization progress in the browser renderer.

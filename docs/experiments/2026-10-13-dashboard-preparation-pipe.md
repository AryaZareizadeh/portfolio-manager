# Windows dashboard preparation pipe fix — 2026-10-13

Run `train-20261013-052303-c806d752` stayed at 0/53,554 prepared windows for about 6 minutes. All four preparation processes were idle while the training parent continued to emit progress snapshots. No GPU training had begun.

The standalone four-worker preparation benchmark completed, but the same spawned process pool stalled when the dashboard's background command reader was blocked in `os.read(stdin)`. An isolated reproduction with an open, idle command pipe reproduced the stall with and without `CREATE_NO_WINDOW`. Worker startup was blocked at Windows handle duplication; terminating the reproduction parent released that call with an access-denied error. That termination error was diagnostic cleanup, not the original failure.

The Windows command reader now calls `PeekNamedPipe` and reads only available bytes. It polls an empty pipe every 50 ms and recognizes a broken pipe as parent EOF. This removes the outstanding synchronous pipe read during process startup, while preserving bounded JSON commands and dashboard-disconnect cancellation. Other platforms retain their existing blocking reader.

The regression test failed before the fix: a preparation worker could not start while the dashboard command pipe remained open and idle. After the fix, real spawned workers start normally. Real pipe tests also verify fragmented training commands, download publication acknowledgements, and exit code 3 when the parent closes the pipe.

Verification:

- Preparation, training-decision, and incremental-download focused suite: 65 tests passed.
- Preparation suite including real command transport and EOF cancellation: 17 tests passed.
- A real isolated dashboard job using two preparation workers, synthetic candle data, and tiny local Kronos weights completed tokenizer epoch 1 and saved its checkpoint. All 34 selected windows were prepared.
- Preparation using the stopped run's actual 33-coin configuration and repaired local data completed **53,719/53,719 windows with four workers in 7.535 seconds**. This check prepared samples only; it did not restart training or alter the stopped run. The larger count reflects the separately repaired LTC/DASH derived histories. Five recently listed coins remain excluded because the chronological training split has insufficient samples.
- Independent read-only review found no actionable issues in the worker reader, coverage handling, or repaired data integrity.
- Final full project regression suite: **542 tests passed in 111.91 seconds**.

Data repair details and exact backup locations are recorded in [the KuCoin candle report](2026-10-13-kucoin-sparse-candle-repair.md).

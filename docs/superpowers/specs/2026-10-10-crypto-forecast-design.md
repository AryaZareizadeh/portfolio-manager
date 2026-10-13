# Cryptocurrency deep learning forecast design

Build a standalone research package in this folder, as requested in the current conversation. The existing handoff supplies context and a candidate universe; the current request selects deep learning and narrows the deliverable to historical forecasting, training, and validation. There is no order execution in this package.

## Outputs and selected assets

Value means spot market price quoted in USDT, rather than intrinsic fair value. Forecasts predict cumulative log returns over exactly 1, 7, and 30 calendar days, converted into estimated prices and percentage changes. The user selected forecast origins every four hours at UTC boundaries. The reference price is the native 15-minute close at that boundary. The selected group is BTC, ETH, SOL, XRP, and ADA; the CLI and JSON configuration accept other selected coins. USDT is the quote unit, rather than a learned target.

## Data and causal features

Download public Binance spot native 15-minute candles without credentials, with checksum-verified cached archives and provenance. Derive 1-hour, 4-hour and daily candles from complete UTC buckets using first open, max high, min low, last close and summed traded volume. Append recent completed 15-minute candles through the public market-data API. Accept normalized local CSV files with explicit instrument, source, interval and USDT metadata. Reject duplicate or invalid candles and exclude gaps and unfinished buckets. Archive timestamps may be milliseconds or microseconds. Each branch requires 60 input candles plus 30 warmup candles; inputs end at or before the forecast origin. The reference series must be continuous from seven days before the origin through the 30-day target. Each branch contains 11 causal price and volume features, including log volume, log changes and relative volume. Legacy daily data and checkpoints remain usable.

## Model and validation

Four separate compact PyTorch GRUs encode the 15m, 1h, 4h and 1d feature sequences. Fuse all four outputs with a learned coin embedding in one head predicting three direct log returns. Fit feature normalization independently per branch and target normalization on training only. Use global chronological boundaries across all assets with 70% training, 15% validation, and 15% test, and remove every training sample whose 30-day label reaches validation and every validation sample whose label reaches test. Select best epoch and early stop on validation only. Freeze the selected model before final test evaluation. Coins without sufficient training samples cannot receive a trained forecast.

Compare against unchanged-price and trailing mean-return baselines on identical held-out origins. Report return MAE/RMSE, percentage-return MAE, direction accuracy, and counts per asset/horizon and coin-balanced aggregate. Include non-overlapping origin metrics because monthly labels overlap. Derive empirical 80% ranges from validation residuals, never test residuals; report their test coverage and width. Ranges are empirical and may be miscalibrated when markets change.

## Reproducibility and interface

Commands: download, train, predict, and demo. Save model weights, model settings, coin mapping, normalization, source data hashes, exact split boundaries, seed, learning curve, and reports. Save forecasts and held-out predictions as CSV/JSON and produce a standalone HTML report. Prediction rejects unknown/untrained coins and data older than a user-configurable freshness limit unless explicitly allowed for historical use. A demo uses clearly labeled synthetic data solely to verify the pipeline.

## Verification

Tests must prove exact calendar targets, time purging, shared boundaries, training-only scalers, gap and duplicate handling, timestamp units, model round-trip equality, stale data handling, and one real gradient-based training cycle. Then download public history, train and validate the selected group, and inspect the outputs. Report actual baseline comparison, even when the neural model loses.

# Cryptocurrency Forecast Implementation Plan

> For agentic workers: parallelize independent data and feature modules, then integrate and review the complete pipeline.

**Goal:** Deliver a configurable deep learning model trained and validated on selected coins with 1-, 7-, and 30-day price forecasts.

**Architecture:** Four OHLCV resolutions (15m, 1h, 4h, 1d) feed separate causal GRU encoders, whose outputs are fused into one forecast head. Four-hour forecast origins use globally purged chronological datasets, validation-only model selection, reproducible artifacts, and held-out comparisons. The original daily mode remains supported.

**Tech Stack:** Python 3.11+, NumPy, pandas, PyTorch, pytest, standard-library HTML/CSV/HTTP.

**Spec:** `docs/superpowers/specs/2026-10-10-crypto-forecast-design.md`

## Global constraints

- Horizons are exactly 1, 7, and 30 calendar days from four-hour UTC origins, referenced to completed native 15m candles.
- Feature and target scalers fit training only; all assets share chronological boundaries.
- Training labels end before validation; validation labels end before test.
- No interpolation, no live orders, no substituted assets, and no final-test model selection.
- Synthetic demonstration outputs must identify their synthetic origin.

## Review focus

- Short-lived or sparsely traded coins must produce explicit exclusions.
- Missing days must never turn a 30-day target into 30 observed bars.
- Public endpoints may be unavailable; cached/local CSV use must remain usable.
- Historical forecasts must expose reference dates and data freshness.
- Artifacts must reconstruct the exact model and scalers without unsafe object loading.

## Task 1: Source data

- [x] Create `crypto_forecast/data.py` with `load_data(data_dir, coins, as_of=None)` and `download_history(coins, start, end, data_dir, refresh=False)`.
- [x] Test malformed OHLCV, duplicate dates, mixed timestamp units, incomplete candles, and gaps in `tests/test_data.py`.
- [x] Cache public daily spot history; write CSVs and provenance; allow CSV imports for supplemental sources.

## Task 2: Causal feature datasets

- [x] Create `crypto_forecast/features.py` with `build_samples(frames, lookback)` and `split_samples(samples, train_fraction, validation_fraction)`; expose sample inputs, targets, coin, origin, label end, reference price and drift baseline.
- [x] Test analytical returns, global purges, contiguous windows, and leakage invariants in `tests/test_features.py`.
- [x] Construct inference windows with the same feature computation.

## Task 3: Training and inference

- [x] Create `crypto_forecast/model.py` with the compact GRU, train-only normalization, deterministic seeded training, early stopping, and weights-only loading.
- [x] Create `crypto_forecast/pipeline.py` for eligibility, train/validation/test reporting, empirical residual intervals, data hashes, and forecast CSVs.
- [x] Test changed validation inputs cannot alter training normalization; verify saved prediction equality and real gradient updates in `tests/test_pipeline.py`.

## Task 4: User commands and reports

- [x] Create `crypto_forecast/cli.py`, `__main__.py`, and `report.py`; support download, train, predict, and labeled synthetic demo.
- [x] Write README with installation, selected coins, CSV format, model semantics, validation, and artifact descriptions.
- [x] Run `python -m pytest -q` and a synthetic end-to-end smoke test.
- [x] Download real selected-coin data, train a frozen model, inspect final-test baseline comparison, and open the HTML report.
- [x] Independently review temporal leakage, inference semantics, and honest reporting; fix findings and repeat affected verification.

## Completion evidence

Full suite: 70 tests passed. Synthetic CLI demo, CUDA historical training, saved-weight CPU inference, and real-run artifact/hash checks passed. The complete five-coin experiment and baseline comparison are documented in docs/experiments/2026-10-10-selected-coins.md.

## Combined-timeframe extension

- [x] Download native 15m archives; derive complete UTC 1h/4h/1d buckets with summed volume and interval provenance.
- [x] Add closed-candle alignment, exact native-15m horizon targets, strict gap exclusion and four-hour origins.
- [x] Train four separate GRU encoders with 11 price/volume features and per-branch training-only scalers.
- [x] Preserve daily checkpoint compatibility; add format-v2 combined model artifacts and 8-hour freshness checks.
- [x] Support recent public-data appends; verify continuity, source hashes, bucket volumes and failure preservation.
- [x] Verify all four encoders learn, volume affects every branch, saved weights reproduce forecasts, and historical cutoffs stay causal.
- [x] Run the combined synthetic CLI demo and independent temporal/volume review.
- [x] Complete real five-coin training, final validation/test verification, and measured experiment note.

Extension evidence: 131 tests passed. The combined historical model trained on 32,615 samples, validation selected epoch 1, and the frozen final test covered 7,185 samples. All three horizons underperformed the unchanged-price baseline; measured results and limitations are recorded in docs/experiments/2026-10-10-multiframe-coins.md.

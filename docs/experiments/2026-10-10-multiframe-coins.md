# Combined-timeframe five-coin experiment

Completed 10 October 2026. Frozen artifacts are in `runs/multiframe-coins`, with the standalone `report.html`, model weights, source manifest, forecasts and held-out predictions. The earlier daily experiment remains in `runs/selected-coins`.

## Inputs and model

BTC, ETH, SOL, XRP and ADA spot pairs quoted in USDT. Historical native 15-minute candles begin 1 January 2021; recent public REST candles extend the data through the close at 10 October 2026 11:45 UTC. Each coin has 202,345 native 15m rows, 50,582 hourly rows, 12,638 four-hour rows and 2,101 daily rows. There are 70 missing native candle slots in the historical source. Complete-bucket aggregation excludes 21 hourly, 12 four-hour and 7 daily buckets. Required windows containing gaps are excluded; no candles are filled.

Four GRU encoders each consume 60 completed candles of their own timeframe, with 11 price/volume features and an additional 30-candle feature warmup. Features include log traded volume, volume change and volume relative to 7/30-bar means. Larger candle volumes are sums of their constituent native 15m traded volumes. Encoders have 48 hidden units; their outputs and a learned coin embedding feed a shared three-horizon head. Forecast origins occur every four hours. Only candles closed by that origin can enter the model; daily inputs at an intraday origin end at the preceding UTC midnight.

The targets are cumulative log returns from the native 15m close at the origin to the native close exactly 1, 7 or 30 calendar days later. This estimates market price, rather than intrinsic value.

## Training and validation

Seed 42; CUDA; batch size 128; Adam at learning rate 0.001; maximum 60 epochs with patience 10. Training and target normalization use training observations only, with feature statistics fitted independently per timeframe. Coin-balanced sampling trains all five coin embeddings. The global chronological split has validation starting 19 May 2025 12:00 UTC and final test starting 14 January 2026 00:00 UTC. Thirty-day labels reaching the following split are purged (900 train and 900 validation samples).

Eligible samples: 32,615 training, 6,285 validation and 7,185 final test. Each coin contributes 6,523 / 1,257 / 1,437 samples respectively; no selected coin was excluded. The run stopped after 11 epochs. Validation selected epoch 1 (normalized loss 0.58355); later epochs reduced training loss while increasing validation loss. Only the selected frozen model was evaluated on final test. Empirical 80% price ranges use pooled validation residuals per horizon; that validation block also selects the epoch, so calibration is not independent of model selection.

## Final-test results

Coin-balanced macro return MAE, in percentage points; lower is better:

| Horizon | Combined GRU | Unchanged price | Trailing 7-day drift |
| --- | ---: | ---: | ---: |
| 1 day | 2.3142 | 2.2807 | 2.4646 |
| 7 days | 7.0065 | 6.2197 | 9.2704 |
| 30 days | 17.3087 | 13.2407 | 31.3776 |

The combined model underperformed the unchanged-price baseline at all three horizons. Its direction accuracy was 48.52%, 45.40% and 46.40%; empirical range coverage was 84.65%, 81.34% and 75.28%. These results do not establish a reliable trading signal or show that adding timeframes improves forecasting accuracy. The daily experiment uses different origins and test boundaries, so its scores are not a controlled comparison with this experiment.

Non-overlapping origins reduce dependence between neighboring horizon outcomes. Their GRU versus unchanged-price MAE was 2.2762 versus 2.2577 (1 day; 1,200 total samples), 7.4922 versus 6.6956 (7 days; 175), and 20.0947 versus 14.6170 (30 days; 40). The monthly non-overlapping sample is small: eight origins per coin.

Future feature or model choices made after inspecting these results require a new untouched final-test period for an honest final assessment.

## Forecasts and verification

The initial report contains 15 forecasts at the 10 October 2026 08:00 UTC origin. Run `download --config config.example.json` to refresh completed candles, then `predict --run-dir runs/multiframe-coins --data-dir data` to regenerate forecasts using frozen weights. The combined model rejects origins older than eight hours unless historical inference is explicitly requested. Commands run on demand; no background schedule was installed.

Verification: 131 tests passed, including analytical exact targets, future-perturbation causality, each branch's volume features, OHLCV volume aggregation, missing and incomplete candle handling, timestamp units, global label purging, train-only branch scalers, real gradient updates in all four encoders, checkpoint reconstruction, recent-REST continuity and failure preservation, freshness and daily compatibility. The combined synthetic CLI demo completed successfully. An independent review found and corrected refresh metadata coverage counts. Real saved-weight CPU inference and source/artifact integrity checks were also performed.

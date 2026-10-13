# First historical deep learning experiment

The frozen research run is saved in `runs/selected-coins`. It uses BTC, ETH, SOL, XRP, and ADA, with 2,108 completed daily USDT spot candles per coin from January 1, 2021 through October 9, 2026. Every requested day is present. Binance public archive checksums and the exact effective input hashes are recorded in the data sidecars and run summary.

## Protocol

The configuration was fixed before evaluation: 60-day lookback, 48 GRU hidden units, 8-dimensional coin embedding, AdamW learning rate 0.001, batch size 128, seed 42, at most 60 epochs, and validation patience 10. CUDA training stopped at epoch 15 and retained epoch 5. Feature and target normalizers use training windows only. Asset-balanced sampling gives each coin equal training exposure.

Shared validation starts January 22, 2025; final test starts November 16, 2025. Thirty training and thirty validation origins per coin are removed because their labels reach the next split. Each coin contributes 1,362 training, 268 validation, and 299 final-test origins. Model weights changed during gradient training. The selected model and all experiment settings are retained without tuning on final-test results.

A first acquisition check excluded SOL after connection failures left its recent history unavailable; that four-coin run is preserved in `runs/acquisition-partial`. Retrying the failed public downloads recovered the full SOL history. The completed five-coin run uses the same predefined architecture and hyperparameters; no model choices were changed in response to test performance.

## Final-test results

These are coin-balanced mean absolute errors in percentage-return points. All methods are evaluated on identical origins and outcomes.

| Horizon | GRU | Unchanged price | Trailing 7-day drift |
| --- | ---: | ---: | ---: |
| 1 day | 2.2974 | 2.2718 | 2.4302 |
| 7 days | 6.3107 | 6.0363 | 9.2200 |
| 30 days | 14.6737 | 12.8462 | 31.6616 |

The GRU beats the drift baseline but underperforms unchanged price at every horizon. This implementation supplies a reproducible training and forecasting workflow; this experiment does not establish forecasting advantage or trading profitability. Further architecture or feature selection requires a fresh evaluation protocol and new untouched data.

Empirical 80% validation-residual ranges achieved 85.42%, 86.42%, and 78.53% test coverage for 1, 7, and 30 days. These pooled ranges also use the epoch-selection validation period, so they are empirical estimates rather than independently calibrated probability guarantees. The report includes per-coin metrics and non-overlapping forecast origins to expose dependence between neighboring labels.

## Forecast outputs

There are 15 price/change forecasts, anchored to October 10, 2026 at 00:00 UTC, using the October 9 close. Their target boundaries are October 11, October 17, and November 9 at 00:00 UTC. They are USDT market-price estimates, with exactly 30 days representing the month. The trained model has not been refitted on validation or test data.

To reproduce inference using saved weights and local history:

```powershell
python -m crypto_forecast predict --run-dir runs/selected-coins --data-dir data
```

Use `report.html`, `summary.json`, and the CSV/JSON forecasts in that run directory for exact values, provenance, and evaluation details.

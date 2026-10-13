# Kronos-base integration verification

Downloaded the original NeoQuasar/Kronos-base predictor into models/kronos-base/model and the official base tokenizer into models/kronos-base/tokenizer. Exact revisions and file checksums are recorded in models/kronos-base/manifest.json.

- Predictor revision: 2b554741eca47781b64468546e77fef3e85130e6.
- Predictor SHA256: abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83.
- Tokenizer revision: 0e0117387f39004a9016484a186a908917e22426.
- Original Kronos-small predictor and tokenizer checksums remain unchanged.

Both chart model selectors offer Kronos-base original weights. Training & fine-tuning offers Kronos-base with the existing optional tokenizer training workflow and a separate checkpoint. The example configuration is config.kronos-base.example.json.

Real browser test forecast-20261011-123723-1d7ffd10 completed on CUDA in approximately 9 seconds: BTC 4h candles, anchor 2025-11-20 08:00 UTC, 30 input candles and 6 future candles. The result preserves observed price and volume for comparison and displays lookback and forecast windows. Screenshot: 2026-10-11-kronos-base.png.

Validation: 369 Python tests passed; 11 JavaScript chart tests passed; app.js, chart.js and testing.js syntax checks passed. Tiny-model tests exercise Base inference file selection, fine-tuning checkpoint identity, prediction reload and dashboard local path selection. No full-size fine-tuning run was started.

# Kronos-small integration verification

The official Kronos-small predictor and base tokenizer are installed in
`models/kronos-small/model` and `models/kronos-small/tokenizer`. Their exact
Hub revisions are recorded in `models/kronos-small/manifest.json`; the vendored
model implementation is pinned to `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.

The local `.venv-kronos` environment uses the existing CUDA PyTorch installation.
The loaded predictor contains **24,741,376 parameters**; its tokenizer contains
**3,958,042 parameters**.

Verification completed:

- Full regression suite: **305 passed**. The Windows sandbox blocked loopback
  connections on the first attempt; the complete suite passed outside the sandbox.
- Browser: Kronos selection, fine-tuning presets, disabled recurrent hidden size,
  single-stream preview, daily context limit, and restoration of LSTM settings.
- Real pretrained checkpoint: two-step inference on local BTC 4-hour candles.
- Real fine-tuning: one predictor optimizer step on local BTC daily candles,
  frozen tokenizer, chronological purged splits, validation and final-test paths,
  and a saved self-contained checkpoint in `runs/kronos-small-smoke`.
- Offline reload: reproduced the smoke run's 1-, 7-, and 30-day forecasts exactly
  on the same CUDA device without loading weights from Hugging Face.

The smoke run selected **one training, one validation, and one test origin** using
strides of 10,000. It checks software integration and is far too small to assess
forecast accuracy. Full fine-tuning uses `config.kronos.example.json`, which
defaults to strides of 6, batch size 8, five epochs, and learning rate 1e-5.

```powershell
.\.venv-kronos\Scripts\python.exe -m crypto_forecast train --config config.kronos.example.json
```

Edit that configuration's coins, date range, and data folder for the desired run.
The dashboard can be started with the same Python environment:

```powershell
.\.venv-kronos\Scripts\python.exe -m crypto_forecast dashboard
```

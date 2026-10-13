# Learning-rate scheduler verification

Training supports fixed, warmup/cosine, validation plateau, step, linear and
learning-rate-only OneCycle policies. GRU, LSTM and both Kronos predictors use
the same policy implementation. Kronos tokenizer training has independent
settings and history. New predictor runs default to warmup/cosine, 5% warmup
and a minimum of 10% of the configured peak; tokenizer defaults to OneCycle
with 3% warmup. AdamW momentum settings are retained.

The training tab shows policy-specific controls, calculated minimum rates,
plateau/early-stopping guidance, live batch rates and mean rates per epoch.
The curve selector supports predictor and tokenizer histories. Rates are saved
in summary JSON and standalone reports. Smooth schedules hold their minimum
during user-approved extra epochs rather than restarting or exceeding their
planned update count.

Numerical tests exercise warmup, decay, validation-driven reduction, epoch
boundaries, invalid settings, one-update runs and extension behavior. Actual
tiny Kronos-small and Base fine-tuning tests exercise independent tokenizer and
predictor schedules, saved configuration and checkpoint reloads. GRU and LSTM
training tests verify real changing rates and saved histories. Worker tests
distinguish epoch mean rates from live last-update rates.

Browser verification:

- A three-epoch tiny synthetic Kronos run is saved as
  `runs/lr-scheduler-synthetic-proof`. It records a linear tokenizer schedule
  separately from the predictor's warmup/cosine schedule. This run verifies
  software behavior; its weights are deliberately tiny, not downloaded weights.
- Browser job `train-20261011-132059-9a05b8f6`, named
  **Learning-rate browser verification**, completed two CPU GRU epochs on saved
  BTC history from 2023-01-01 through 2025-01-01. The browser submitted linear
  decay; rates started at 0.001 and finished at 0.0001. Live job history and the
  learning-rate curve were verified. The original coin choices and training
  date defaults were restored afterward, with Kronos-base and warmup/cosine
  selected for the next run.
- Freezing the tokenizer disables its schedule controls. Saved tokenizer and
  predictor curves can each be selected. Browser console has no errors.
- Screenshot: `2026-10-11-learning-rate-schedules.png`.

Final validation: **396 Python tests passed**, **11 JavaScript chart tests
passed**, and app.js passed its syntax check. The restricted verification run
blocked local dashboard HTTP connections; rerunning with local socket access
passed. Original Kronos-small and Kronos-base predictor SHA256 hashes are
unchanged from their download manifests.

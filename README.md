# Cryptocurrency deep learning forecasts

This project predicts future prices and market movement as candle sequences with **Kronos-small** or **Kronos-base**, using original weights or optional fine-tuning on your own data. Kronos fine-tuning supports **15-minute, 1-hour, 4-hour, and daily candles** with a configurable forecast length. Reports evaluate the price and direction at every future candle. The **Chart forecasts** and **Model testing** tabs generate a chosen number of future OHLCV candles in USDT. The browser workspace also retains the GRU/LSTM models with their original 1-, 7-, and 30-day targets.

Kronos forecast origins follow the selected candle interval. Its example configurations use **one-hour candles, 120 input bars, and 24 forecast candles**. The reference price is the close at the most recent eligible origin. “Value” means observed market price, not intrinsic fair value. This is a forecasting research package; it does not place orders.

## Configurable Kronos forecast targets

In **Training & fine-tuning**, choose **Candle timeframe** and **Forecast candles**.
For example, 24 hourly candles cover the next 24 hours, while 168 hourly candles
cover seven days. There is no mandatory day/week/month forecast target.
Fine-tuning learns every candle in that path; reports retain each predicted close,
percentage movement, and held-out direction accuracy. Saved runs offer their
actual forecast horizons, including minutes and hours.

The predictor training sequence must satisfy **input bars + forecast candles − 1 ≤ 512**.
With 120 input bars, you can train on up to 393 future candles. This model context
limit remains; the app no longer reserves space for a compulsory 30-day target.
Chart forecasting can choose a different output length using the rolling inference
context, but a saved fine-tuned model must use its trained candle interval.

Kronos data preparation checks valid candle windows and chronological split
boundaries with compact indices, then copies only the origins selected by the
training and evaluation strides. The splits, boundary purges, excluded coins,
and selected examples retain their original definitions.

**Data preparation workers** selects Automatic, serial, or 2–4 CPU processes.
Automatic uses serial preparation below 50,000 selected windows; larger runs
use up to four workers, bounded by available CPU cores and eligible coins.
Workers read shared temporary candle arrays and return bounded batches; the
parent alone writes progress. Cancelling a dashboard job also stops its CPU
workers. GPU tokenization and model training retain their existing batching.
The same setting is available as `--kronos-preparation-workers` or the JSON
field `kronos_preparation_workers` (`0` automatic, `1` serial, `2`–`4` parallel).

Preparation benchmarks can be reproduced with
`python -m scripts.benchmark_kronos_preparation --mode legacy`, `--mode serial`,
or `--mode parallel --workers 4`. These only prepare samples; they do not train
a model. Use separate invocations to compare peak parent RAM. Synthetic results
and a representative local-history comparison are recorded in
`docs/experiments/2026-10-11-kronos-preparation.md`.

CLI example:

```powershell
.\.venv-kronos\Scripts\python.exe -m crypto_forecast train --config config.kronos-base.example.json --kronos-timeframe 1h --kronos-prediction-length 24 --lookback 120 --batch-size 1
```

Old checkpoints keep their original 1/7/30-day definitions. An explicit
`"kronos_prediction_length": null` retains that legacy training mode in JSON;
new configurations default to 24 future candles. Restart the dashboard process
and reload its browser page after updating the application.

## Learning-rate schedules

Training & fine-tuning includes a **Model learning-rate schedule** selector for
GRU, LSTM, Kronos-small and Kronos-base. Kronos tokenizer training has its own
independent schedule selector. The model defaults to **Warmup + cosine decay**:
5% warmup, then decay from the configured peak to 10% of that peak. For a Kronos
peak of `0.00001`, the minimum is `0.000001`. Tokenizer training defaults to
**OneCycle (learning rate only)** with 3% warmup. Both retain AdamW's configured
momentum settings.

Choose fixed, warmup/cosine, validation plateau, step, linear, or OneCycle.
The UI displays only the settings used by the selected policy and shows the
calculated minimum. Smooth policies change after each optimizer update; step
decay changes after the chosen number of epochs, and plateau reduction monitors
validation loss after each epoch. Plateau patience is the tolerated number of
non-improving epochs; reduction happens when that count is exceeded. Set
early-stop patience high enough to train at the reduced rate.

The training monitor shows the latest batch rate and a curve of the mean rate
used per epoch. Select **Tokenizer** or **Model / predictor** to inspect either
stage, including saved runs. Reports and summary JSON retain separate curves
and histories. Older runs without recorded rates show an empty rate chart.
Smooth schedules reach their minimum at the last planned update and hold that
minimum if you choose to train extra epochs. Very short runs shorten or skip
warmup; a single-update run uses the peak once.

CLI example (the same fields work in JSON configuration):

```powershell
.\.venv-kronos\Scripts\python.exe -m crypto_forecast train --config config.kronos-base.example.json --lr-scheduler warmup_cosine --lr-warmup-fraction 0.05 --lr-min-ratio 0.1
```

Other controls: `--lr-factor`, `--lr-step-epochs`, `--lr-patience`. Tokenizer
controls use the `--kronos-tokenizer-` prefix, for example
`--kronos-tokenizer-lr-scheduler plateau`. Learning-rate policies affect training
only; downloaded original weights and read-only chart forecasting stay unchanged.

## Install

Use Python 3.11 or newer. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

The package uses NumPy, pandas, and PyTorch. `--device auto` uses CUDA when available and otherwise CPU. Use `--device cpu` for an explicit CPU run; a CUDA run requires a compatible PyTorch installation and GPU.

## Kronos-small on your own data

The integration uses the official [Kronos](https://github.com/shiyu-coder/Kronos) tokenizer and Transformer implementation, with the pretrained [Kronos-small predictor](https://huggingface.co/NeoQuasar/Kronos-small) and [base tokenizer](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base). Kronos-small has approximately 24.7 million predictor parameters and a 512-bar context. Its architecture is fixed; the GRU/LSTM `hidden_size` setting does not resize it.

Install the optional dependencies and download the pretrained weights once:

```powershell
python -m pip install -e ".[kronos,test]"
python -m crypto_forecast kronos-setup --model-dir models/kronos-small
```

This workspace also has a ready `.venv-kronos` environment using the existing CUDA PyTorch installation. Run commands with `.\.venv-kronos\Scripts\python.exe`, or activate it with `.\.venv-kronos\Scripts\Activate.ps1`. To create the same environment on another machine with PyTorch already installed, use `python -m venv --system-site-packages .venv-kronos` and install the optional dependencies there.

**Fine-tuning is optional.** The original downloaded model and tokenizer remain in `models/kronos-small/model` and `models/kronos-small/tokenizer`. Forecasting never saves weights, and fine-tuning saves a separate checkpoint under `runs/`.

**Kronos-base** is also supported. Download its official pretrained predictor
and base tokenizer into a separate folder:

```powershell
.\.venv-kronos\Scripts\python.exe -m crypto_forecast kronos-setup --model kronos-base
```

Choose **Kronos-base · original weights** in either Chart forecasts or Model
testing. Its files live under `models/kronos-base/`; model selection never
modifies either variant's downloaded weights. Both models support 512 input
candles and this workspace's existing 512-prediction limit. The download manifest
records the exact Hugging Face revisions and file hashes. Base's larger model
requires more computation than Small.

For optional fine-tuning, choose **Kronos-base (fine-tune)** in Training &
fine-tuning, or run:

```powershell
.\.venv-kronos\Scripts\python.exe -m crypto_forecast train --config config.kronos-base.example.json
```

Base uses the same historical splits and optional tokenizer-training workflow,
saving a separate self-contained checkpoint under `runs/`. Both original models
and saved checkpoints remain selectable for forecasting and model testing.

Start the dashboard:

```powershell
.\.venv-kronos\Scripts\python.exe -m crypto_forecast dashboard
```

Open `http://127.0.0.1:8765` and use the default **Chart forecasts** tab:

1. Keep **Kronos-small · original weights** selected. Choose a **coin**; its downloaded price history is loaded automatically, with no file selection.
2. Select **15m, 1h, 4h, or 1d** candles and how far ahead to forecast in **hours, days, or candles**. For example, 24 hours with 4-hour candles generates six candles, while seven days generates 42. Partial periods are rounded up to complete candles and disclosed. Requests exceeding 512 future candles are rejected.
3. Choose **2–512 input candles**. The [official GitHub README](https://github.com/shiyu-coder/Kronos#-making-forecasts) specifies a 512-candle context for Kronos-small and recommends staying within it; longer inputs are truncated by the upstream predictor. This is a supported context limit, not a guarantee that more input candles improve forecast accuracy.
4. Optionally set a UTC cutoff to select a historical window. Only completed candles before that cutoff enter the model; gaps in the selected context are rejected.
5. Click **Preview chart** or **Forecast future candles**. Forecast candlesticks appear after a marked boundary, with an OHLCV table, **Download CSV**, and full JSON containing raw outputs.

The interactive chart uses locally bundled [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts).
Drag to pan, scroll/pinch to zoom, and drag either axis to scale; double-click an
axis to reset. **Fit all** restores the full chart and **Show forecast** focuses
the future window. Historical and forecast volume bars share the price chart's
time axis in a separate resizable pane. Purple bars identify forecast values.
Price is labeled in **USDT**, volume in the **selected base coin**, and timestamps
in **UTC**. The crosshair shows OHLCV values and candle opening times. Keyboard
controls: +/− zoom, arrows pan, Home fits all, End shows the forecast.

Forecast jobs run locally and can be stopped. Closing the browser leaves an active job running while the dashboard process stays open. Previous forecasts remain selectable after reconnecting; their selected input window and output are saved under `.dashboard/charts/JOB-ID/`. A historical/stale chart is explicitly labeled: its forecast starts after the selected last candle, not after the current time. Refresh/download saved market history in **Training & fine-tuning** when you need newer candles.

Use **Model testing** to compare a forecast with what happened after a historical
point. Choose original Kronos-small or a saved fine-tuned Kronos checkpoint, a
coin, and a candle interval. The chart loads **all completed saved history**;
the 512-candle context limit does not truncate the displayed chart. Enter
**Lookback candles** (2–512) and use **Forecast for** with **Hours, Days, or
Candles**, just as in Chart forecasts. The calculated future candle count and
duration appear below the controls; partial candles round up and requests beyond
512 future candles are rejected. Click a candle in the price pane to start a
test. Both tabs share the same original/fine-tuned model choices. The selected candle is the last model
input; predictions begin at the next candle opening. The selected lookback must
be contiguous and contain enough completed candles. Dragging pans the chart
without starting a test.

Purple outlined price candles and purple volume overlay the observed candles.
Green shading marks the exact lookback supplied to the model, including the
selected candle; purple shading marks the future forecast window. Both shades
span the price and volume panes and track panning and zooming. **Show test
window** fits the entire lookback and forecast together; **Fit full chart**
restores the entire history. The crosshair and comparison table show predicted and observed
values. Future observations never enter the predictor. Unavailable observations
appear as a dash, including predictions beyond the saved history. Tests can be
stopped and restored through **Saved model tests**, with their original input
and comparison snapshot retained locally. Change the counts and use **Predict
from selected candle** to rerun that point. All weight files remain unchanged.

The amount/quote-turnover channel is zero, matching the project's fine-tuning preprocessing. Multiple sampled paths are averaged. Outputs are estimates, not observed candles or calibrated confidence intervals. For valid candlesticks, predicted high/low ranges are expanded to include open/close and negative volume is clipped to zero; any adjustments are disclosed and raw outputs remain in JSON.

If you later fine-tune Kronos, its separate saved run becomes selectable in **Model weights**. Choose that run's trained candle interval. The original model remains the default and retains its downloaded weights.

Place your own completed **4-hour** candles in `data/4h/BTC.csv`, `data/4h/ETH.csv`, etc., using the [CSV format below](#local-data-format). Kronos requires only the selected timeframe. Daily fine-tuning uses `data/1d/COIN.csv` and `--kronos-timeframe 1d`. Use your exact symbols, UTC opening timestamps, prices in USDT, and base-asset volume. Missing candles exclude affected input/target windows; the loader never fills them. The tokenizer also has a quote-turnover input channel; because this project's CSV format does not contain quote turnover, that channel is explicitly set to zero.

Edit `config.kronos.example.json` to select your coins, date range, and data folder, then fine-tune and predict:

```powershell
python -m crypto_forecast train --config config.kronos.example.json --coins BTC ETH --run-dir runs/kronos-my-data
python -m crypto_forecast predict --run-dir runs/kronos-my-data --data-dir data
```

Use `download --config config.kronos.example.json` if you want the existing Binance acquisition workflow to supply the selected candle series instead. The download command derives all four candle resolutions; Kronos training reads only its selected resolution.

The training method follows **tokenizer → predictor**, either as two independent jobs or one combined job. CLI configurations retain combined training by default for compatibility. Both components start from the downloaded pretrained weights. First, the tokenizer learns from normalized training windows using `(coarse reconstruction MSE + full reconstruction MSE + BSQ loss) / 2`. Full reconstruction MSE on validation windows selects its best epoch. Training then restores that tokenizer checkpoint, freezes it, and generates fresh training/validation tokens for the predictor. **All predictor weights are fine-tuned** with future-token cross-entropy loss; predictor validation selects its own best epoch. Each stage uses the same chronological partitions, with test data reserved for final evaluation.

Normalization uses each sample's completed input window only; future candles use the same historical mean/std. The model autoregressively generates the configured number of future candles, and every future close supplies a price and movement forecast. `kronos_sample_count` controls how many generated paths are averaged. Both stages pool your selected coins without adding a learned coin embedding, and weight coins equally despite differing history lengths. The amount channel remains zero when quote turnover is unavailable, as allowed by the [Kronos CSV workflow](https://github.com/shiyu-coder/Kronos/blob/master/finetune_csv/README.md).

Tokenizer controls are independent: `kronos_tokenizer_epochs` defaults to 5, `kronos_tokenizer_patience` to 2, and `kronos_tokenizer_learning_rate` to `2e-4`. Its AdamW optimizer uses betas `(0.9, 0.95)`, weight decay `0.1`, a configurable learning-rate-only OneCycle schedule with 3% warmup by default, and gradient clipping at 2.0. The reconstruction objective follows the [upstream tokenizer trainer](https://github.com/shiyu-coder/Kronos/blob/master/finetune/train_tokenizer.py). The BSQ entropy term can make the combined training loss negative; validation reconstruction MSE remains nonnegative and is a different objective. Tokenizer early stopping finishes a tokenizer-only job or advances to the predictor in combined mode. `kronos_tokenizer_batch_size` overrides the shared batch size for the tokenizer; omitting it retains older configurations' shared-batch behavior. Training/evaluation strides apply to both stages. Predictor settings remain the custom future-only objective and training controls described here.

For predictor-only training, set `kronos_finetune_tokenizer` to `false` or pass `--no-kronos-finetune-tokenizer`. This retains the pretrained tokenizer unchanged. Existing predictor-only checkpoints remain usable.

You can now run tokenizer and predictor tuning independently. In the dashboard,
choose **Tune tokenizer only**, set the tokenizer batch size (for example, 36),
and start training. Each new best validation epoch is saved immediately as
`tokenizer.pt`, with constructor settings, data provenance, split boundaries,
and training history in `tokenizer_summary.json`. Stopping a tokenizer job after
a validation best leaves that checkpoint reusable. No predictor is loaded and
no final-test forecasts are generated in this stage.

Next choose **Tune predictor only**, select the saved tokenizer, and use its
saved data settings. Set the predictor batch size independently (for example,
16). The saved tokenizer stays frozen, including its normalization and token
vocabulary. Predictor optimization starts from the original pretrained predictor;
this is a new tuning run, not a resume of an interrupted optimizer. Validation
selects the best predictor before final-test evaluation. The resulting `model.pt`
contains both matched components and needs no external tokenizer at inference.

The CLI equivalents are:

```powershell
python -m crypto_forecast train --config config.kronos.example.json --kronos-training-stage tokenizer --kronos-tokenizer-batch-size 36 --run-dir runs/my-tokenizer
python -m crypto_forecast train --config config.kronos.example.json --kronos-training-stage predictor --kronos-tokenizer-run-dir runs/my-tokenizer --batch-size 16 --run-dir runs/my-predictor
```

Both commands must use the same candle data, coins, input/forecast windows,
chronological shares, and sampling strides. Reuse rejects changed data or
partitions, preventing a tokenizer selected on one partition from contaminating
another run's validation or final test. Learning rates, epoch limits, device and
batch sizes can differ. CLI source selection also supports an earlier completed
Kronos run containing a fine-tuned tokenizer and its `summary.json`.
**Tokenizer, then predictor** remains available for combined runs and also saves
the tokenizer independently before predictor preparation. Existing stopped runs
from before this change cannot recover weights that were never checkpointed.

The example configurations use batch size 1, learning rate `1e-5`, and 5 epochs. A CUDA GPU is recommended; CPU runs work but generating long candle paths can take substantial time. The maximum lookback is **513 minus the selected forecast length**, leaving room for future tokens inside the pretrained context. Use the following overrides to adjust cost:

```powershell
python -m crypto_forecast train --config config.kronos.example.json --kronos-timeframe 1d --lookback 60 --batch-size 8 --learning-rate 0.00001 --epochs 5 --kronos-train-stride 6 --kronos-eval-stride 6 --kronos-sample-count 1 --run-dir runs/kronos-daily
```

Training and evaluation strides select every Nth eligible origin **per coin, after chronological splitting**. The default stride of 6 reduces training and path-generation work. Set either stride to 1 to use every eligible origin in that stage. Reports disclose the evaluation stride, and every baseline uses the same sampled held-out origins as Kronos. Scores from differently sampled runs need that context before comparison.

All coins share chronological train/validation/test boundaries. Samples whose selected future targets reach the next split are purged; validation chooses the best epoch and calibrates residual ranges, and the final test evaluates that saved epoch. These local splits prevent future-data leakage during fine-tuning. **Overlap between your historical dates and upstream Kronos pretraining is unknown**: a held-out local test does not prove that those dates were unseen during pretraining. Prefer new data beyond known pretraining coverage when making claims about out-of-sample performance.

The downloaded model/tokenizer folders are used by the example configuration. CLI/config `kronos_model_path` and `kronos_tokenizer_path` can also reference Hugging Face model IDs; the first such run needs network access, while cached/local weights support offline use. The dashboard automatically uses `models/kronos-small/model` and `models/kronos-small/tokenizer` when both folders exist.

Each fine-tuned `model.pt` includes the matched tokenizer/predictor weights and constructor settings, so prediction does not contact Hugging Face. The checkpoint and `summary.json` record tokenizer training history, its selected epoch, and whether tokenizer weights changed, separately from predictor history. Reports show both learning curves and identify whether the tokenizer was fine-tuned. Prediction uses the run's saved device setting. Seeded sampling is reproducible on the same device and software environment; changing CPU/GPU or hardware can change generated paths. `auto` can choose a different device when moving the checkpoint to another machine.

## Browser model workspace

Start the local dashboard from this folder:

```powershell
.\.venv-kronos\Scripts\python.exe -m crypto_forecast dashboard
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Select coins, dates, and training settings, then choose **Train model** or **Fine-tune Kronos-small**. Use **Download candles** first when the selected history is missing. GRU/LSTM combine 15m, 1h, 4h, and daily candles; Kronos uses its selected 4-hour or daily stream. All models include volume and forecast 1, 7, and 30 days ahead.

All architecture and training controls are grouped under **Model settings**. **Available models** selects **Multi-timeframe GRU**, **Multi-timeframe LSTM**, or **Kronos-small (fine-tune)**. Set the input lookback, maximum epochs, early-stop patience, batch size, learning rate, device, and random seed there. Hidden size applies to GRU/LSTM; it is disabled for Kronos. Selecting Kronos reveals its timeframe, forecast-path count, training/evaluation strides, and tokenizer training controls. Tokenizer training defaults to enabled, with 5 epochs and learning rate `2e-4`; predictor training starts with 5 epochs, batch size 8, and learning rate `1e-5`. The live monitor shows each stage separately. Switching back restores the previous recurrent training settings. **Training share** and **Validation share** configure the chronological splits as decimals; the remaining share is reserved for the final test. Their sum must be less than 1.

The **Model preview** updates while you edit settings. GRU/LSTM previews show the four candle streams, recurrent encoders, coin embedding, shared prediction head, and 1/7/30-day outputs. The Kronos preview shows one selected timeframe, its frozen tokenizer, the fixed pretrained Transformer, and generated future candles. Both display input-window durations and trainable parameter counts. Hover over a setting or its label to see a short explanation; focusing a control with the keyboard shows the same help. The preview describes the next run, while the monitor and saved results describe actual jobs.

Enter comma-separated base symbols under **Additional coins** and click **Add coins** (or press Enter). Each becomes a selectable button beside the existing coins. The browser remembers your coin buttons and selections after a refresh; coins in saved data or past experiments also appear automatically. Starting a download or training run adds any symbols still in the field.

Right-click any coin button and choose **Remove COIN** to remove it from the choices and future selections. The browser remembers removals, including default coins and coins discovered from saved experiments. Add the symbol again to restore its button. Removing a button keeps its downloaded candles and saved models. Keyboard users can focus a coin and press Shift+F10 to open the same menu.

The monitor shows the current stage, epoch and batch progress, elapsed time, training and validation loss curves, sample counts, and activity logs. **Stop training** cancels the active job. Closing the browser leaves training running; reopen the page to reconnect. Keep the dashboard process running while a job is active. Closing that process stops its worker, and unfinished jobs are marked interrupted on the next launch.

When a browser training run reaches **Early-stop patience**, it pauses and asks whether to **Train 10 more epochs** or **Finish and save**. Continuing keeps the same model and training state, runs exactly 10 additional epochs, and asks again. You can repeat this as often as you choose; finishing saves the best validation epoch across the whole run. A waiting choice stays available after refreshing or reopening the page. **Stop training** still cancels the run while it waits.

Saved experiments include forecasts, validation and final-test comparisons, and a full report. Each new run gets its own folder under `runs/`; job history is stored in `.dashboard/`. One job can run at a time per workspace. The server accepts local connections only. Use `--port 8766` to choose another port, or `--workspace PATH` to use another data and runs folder.

The **Download candles** button verifies saved native 15-minute candle coverage for the requested dates and fetches only missing ranges on the saved exchange. Candles outside the requested range are retained; missing larger timeframes are rebuilt from the native candles. Dates before the exchange's recorded first candle are skipped. Two coins prepare in parallel, with up to four HTTP requests shared across all coin, page, and archive fetchers in the download worker. Missing-page and KuCoin historical-page requests also run in parallel. Each completed coin is staged, verified, and saved with all four timeframes through the dashboard's single publication coordinator, independently of other coins. Failed coins are reported while the remaining coins are still attempted. Stopping a download keeps completed coins saved and preserves the previous files for unfinished coins. Failed download diagnostics remain in that job's `.dashboard/jobs/JOB-ID-data/download_report.json`.

## Train on real historical data

The default configuration selects **BTC, ETH, SOL, XRP, and ADA**, with 60 candles per timeframe. Edit `config.example.json` to change the selected coins, dates, model architecture and size, learning rate, or training limits. Its `input_mode` is `multiframe`, `model_type` is `gru`, and `origin_interval` is `4h`. Set `model_type` to `lstm` for an LSTM experiment.

```powershell
python -m crypto_forecast download --config config.example.json
python -m crypto_forecast train --config config.example.json
python -m crypto_forecast predict --run-dir runs/multiframe-coins --data-dir data
```

Open `runs/multiframe-coins/report.html` to inspect the forecasts, learning curve, validation metrics, final-test baseline comparisons, and exclusions. Downloading needs network access; training and prediction use local data. The download range includes `start` and excludes an explicit `end`. With `end: null` (the default), downloading adds current completed candles using the selected exchange's public market-data API (Binance or KuCoin). `--latest` also requests this append when an explicit end is set. Inspect each forecast's `as_of` timestamp.

The origin interval defines training samples and eligible prediction times (4 hours for combined models; the selected timeframe for Kronos). To generate a fresh forecast, refresh the local data and rerun `predict`. These commands run on demand; they do not install a background scheduler.

The CLI accepts overrides:

```powershell
python -m crypto_forecast download --config config.example.json --coins BTC ETH SOL --start 2021-01-01 --end 2026-10-10 --data-dir data
python -m crypto_forecast train --config config.example.json --input-mode multiframe --origin-interval 4h --epochs 60 --run-dir runs/experiment-01 --device cpu
python -m crypto_forecast predict --run-dir runs/experiment-01 --data-dir data --coins BTC ETH
```

To train the alternative architecture in a separate run:

```powershell
python -m crypto_forecast train --config config.example.json --model-type lstm --run-dir runs/lstm-experiment
```

Keep the download coin selection and training configuration aligned. Use a new run directory for each experiment; replacing an existing trained run requires the explicit `--overwrite` option. Prediction accepts only coins with a trained embedding and sufficient valid history. The combined model rejects forecast origins older than **8 hours** by default (daily compatibility mode uses 2 days); `--max-age-days` overrides the limit. For deliberate historical inference, pass `--allow-stale`; the output remains labeled as historical/stale.

The earlier candidate group can be configured as BTC, ETH, BNB, XRP, SOL, ZEC, HYPE, ADA, XLM, NEAR, UNI, LTC, SUI, ENA, and XAUT. Each needs valid history in all four timeframes and enough samples. The downloader compares Binance and KuCoin spot history automatically, including for HYPE and XAUT. Listing a coin does not guarantee enough history on either exchange. Matching local CSVs can supplement unavailable assets; assets and quote units are never silently substituted. USDT is the quote unit and is not a forecast target.

## Local data format

Data files are grouped by timeframe: `data/15m/BTC.csv`, `data/1h/BTC.csv`, `data/4h/BTC.csv`, and `data/1d/BTC.csv`, with the same structure for every selected coin. Each CSV contains one spot candle per row:

```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42283.58,42350.00,42260.00,42320.00,210.30
2024-01-01T00:15:00Z,42320.00,42390.00,42300.00,42370.00,195.38
```

The rows above illustrate a 15-minute file only; use actual OHLCV values from your chosen source. `timestamp` identifies the candle's **UTC opening time** and must align with its timeframe. Prices must be positive and quoted in USDT; volume is the base-asset traded volume. Preserve the source's exact identity and units. The loader rejects invalid OHLCV and duplicate timestamps, excludes incomplete candles, and never fills gaps. Samples require contiguous history for each input branch and exact future target timestamps.

The downloader checks the actual first available 15-minute candle for each asset/USDT pair on **Binance Spot and KuCoin Spot**, then tries the exchange with the older start date (Binance wins equal dates). If that download is incomplete or unusable, it checks the other exchange and prefers usable post-listing coverage before the oldest start date. One exchange supplies every candle for an asset; exchange prices and volumes are never spliced. The requested date range stays unchanged, so dates before listing remain unavailable.

Binance candles come from its [official public spot archives](https://github.com/binance/binance-public-data), with published SHA256 checksums verified. KuCoin candles come from its [public candle API](https://www.kucoin.com/docs-new/rest/spot-trading/market-data/get-klines), paginated in windows of at most 1,500 candles. KuCoin's field order is normalized to the same OHLCV schema. Completed response windows are cached with local SHA256 hashes; these detect cache corruption and are not exchange-published checksums. First-candle comparisons are cached for one UTC day; `--refresh` bypasses the caches.

The downloader derives the 1-hour, 4-hour, and daily series from complete UTC buckets: first open, maximum high, minimum low, last close, and **sum of volume**. Buckets with missing or unfinished constituent candles are excluded. Companion metadata and `data/download_report.json` record the selected exchange, both first-candle dates, attempted download coverage, acquisition failures, and unavailable dates. Recent refreshes stay on the selected exchange. The daily-only compatibility downloader uses the same exchange selection rule. Binance's historical millisecond and microsecond timestamp formats are both supported. Keep provenance metadata with supplemental CSVs so the source and quote currency are auditable.

A multiframe download is published only when every required timeframe has at least one usable candle. Otherwise it reports failure with `persisted: false` and preserves existing candle files, so an incomplete exchange switch cannot mix old and new sources.

The daily-only compatibility mode uses `data/BTC.csv`, `data/ETH.csv`, and so on. Existing daily checkpoints remain usable with their original data layout; the new combined model requires a new training run.

## Model and evaluation

The recurrent architectures use [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) or [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html). GRU uses fewer parameters at the same hidden size; LSTM maintains a separate memory cell. Both use a separate encoder for each timeframe. Their outputs are concatenated with a learned coin embedding, and a shared prediction head produces cumulative log returns for 1, 7, and 30 days. Each recurrent forecast uses all four candle resolutions together. A forecast return is converted into price using the close at its 4-hour origin. Kronos uses the single-stream tokenizer/Transformer method described [above](#kronos-small-on-your-own-data). The selected architecture is saved with the run and reconstructed automatically for prediction; earlier checkpoints without an architecture field retain GRU behavior.

Each recurrent branch reads 11 causal features: returns, price ranges, momentum, volatility, **log traded volume**, log volume changes, and volume relative to recent 7- and 30-candle means. Each branch receives its own training-only normalization, allowing the model to compare activity across coins and timeframes. Momentum and volatility windows count candles within their own branch.

| Input branch | Default input window | Additional feature warmup |
| --- | --- | --- |
| 15 minutes | 60 candles / 15 hours | 30 candles |
| 1 hour | 60 candles / 60 hours | 30 candles |
| 4 hours | 60 candles / 10 days | 30 candles |
| 1 day | 60 candles / 60 days | 30 candles |

At each 4-hour origin, the model uses only candles whose closing boundary is at or before that origin. For example, a forecast at 08:00 UTC uses the daily candle that closed at 00:00 UTC; that day's unfinished daily candle is unavailable to the model. A missing required completed candle excludes the sample instead of silently substituting an older bar. Target prices occur exactly 1, 7, and 30 days after the forecast origin.

All assets share global chronological boundaries. The default split uses 70% for training, 15% for validation, and 15% for the final test. Configure `train_fraction` and `validation_fraction` in the configuration, CLI, or dashboard Model settings to change those shares; the remainder supplies the final test. Samples whose 30-day labels reach the following split are purged. Each branch's feature normalization and the target normalization are fitted on training samples only. Validation loss selects the best epoch and controls early stopping; the final test evaluates that frozen model.

Reports identify the selected GRU, LSTM, or Kronos-small architecture and compare it against unchanged-price and trailing mean-return baselines on identical held-out origins. Kronos reports disclose their sampling stride and frozen tokenizer. Reports include return MAE/RMSE in percentage points, log-return MAE, direction accuracy, sample counts, and per-coin and coin-balanced macro scores. Direction accuracy excludes zero actual or predicted returns, so it is undefined for the unchanged-price baseline. Additional metrics use non-overlapping origins for each horizon, because neighboring monthly targets otherwise overlap.

Empirical 80% ranges come from the selected model's **validation residuals**, pooled across eligible coins for each horizon, never final-test residuals. The same validation block selects the epoch, so calibration is not independent of model selection. Test coverage and range width show how these ranges behaved on held-out data. An 80% label is a calibration target, not a guaranteed future probability. Market conditions can change, and historical forecasts or directional accuracy do not establish profitability.

A first small training run checks that the model learns and supplies an initial baseline comparison. It does not establish an optimized forecasting system. After seeing final-test results, further model or feature choices need new untouched data for an honest final evaluation.

The completed **combined-timeframe** five-coin experiment is recorded in [its experiment note](docs/experiments/2026-10-10-multiframe-coins.md), with artifacts in `runs/multiframe-coins`. Final-test return MAE was 2.3142 / 7.0065 / 17.3087 percentage points for 1 / 7 / 30 days, compared with 2.2807 / 6.2197 / 13.2407 for unchanged price. This first combined model underperformed that baseline at all three horizons. The archived **daily-only** experiment is recorded [separately](docs/experiments/2026-10-10-selected-coins.md), with artifacts retained in `runs/selected-coins`; its scores use different origins and are not a controlled timeframe comparison.

## Saved artifacts

Each run directory contains:

| File | Purpose |
| --- | --- |
| `model.pt` | Selected weights and architecture, input mode, timeframes, model settings, and inference metadata. Kronos checkpoints also contain the frozen tokenizer weights and constructor configurations, so prediction needs no separate pretrained download |
| `summary.json` | Configuration, seed, source data hashes, exact splits, learning curve, metrics, and exclusions |
| `validation_predictions.csv` | Actual and predicted held-out validation outcomes |
| `test_predictions.csv` | Actual and predicted final-test outcomes with empirical ranges |
| `forecasts.csv`, `forecasts.json` | Forecast origin, target date, last close, price/change estimate, empirical range, and freshness status |
| `report.html` | Standalone readable report; no internet connection required |

Inspect the report's dates and data status before interpreting a forecast. Coins with insufficient training data appear as explicit exclusions rather than receiving an untrained prediction.

## Verify the pipeline

```powershell
python -m pytest -q --basetemp=.test-tmp/pytest
python -m crypto_forecast demo --run-dir runs/demo --epochs 3
```

Keep temporary test artifacts under `.test-tmp/`; that directory can be removed after verification.

The demo generates **synthetic 15-minute data**, derives all four timeframes, and runs combined-model training, evaluation, and reporting. It verifies the software path only; its prices and scores do not measure real cryptocurrency forecasting accuracy. Use the download/train workflow for historical market evaluation.

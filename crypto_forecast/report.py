"""Self-contained, escaped HTML reports for historical forecasting experiments."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, decimals: int = 2, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return _escape(value)
    if not math.isfinite(number):
        return "—"
    return _escape(f"{number * 100 if percent else number:,.{decimals}f}")


def _price(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return _escape(value)
    if not math.isfinite(number):
        return "—"
    return _escape(f"{number:,.8g}")


def _change(value: Any) -> str:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return _escape(value)
    if not math.isfinite(number):
        return "—"
    return _escape(f"{number:+,.2f}%")


def _horizon(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        minutes = value * 1440
        if minutes < 60:
            return f"{minutes:g} minutes"
        if minutes < 1440:
            hours = minutes / 60
            return f"{hours:g} {'hour' if hours == 1 else 'hours'}"
        return f"{value:g} {'day' if value == 1 else 'days'}"
    return _escape(value)


def _model(value: Any) -> str:
    return {
        "gru": "GRU",
        "lstm": "LSTM",
        "kronos-small": "Kronos-small",
        "kronos-base": "Kronos-base",
        "persistence": "Unchanged price",
        "drift": "Trailing drift",
    }.get(value, _escape(value))


def _ordinal(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool):
        return _escape(value)
    suffix = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _metric_table(rows: list[dict], *, include_coin: bool = False) -> str:
    if not rows:
        return '<p class="empty">No metrics are available for this split.</p>'
    headings = (["Coin"] if include_coin else []) + [
        "Horizon", "Model", "Samples", "Return MAE (pp)", "Return RMSE (pp)",
        "Log-return MAE", "Direction accuracy", "Range coverage", "Range width (pp)",
    ]
    table_rows = []
    for row in rows:
        cells = ([_escape(row.get("coin", ""))] if include_coin else []) + [
            _horizon(row.get("horizon_days")),
            _model(row.get("model", "")),
            _number(row.get("samples"), 0),
            _number(row.get("return_mae_pct")),
            _number(row.get("return_rmse_pct")),
            _number(row.get("log_return_mae"), 4),
            _number(row.get("direction_accuracy"), percent=True) + ("%" if row.get("direction_accuracy") is not None else ""),
            _number(row.get("interval_coverage"), percent=True) + ("%" if row.get("interval_coverage") is not None else ""),
            _number(row.get("interval_width_pct")),
        ]
        row_class = ' class="model-row"' if row.get("model") in {"gru", "lstm", "kronos-small", "kronos-base"} else ""
        table_rows.append(f"<tr{row_class}>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join(f'<th scope="col">{_escape(heading)}</th>' for heading in headings)
        + "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table></div>"
    )


def _forecast_table(forecasts: list[dict]) -> str:
    if not forecasts:
        return '<p class="empty">No forecasts were produced. See exclusions and the run configuration below.</p>'
    headings = ["Coin", "As of (UTC)", "Target (UTC)", "Horizon", "Last close (USDT)",
                "Forecast (USDT)", "Change", "Empirical 80% range (USDT)", "Data status"]
    rows = []
    for row in forecasts:
        cells = [
            _escape(row.get("coin", "")),
            _escape(row.get("as_of", "")),
            _escape(row.get("target_date", "")),
            _horizon(row.get("horizon_days")),
            _price(row.get("last_price_usdt")),
            _price(row.get("forecast_price_usdt")),
            _change(row.get("change_pct")),
            _price(row.get("lower_price_usdt")) + " – " + _price(row.get("upper_price_usdt")),
            '<span class="stale">Stale / historical</span>' if row.get("is_stale") else "Within freshness limit",
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return (
        '<div class="table-scroll"><table class="forecasts"><thead><tr>'
        + "".join(f'<th scope="col">{_escape(heading)}</th>' for heading in headings)
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _learning_curve(history: list[dict], *, rate=False, curve_id="curve") -> str:
    points = []
    for row in history:
        try:
            values = tuple(float(row[key]) for key in (("epoch", "learning_rate") if rate else ("epoch", "train_loss", "validation_loss")))
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
        if all(math.isfinite(value) for value in values):
            points.append(values)
    if not points:
        return '<p class="empty">No recorded learning rates are available for this run.</p>' if rate else '<p class="empty">No learning curve is available.</p>'
    points.sort()
    left, top, width, height = 62, 22, 698, 196
    x_min, x_max = points[0][0], points[-1][0]
    y_max = max(0.0, *(max(point[1:]) for point in points)) or 1.0
    y_min = min(0.0, *(min(point[1:]) for point in points))
    y_span = y_max - y_min or 1.0

    def coordinates(index: int) -> str:
        return " ".join(
            f"{left + (point[0] - x_min) / (x_max - x_min or 1) * width:.2f},"
            f"{top + (y_max - point[index]) / y_span * height:.2f}"
            for point in points
        )

    labels = []
    for fraction in (0, 0.5, 1):
        y = top + fraction * height
        value = y_max - fraction * y_span
        labels.append(f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" class="grid"/>')
        label = f"{value:.1e}" if rate else _number(value, 3)
        labels.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{label}</text>')
    title = "Mean learning rate by epoch" if rate else "Training and validation loss by epoch"
    caption = "Mean rate actually used for optimizer updates in each epoch." if rate else '<span class="train-key">● Training loss</span> <span class="validation-key">● Validation loss</span>'
    series = ((1, "#087e8b"),) if rate else ((1, "#087e8b"), (2, "#c76b19"))
    lines = "".join(f'<polyline points="{coordinates(index)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
                    for index, color in series)
    return (
        f'<figure><svg viewBox="0 0 800 265" role="img" aria-labelledby="{curve_id}-title {curve_id}-desc">'
        f'<title id="{curve_id}-title">{title}</title>'
        f'<desc id="{curve_id}-desc">{caption if rate else "The model is selected by validation loss. Final test data is not used to select an epoch."}</desc>'
        + "".join(labels)
        + lines
        + f'<text x="{left}" y="240">{_number(x_min, 0)}</text>'
        + f'<text x="{left + width}" y="240" text-anchor="end">{_number(x_max, 0)}</text>'
        + '<text x="411" y="258" text-anchor="middle">Epoch</text></svg>'
        f'<figcaption>{caption}</figcaption></figure>'
    )


_STYLE = """
:root { color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #172d3a; background: #f4f7f9; }
* { box-sizing: border-box; } body { margin: 0; } main { max-width: 1480px; margin: 0 auto; padding: 40px 32px 64px; }
header { margin-bottom: 28px; } h1 { font-size: clamp(28px, 3vw, 42px); line-height: 1.15; margin: 12px 0; letter-spacing: -.03em; }
h2 { font-size: 22px; margin: 0 0 12px; } h3 { font-size: 17px; margin: 24px 0 10px; } p { line-height: 1.65; margin: 10px 0; }
.eyebrow { color: #476171; font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.subtitle, .note { color: #476171; max-width: 1000px; } .badge { display: inline-block; border: 1px solid #b9d4d6; color: #065d67; background: #e7f3f4; padding: 6px 10px; border-radius: 5px; font-size: 13px; font-weight: 650; }
.demo { background: #fff4dd; border-color: #e7cb8c; color: #784d09; } .notice { border-left: 4px solid #c88a27; padding: 12px 18px; background: #fff8e9; margin-top: 18px; }
section { background: #fff; border: 1px solid #dce5eb; border-radius: 10px; padding: 24px; margin: 18px 0; }
.meta { display: flex; gap: 26px; flex-wrap: wrap; margin: 20px 0 0; } .meta div { min-width: 130px; } .meta dt { color: #627886; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; } .meta dd { margin: 6px 0 0; font-weight: 650; }
.table-scroll { overflow-x: auto; margin-top: 16px; } table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }
th { font-size: 11px; text-align: right; color: #496473; background: #f1f6f8; letter-spacing: .025em; padding: 13px 12px; border-bottom: 1px solid #cadbe3; }
td { text-align: right; padding: 13px 12px; border-bottom: 1px solid #e6edf1; } td:first-child, th:first-child { text-align: left; } tbody tr:last-child td { border-bottom: 0; }
.model-row { background: #f0f8f8; } .forecasts td:first-child { font-weight: 700; } .stale { color: #946015; font-weight: 600; } .empty { color: #627886; padding: 12px 0; }
details { margin: 20px 0 0; } summary { color: #065d67; font-weight: 650; cursor: pointer; line-height: 1.6; } pre { overflow-x: auto; background: #f3f7f9; padding: 16px; border-radius: 5px; font-size: 12px; line-height: 1.6; }
figure { margin: 20px 0 0; max-width: 1000px; } svg { width: 100%; height: auto; } svg text { fill: #627886; font-size: 12px; } .grid { stroke: #dce5eb; stroke-width: 1; } figcaption { color: #627886; font-size: 13px; } .train-key { color: #087e8b; margin-right: 20px; } .validation-key { color: #c76b19; }
a { color: #065d67; text-underline-offset: 3px; } footer { color: #627886; font-size: 13px; margin-top: 24px; } .artifact-links { display: flex; gap: 20px; flex-wrap: wrap; }
@media (max-width: 700px) { main { padding: 24px 14px 40px; } section { padding: 18px 14px; } h2 { font-size: 20px; } }
@media print { :root { background: white; } main { padding: 0; } section { break-inside: avoid; border-radius: 0; } .table-scroll { overflow: visible; } table { white-space: normal; font-size: 9px; } th, td { padding: 7px 4px; } }
"""


def write_report(run_dir: Path | str, summary: dict, forecasts: list[dict]) -> Path:
    """Write a portable report with no network, script, or third-party dependency.

    All experiment values are escaped before insertion. Metrics and forecasts are
    displayed as supplied; the report does not recompute scores or select a model.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    synthetic = summary.get("data_kind") == "synthetic_demo"
    kind = "Synthetic demonstration" if synthetic else "Historical-data experiment"
    badge_class = "badge demo" if synthetic else "badge"
    demo_note = (
        '<p class="notice"><strong>Synthetic data.</strong> This run verifies the software pipeline only. '
        'Its prices, forecasts, and metrics do not measure real cryptocurrency performance.</p>'
        if synthetic else ""
    )
    metrics = summary.get("metrics") or {}
    test_rows = metrics.get("test") or []
    test_macro = [row for row in test_rows if row.get("coin") == "ALL"]
    test_coins = [row for row in test_rows if row.get("coin") != "ALL"]
    non_overlapping = metrics.get("test_non_overlapping") or []
    non_overlapping_macro = [row for row in non_overlapping if row.get("coin") == "ALL"]
    excluded = summary.get("excluded_coins") or {}
    exclusion_html = ""
    if excluded:
        exclusion_html = (
            '<section><h2>Excluded coins</h2><p class="note">These coins were not eligible for this trained model.</p>'
            '<div class="table-scroll"><table><thead><tr><th scope="col">Coin</th><th scope="col">Reason</th></tr></thead><tbody>'
            + "".join(f"<tr><td>{_escape(coin)}</td><td>{_escape(reason)}</td></tr>" for coin, reason in excluded.items())
            + "</tbody></table></div></section>"
        )
    coins = ", ".join(str(coin) for coin in summary.get("coins", [])) or "None"
    interval_note = summary.get("interval_note") or (
        "Empirical 80% ranges use validation residuals of the selected model. "
        "They are not guaranteed probabilities and may be miscalibrated when market conditions change."
    )
    config = summary.get("config") or {}
    model_type = summary.get("model_type", config.get("model_type", "gru"))
    architecture = _model(model_type)
    # Historical summaries did not carry input_mode; render those as daily runs.
    multiframe = config.get("input_mode", "daily") == "multiframe"
    lookback = config.get("lookback", 60)
    evaluation_note = ""
    training_note = "Training learns model weights and normalization."
    artifact_note = "The run summary and model checkpoint preserve settings, normalization, coin mapping, data provenance, and split boundaries."
    tokenizer_curve = ""
    if model_type in {"kronos-small", "kronos-base"}:
        kronos = summary.get("kronos") or {}
        timeframe = kronos.get("timeframe", config.get("kronos_timeframe", "4h"))
        eval_stride = kronos.get("eval_stride", config.get("kronos_eval_stride", 1))
        train_stride = kronos.get("train_stride", config.get("kronos_train_stride", 1))
        tokenizer_finetuned = kronos.get("tokenizer_finetuned", False)
        tokenizer_label = "fine-tuned tokenizer" if tokenizer_finetuned else "frozen pretrained tokenizer"
        prediction_length = config.get("kronos_prediction_length")
        forecast_note = (f'Forecasts generate {_escape(prediction_length)} future candles and evaluate the price and direction at each closing boundary; '
                         if prediction_length is not None else
                         'Forecasts generate a 30-day candle path and read its 1-, 7-, and 30-day closes; ')
        model_label = f"{_model(model_type)} Transformer + {tokenizer_label}"
        input_metadata = (
            f'<div><dt>Input timeframe</dt><dd>{_escape(timeframe)} candles</dd></div>'
            f'<div><dt>Forecast cadence</dt><dd>{_escape(timeframe)} origins</dd></div>'
            f'<div><dt>Input lookback</dt><dd>{_escape(lookback)} candles</dd></div>'
        )
        reference_note = (
            f'The reference price is the close at the last completed UTC {_escape(timeframe)} forecast origin. '
            '“As of” is that closing boundary. Only completed candles are available to the model. '
            + ('The forecast length is measured in candles of the selected timeframe.'
               if prediction_length is not None else 'The month horizon is exactly 30 calendar days.')
        )
        amount_note = kronos.get("amount_note", "The tokenizer's quote-turnover channel is set to zero because the input CSV has no quote-turnover field.")
        input_note = (
            f'<p class="note">{_escape(_model(model_type))} reads one {_escape(timeframe)} OHLCV sequence per coin with UTC calendar features. '
            f'The {_escape(tokenizer_label)} converts normalized candles to discrete tokens. '
            'Fine-tuning updates all predictor weights with future-token cross-entropy loss. '
            'Normalization uses only each completed input window; future labels use the same input statistics. '
            f'{forecast_note}'
            'multiple generated paths are averaged. There is no learned coin embedding or recurrent encoder.</p>'
            f'<p class="note">{_escape(amount_note)}</p>'
            f'<p class="note">The pretrained context allows {_escape(kronos.get("max_context", 512))} bars. '
            f'Training selects every {_ordinal(train_stride)} eligible origin per coin after chronological splitting. '
            'Prediction runs on demand after refreshing local data; the forecast cadence does not start a background scheduler.</p>'
        )
        evaluation_note = (
            f'<p class="note">Kronos evaluation selects every {_ordinal(eval_stride)} eligible held-out origin per coin. '
            'The model and baselines use those same sampled origins. '
            'A local held-out split does not establish that these dates were absent from Kronos pretraining; '
            'upstream pretraining overlap with this dataset is unknown.</p>'
        )
        training_note = "Fine-tuning learns all Kronos predictor weights while the pretrained tokenizer stays frozen."
        artifact_note = "The run saves fine-tuned predictor weights, a frozen tokenizer, model settings, data provenance, and split boundaries."
        if tokenizer_finetuned:
            training_note = ("Fine-tuning first adapts the tokenizer with coarse/full reconstruction MSE and BSQ loss. "
                             "Its best validation checkpoint is frozen before predictor training with future-token cross entropy.")
            artifact_note = "The run saves the matched fine-tuned tokenizer and predictor, both stages' histories, model settings, data provenance, and split boundaries."
            tokenizer_training = summary.get("tokenizer_training") or {}
            if tokenizer_training.get("reused"):
                training_note = ("Previously tuned tokenizer reused with all weights frozen. "
                                 "This run fine-tunes only the predictor with future-token cross entropy.")
                input_note += (f'<p class="note">Tokenizer source: '
                               f'{_escape(tokenizer_training.get("source_run", "—"))}. '
                               'The tokenizer curve below records its original tuning stage.</p>')
            tokenizer_curve = (
                '<h3>Tokenizer learning curve</h3><p class="note">'
                'Training loss combines coarse/full reconstruction MSE and BSQ loss; validation uses full reconstruction MSE. '
                'These losses have different objectives and are separate from predictor cross entropy. '
                f'Tokenizer selected epoch: {_escape(tokenizer_training.get("best_epoch", "—"))}.</p>'
                + _learning_curve(tokenizer_training.get("history") or [], curve_id="tokenizer-loss")
                + '<h3>Tokenizer learning rate</h3>'
                + _learning_curve(tokenizer_training.get("history") or [], rate=True, curve_id="tokenizer-rate")
            )
    elif multiframe:
        origin_interval = str(config.get("origin_interval", "4h"))
        model_label = f"Four {architecture} encoders + coin embedding"
        input_metadata = (
            '<div><dt>Combined timeframes</dt><dd>15m · 1h · 4h · 1d</dd></div>'
            f'<div><dt>Forecast cadence</dt><dd>{_escape(origin_interval)} origins</dd></div>'
            f'<div><dt>Input lookback</dt><dd>{_escape(lookback)} candles per timeframe</dd></div>'
        )
        reference_note = (
            f'The reference price is the close at the last completed UTC {_escape(origin_interval)} forecast origin. '
            '“As of” is that closing boundary. Each branch uses only candles completed by that time, '
            'including the most recent completed daily candle. The month horizon is exactly 30 calendar days.'
        )
        input_note = (
            '<p class="note">The model combines 15-minute, 1-hour, 4-hour, and daily candle sequences. '
            'Each branch reads price returns, ranges, momentum, volatility, log traded volume, volume changes and relative volume. '
            'Higher-timeframe volume is the sum of its constituent 15-minute volumes; incomplete buckets are excluded. '
            f'Four {architecture} encoders merge their outputs with a learned coin embedding to predict all three horizons together.</p>'
            '<p class="note">Lookbacks count candles in each timeframe, with 30 additional candles for feature warmup. '
            'A 60-candle lookback covers 15 hours, 60 hours, 10 days, and 60 days, respectively. '
            'Prediction runs on demand after refreshing local data; the forecast cadence does not start a background scheduler.</p>'
        )
    else:
        model_label = f"Shared {architecture} + coin embedding"
        input_metadata = '<div><dt>Input timeframe</dt><dd>Daily</dd></div>'
        reference_note = (
            'The reference price is the last completed UTC daily candle close, not a live quote. '
            '“As of” is the boundary after that candle. The month horizon is exactly 30 calendar days.'
        )
        input_note = (
            '<p class="note">This daily-input model reads price returns, ranges, momentum, volatility, '
            'and volume changes and relative volume from completed daily candles. '
            f'A shared {architecture} encoder combines them with a learned coin embedding.</p>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{architecture} cryptocurrency forecast experiment</title><style>{_STYLE}</style></head><body><main>
<header><div class="eyebrow">Selected-coin forecasting research</div><h1>Cryptocurrency price forecasts · {architecture}</h1>
<span class="{badge_class}">{kind}</span>
<p class="subtitle">Estimated spot market prices in USDT for the next 1, 7, and 30 calendar days, with held-out historical evaluation.</p>
{demo_note}
<dl class="meta"><div><dt>Trained coins</dt><dd>{_escape(coins)}</dd></div><div><dt>Selected epoch</dt><dd>{_escape(summary.get('best_epoch', '—'))}</dd></div><div><dt>Model</dt><dd>{model_label}</dd></div>{input_metadata}</dl></header>
<section><h2>Forecasts from the last completed close</h2>
<p class="note">{reference_note}</p>
{_forecast_table(forecasts)}
<p class="note">{_escape(interval_note)}</p>
<p class="note">Forecasts estimate market price rather than intrinsic value. Historical accuracy does not establish future accuracy or profitability.</p></section>
<section><h2>Final test: model and baseline comparison</h2>
<p class="note">The selected epoch is frozen before final-test evaluation. Each model uses the same forecast origins. ALL rows are coin-balanced macro averages; lower error is better. “pp” means percentage points of cumulative return.</p>
{evaluation_note}
{_metric_table(test_macro)}
<p class="note">Unchanged price predicts zero return. Trailing drift projects the recent 7-day log-return rate. Direction accuracy compares return signs, excluding zero actual or predicted returns; it is undefined for the unchanged-price baseline and shown as —. Coverage is the fraction of observed outcomes inside the empirical range.</p>
<details><summary>Per-coin final-test metrics</summary>{_metric_table(test_coins, include_coin=True)}</details>
<details><summary>Non-overlapping final-test origins</summary>
<p class="note">Origins are spaced by each horizon to reduce repeated coverage of the same target days. These rows supplement the full test set; fewer samples can make their scores unstable.</p>
{_metric_table(non_overlapping_macro)}
{_metric_table([row for row in non_overlapping if row.get('coin') != 'ALL'], include_coin=True)}</details></section>
<section><h2>Training and validation</h2>
{input_note}
<p class="note">{training_note} Validation selects the epoch and calibrates residual ranges. All coins share chronological split boundaries; labels reaching the following split are purged.</p>
{_learning_curve(summary.get('history') or [])}
<h3>Model learning rate</h3>
{_learning_curve(summary.get('history') or [], rate=True, curve_id='model-rate')}
{tokenizer_curve}
<details><summary>Validation metrics used during development</summary>{_metric_table(metrics.get('validation') or [], include_coin=True)}</details>
<details><summary>Exact split boundaries and counts</summary><pre>{_escape(json.dumps(summary.get('splits', {}), indent=2, default=str))}</pre></details>
<details><summary>Run configuration</summary><pre>{_escape(json.dumps(summary.get('config', {}), indent=2, default=str))}</pre></details></section>
{exclusion_html}
<section><h2>Saved experiment artifacts</h2><p class="artifact-links"><a href="summary.json">Run summary</a><a href="forecasts.csv">Forecast CSV</a><a href="forecasts.json">Forecast JSON</a><a href="validation_predictions.csv">Validation predictions</a><a href="test_predictions.csv">Test predictions</a></p>
<p class="note">{artifact_note} A new evaluation after changes should use untouched future data.</p></section>
<footer>Standalone research report · No live order execution · Prices are quoted in USDT</footer>
</main></body></html>"""
    path = run_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path

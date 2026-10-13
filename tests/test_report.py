"""Reports must distinguish market data from synthetic output and escape values."""

from html.parser import HTMLParser

from crypto_forecast.report import write_report


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []

    def handle_starttag(self, tag, attrs):
        self.names.append(tag)


def test_report_labels_data_origin_historical_reference_and_stale_forecasts(tmp_path):
    forecast = {
        "coin": "BTC", "as_of": "2025-10-01T00:00:00+00:00",
        "target_date": "2025-10-31T00:00:00+00:00", "horizon_days": 30,
        "last_price_usdt": 100.0, "forecast_price_usdt": 110.0, "change_pct": 10.0,
        "lower_price_usdt": 80.0, "upper_price_usdt": 130.0, "is_stale": True,
    }
    for kind, expected in (("historical", "Historical-data experiment"), ("synthetic_demo", "Synthetic demonstration")):
        document = write_report(tmp_path / kind, {"data_kind": kind, "coins": ["BTC"]}, [forecast]).read_text(encoding="utf-8")
        assert expected in document
        assert ("<strong>Synthetic data.</strong>" in document) == (kind == "synthetic_demo")
        assert "Stale / historical" in document
        assert forecast["as_of"] in document and forecast["target_date"] in document
        assert "last completed UTC daily candle close, not a live quote" in document
        assert "exactly 30 calendar days" in document
        assert "guaranteed probabilities" in document
        assert "+10.00%" in document


def test_report_escapes_every_displayed_experiment_field(tmp_path):
    injection = '<script>alert("coin")</script><img src=x onerror="alert(1)">'
    metric = dict.fromkeys(("coin", "horizon_days", "model", "samples", "return_mae_pct", "return_rmse_pct", "log_return_mae", "direction_accuracy", "interval_coverage", "interval_width_pct"), injection)
    forecast = dict.fromkeys(("coin", "as_of", "target_date", "horizon_days", "last_price_usdt", "forecast_price_usdt", "change_pct", "lower_price_usdt", "upper_price_usdt"), injection)
    forecast["is_stale"] = False
    summary = {
        "data_kind": "historical", "coins": [injection], "best_epoch": injection,
        "config": {injection: injection}, "splits": {injection: injection},
        "excluded_coins": {injection: injection}, "interval_note": injection,
        "metrics": {"validation": [metric], "test": [dict(metric, coin="ALL"), metric], "test_non_overlapping": [metric]},
    }
    document = write_report(tmp_path, summary, [forecast]).read_text(encoding="utf-8")
    assert injection not in document
    assert "&lt;script&gt;" in document
    assert "&quot;" in document
    parser = _Tags()
    parser.feed(document)
    assert "script" not in parser.names
    assert "img" not in parser.names


def test_combined_report_describes_closed_timeframes_volume_and_on_demand_cadence(tmp_path):
    summary = {
        "coins": ["BTC"],
        "config": {"input_mode": "multiframe", "origin_interval": "4h", "lookback": 60},
    }
    document = write_report(tmp_path, summary, []).read_text(encoding="utf-8")
    assert "Four GRU encoders + coin embedding" in document
    assert "Combined timeframes" in document
    assert "15m · 1h · 4h · 1d" in document
    assert "last completed UTC 4h forecast origin" in document
    assert "only candles completed by that time" in document
    assert "60 candles per timeframe" in document
    assert "volume changes and relative volume" in document
    assert "sum of its constituent 15-minute volumes" in document
    assert "does not start a background scheduler" in document
    assert "last completed UTC daily candle close" not in document


def test_combined_report_escapes_dynamic_input_configuration(tmp_path):
    injection = '<img src=x onerror="alert(1)">'
    summary = {"config": {"input_mode": "multiframe", "origin_interval": injection, "lookback": injection}}
    document = write_report(tmp_path, summary, []).read_text(encoding="utf-8")
    assert injection not in document
    assert "&lt;img" in document

"""Chart input boundaries and real read-only inference with tiny Kronos weights."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_forecast.chart_forecast import forecast_chart, prepare_chart, settings
from crypto_forecast.dashboard import DashboardError, JobManager
from crypto_forecast.dashboard_worker import run_job
from crypto_forecast.kronos_backend import RAW_FEATURES, _constructor_configs
from test_kronos import candles, tiny_modules
from test_dashboard import dashboard, _post


def payload(**values):
    return dict(source="csv", coin="BTC", timeframe="1d", lookback=8, horizon=3,
                csv=candles(12).to_csv(index=False), **values)


def install_tiny(workspace, variant="kronos-small"):
    model, tokenizer = tiny_modules()
    root = workspace / "models" / variant
    for part, module in (("model", model), ("tokenizer", tokenizer)):
        module.save_pretrained(root / part)
    return root


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("changed", [
    {"lookback": 1}, {"lookback": 513}, {"lookback": True}, {"horizon": 0},
    {"horizon": 513}, {"horizon": "24"}, {"sample_count": 11}, {"sample_count": False},
    {"seed": -1}, {"temperature": float("nan")}, {"top_p": 0}, {"device": "../cpu"},
    {"source": "remote"}, {"timeframe": "5m"}, {"coin": "../secret"}, {"end": "NaT"},
    {"csv": "x" * 1_000_001}, {"model_path": "outside"},
])
def test_chart_rejects_invalid_input(changed):
    with pytest.raises((ValueError, TypeError)):
        settings({**payload(), **changed})


def test_context_cutoff_and_gap_check(tmp_path):
    value, chart = prepare_chart(tmp_path, payload(end="2024-01-11"))
    assert len(chart["history"]) == 8
    assert chart["history"][0]["timestamp"] == "2024-01-03T00:00:00+00:00"
    assert chart["origin"] == "2024-01-11T00:00:00+00:00"
    assert chart["forecast_end"] == "2024-01-14T00:00:00+00:00"
    assert chart["incomplete_dropped"] == 2 and chart["historical"]
    broken = candles(12).drop(index=9)
    with pytest.raises(ValueError, match="gaps"):
        prepare_chart(tmp_path, {**payload(), "csv": broken.to_csv(index=False)})
    with pytest.raises(ValueError, match="Need 20"):
        prepare_chart(tmp_path, {**payload(), "lookback": 20})


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1d"])
def test_saved_chart_preview_leaves_source_unchanged(tmp_path, timeframe):
    path = tmp_path / "data" / timeframe / "BTC.csv"
    path.parent.mkdir(parents=True)
    path.write_text(candles(12, timeframe).to_csv(index=False), encoding="utf-8")
    before = hashes(tmp_path / "data")
    value, chart = prepare_chart(tmp_path, dict(source="saved", coin="BTC", timeframe=timeframe, lookback=8, horizon=3))
    assert len(chart["history"]) == 8 and value["model"] == "original"
    assert hashes(tmp_path / "data") == before


def test_csv_accepts_upstream_timestamp_alias_and_missing_volume(tmp_path):
    csv = candles(12).drop(columns="volume").rename(columns={"timestamp": "timestamps"}).to_csv(index=False)
    _, chart = prepare_chart(tmp_path, {**payload(), "csv": csv})
    assert all(candle["volume"] == 0 for candle in chart["history"])


def test_real_kronos_forecast_is_repeatable_and_preserves_original_files(tmp_path):
    root = install_tiny(tmp_path)
    original = hashes(root)
    value, chart = prepare_chart(tmp_path, {**payload(), "device": "cpu"})
    result = forecast_chart(tmp_path, value, chart)
    repeat = forecast_chart(tmp_path, value, chart)
    assert result["model"] == "original"
    assert result["forecast"] == repeat["forecast"]
    assert len(result["forecast"]) == 3
    assert result["forecast"][0]["timestamp"] == chart["origin"]
    for candle in result["forecast"]:
        assert candle["high"] >= max(candle["open"], candle["close"], candle["low"])
        assert candle["low"] <= min(candle["open"], candle["close"])
        assert candle["volume"] >= 0
    assert np.isfinite([candle["close"] for candle in result["forecast"]]).all()
    assert hashes(root) == original
    assert not (tmp_path / "runs").exists()


def test_forecast_job_worker_persists_results_without_training(tmp_path, monkeypatch):
    root = install_tiny(tmp_path)
    original = hashes(root)
    manager = JobManager(tmp_path)
    launches = []
    monkeypatch.setattr(manager, "_launch", lambda job: launches.append(job))
    try:
        job = manager.start_chart_forecast({**payload(), "device": "cpu"})
        assert job["operation"] == "forecast" and "csv" not in job["config"]
        assert launches and job["run_id"] is None
        with pytest.raises(DashboardError, match="not available"):
            manager.chart_result(job["id"])
        with pytest.raises(DashboardError, match="already running"):
            manager.start_job({"operation": "train", "config": {}})
        assert run_job(manager._path(job["id"])) == 0
        assert manager.get_job(job["id"])["status"] == "completed"
        result = manager.chart_result(job["id"])
        assert len(result["forecast"]) == 3
        assert hashes(root) == original
        assert not (tmp_path / "runs").exists()
        assert len(manager.list_jobs()) == 1
    finally:
        manager.close()


def test_saved_finetuned_chart_inference_is_separate_and_checks_interval(tmp_path, monkeypatch):
    original = install_tiny(tmp_path)
    model, tokenizer = tiny_modules()
    model_config, tokenizer_config = _constructor_configs(model, tokenizer)
    run = tmp_path / "runs" / "my-finetune"
    run.mkdir(parents=True)
    torch.save({"metadata": {"format_version": 3, "model_type": "kronos-small",
                            "features": list(RAW_FEATURES), "timeframes": ["1d"],
                            "model_config": model_config, "tokenizer_config": tokenizer_config},
                "state_dict": model.state_dict(), "tokenizer_state_dict": tokenizer.state_dict()}, run / "model.pt")
    (run / "summary.json").write_text(json.dumps({"model_type": "kronos-small", "config": {"kronos_timeframe": "1d"}}))
    (run / "forecasts.json").write_text("[]")
    before = {**hashes(original), "fine-tuned": hashes(run)["model.pt"]}
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    try:
        assert manager.chart_models()["original_ready"]
        assert manager.chart_models()["runs"][0]["id"] == "my-finetune"
        with pytest.raises(DashboardError, match="trained on this candle interval"):
            manager.start_chart_forecast({**payload(), "model": "my-finetune", "timeframe": "4h",
                                          "csv": candles(12, "4h").to_csv(index=False)})
        with pytest.raises(DashboardError):
            manager.start_chart_forecast({**payload(), "model": "../outside"})
        job = manager.start_chart_forecast({**payload(), "model": "my-finetune", "device": "cpu"})
        assert run_job(manager._path(job["id"])) == 0
        result = manager.chart_result(job["id"])
        assert result["model"] == "my-finetune" and "fine-tuned run" in result["model_label"]
        assert before == {**hashes(original), "fine-tuned": hashes(run)["model.pt"]}
    finally:
        manager.close()


def test_chart_http_routes_require_csrf_and_separate_jobs(dashboard):
    server, request, bootstrap, launches, address, workspace = dashboard
    status, _, _ = request("/api/chart/preview", method="POST", value=payload())
    assert status == 403
    status, chart, _ = _post(dashboard, payload(), path="/api/chart/preview")
    assert status == 200 and len(chart["history"]) == 8 and not launches
    status, error, _ = _post(dashboard, payload(), path="/api/chart/forecast")
    assert status == 400 and "weights are missing" in error["error"]
    install_tiny(workspace)
    status, job, _ = _post(dashboard, payload(), path="/api/chart/forecast")
    assert status == 201 and job["operation"] == "forecast" and len(launches) == 1
    status, _, _ = request("/api/chart/forecasts/" + job["id"])
    assert status == 404
    assert run_job(server.manager._path(job["id"])) == 0
    status, result, _ = request("/api/chart/forecasts/" + job["id"])
    assert status == 200 and len(result["forecast"]) == 3
    status, source, _ = request("/chart.js")
    assert status == 200 and "KronosChart.create" in source
    status, chart_source, _ = request("/chart_view.js")
    assert status == 200 and "HistogramSeries" in chart_source
    status, library, _ = request("/vendor/lightweight-charts.js")
    assert status == 200 and "TradingView Lightweight Charts" in library
    status, page, _ = request("/")
    assert 'id="chart-tab"' in page and 'id="training-tab"' in page
    assert 'aria-selected="true" aria-controls="chart-tab"' in page
    assert 'id="chart-coin"' in page and 'id="chart-duration"' in page
    assert 'id="chart-duration-unit"' in page and 'max="512"' in page
    assert 'id="chart-file"' not in page and 'id="chart-source"' not in page
    assert 'id="candle-volume-title"' in page and 'Time (UTC)' in page
    assert 'id="candle-fit"' in page and 'id="candle-focus"' in page

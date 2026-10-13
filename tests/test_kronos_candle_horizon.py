"""Configurable candle targets must work through training and checkpoint reload."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_forecast import kronos_backend as backend
from crypto_forecast.chart_forecast import forecast_chart, prepare_chart
from crypto_forecast.cli import _settings, parser
from crypto_forecast.features import split_samples
from crypto_forecast.pipeline import TrainingConfig, predict, train
from crypto_forecast.report import write_report
from test_kronos import candles, config_for, tiny_modules, write_history
from test_dashboard import dashboard, _post


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1d"])
def test_all_candle_intervals_accept_a_configurable_training_target(timeframe):
    config = TrainingConfig(coins=["BTC"], model_type="kronos-base",
                            kronos_timeframe=timeframe, kronos_prediction_length=24,
                            lookback=120)
    config.validate()
    assert config.kronos_prediction_length == 24


def test_context_budget_depends_on_selected_target_length():
    config = TrainingConfig(coins=["BTC"], model_type="kronos-small",
                            kronos_timeframe="1h", kronos_prediction_length=24,
                            lookback=489)
    config.validate()
    with pytest.raises(ValueError, match="512-token"):
        replace(config, lookback=490).validate()


@pytest.mark.parametrize("length", [0, -1, True, 1.5, "24", 513])
def test_prediction_length_rejects_invalid_values(length):
    with pytest.raises(ValueError, match="kronos_prediction_length"):
        TrainingConfig(coins=["BTC"], model_type="kronos-base",
                       kronos_prediction_length=length).validate()


def test_hourly_samples_use_only_selected_context_and_future_candles():
    frame = candles(64, "1h")
    sample = backend.build_samples({"BTC": frame}, 8, "1h", prediction_length=3)[0]
    assert sample.origin == pd.Timestamp("2024-01-01T08:00:00Z")
    assert sample.x.shape == (8, 6)
    assert sample.future.shape == (3, 6)
    assert sample.label_end == pd.Timestamp("2024-01-01T11:00:00Z")
    assert list(sample.future_timestamps.hour) == [8, 9, 10]
    np.testing.assert_allclose(sample.y, np.log(frame.close.iloc[8:11].to_numpy() / frame.close.iloc[7]))
    latest = backend.inference_sample("BTC", frame.tail(8).reset_index(drop=True), 8, "1h", prediction_length=3)
    assert len(latest.x) == 8 and len(latest.future_timestamps) == 3
    assert backend.inference_sample("BTC", frame.tail(8).reset_index(drop=True), 8, "1h",
                                    prediction_length=5).y.shape == (5,)


def test_hourly_split_purges_the_actual_target_window():
    samples = backend.build_samples({"BTC": candles(240, "1h")}, 8, "1h", prediction_length=3)
    splits, info = split_samples(samples, 0.7, 0.15)
    assert all(s.label_end < pd.Timestamp(info["val_start"]) for s in splits["train"])
    assert all(s.label_end < pd.Timestamp(info["test_start"]) for s in splits["validation"])
    assert len(splits["validation"]) > 20


def test_cli_accepts_hourly_training_and_candle_target_length():
    config = _settings(parser().parse_args(["train", "--model-type", "kronos-base",
                                            "--kronos-timeframe", "1h",
                                            "--kronos-prediction-length", "24"]))
    config.validate()
    assert config.kronos_timeframe == "1h" and config.kronos_prediction_length == 24


def test_dashboard_saves_hourly_training_length_in_worker_job(dashboard):
    status, job, _ = _post(dashboard, {"operation": "train", "config": {
        "coins": ["BTC"], "model_type": "kronos-base", "kronos_timeframe": "1h",
        "kronos_prediction_length": 168, "lookback": 120,
    }})
    assert status == 201
    assert job["config"]["kronos_timeframe"] == "1h"
    assert job["config"]["kronos_prediction_length"] == 168
    assert job["config"]["lookback"] == 120


def test_dashboard_rejects_oversized_combined_training_sequence(dashboard):
    status, value, _ = _post(dashboard, {"operation": "train", "config": {
        "coins": ["BTC"], "model_type": "kronos-base", "kronos_timeframe": "1h",
        "kronos_prediction_length": 400, "lookback": 120,
    }})
    assert status == 400 and "512-token" in value["error"]


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1d"])
def test_candle_training_reload_and_report_preserve_each_future_close(tmp_path, monkeypatch, timeframe):
    model, tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_pretrained", lambda *_: (model, tokenizer))
    frame = write_history(tmp_path / "data", timeframe=timeframe)
    config = replace(config_for(tmp_path), kronos_timeframe=timeframe,
                     kronos_prediction_length=3, kronos_train_stride=64,
                     run_dir=str(tmp_path / "runs/candles"),
                     model_type="kronos-base" if timeframe == "1h" else "kronos-small")
    summary = train(config, progress=lambda _: None)
    run = tmp_path / "runs/candles"
    checkpoint = torch.load(run / "model.pt", weights_only=True)
    assert checkpoint["metadata"]["format_version"] == 4
    assert summary["kronos"]["prediction_bars"] == [1, 2, 3]
    before = json.loads((run / "forecasts.json").read_text())
    after = predict(run, tmp_path / "data", allow_stale=True, update_report=False)
    assert before == after
    assert [r["horizon_candles"] for r in after] == [1, 2, 3]
    step = backend.interval_delta(timeframe)
    for row in after:
        assert row["timeframe"] == timeframe
        assert pd.Timestamp(row["target_date"]) == pd.Timestamp(row["as_of"]) + step * row["horizon_candles"]
    assert len(summary["metrics"]["test"]) == 18
    document = (run / "report.html").read_text()
    assert "3 future candles" in document
    assert "generate a 30-day candle path" not in document
    value, chart = prepare_chart(tmp_path, dict(source="csv", csv=frame.tail(12).to_csv(index=False),
                                               coin="BTC", timeframe=timeframe, lookback=8,
                                               horizon=3, model="candles", device="cpu"))
    assert len(forecast_chart(tmp_path, value, chart)["forecast"]) == 3


def test_old_checkpoint_without_prediction_length_keeps_legacy_horizons(tmp_path, monkeypatch):
    model, tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_pretrained", lambda *_: (model, tokenizer))
    write_history(tmp_path / "data")
    config = replace(config_for(tmp_path), kronos_prediction_length=None)
    train(config, progress=lambda _: None)
    checkpoint = torch.load(tmp_path / "run/model.pt", weights_only=True)
    checkpoint["metadata"]["config"].pop("kronos_prediction_length")
    torch.save(checkpoint, tmp_path / "run/model.pt")
    rows = predict(tmp_path / "run", tmp_path / "data", allow_stale=True, update_report=False)
    assert [row["horizon_days"] for row in rows] == [1, 7, 30]

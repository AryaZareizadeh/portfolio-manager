"""Numerical policies and real training exercise rates, floor and continuation."""

from dataclasses import replace
import json

import pytest
import torch

from crypto_forecast.learning_rate import LearningRateSchedule
from crypto_forecast.pipeline import TrainingConfig, train, predict
from crypto_forecast.dashboard import JobManager, DashboardError
from crypto_forecast.cli import parser, _settings
from test_kronos import tiny_modules, write_history, config_for
from test_pipeline import _completed_candles


def schedule(policy, **settings):
    config = TrainingConfig(coins=["BTC"], lr_scheduler=policy, **settings)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    return LearningRateSchedule(optimizer, config, 5, 20)


@pytest.mark.parametrize("policy", ["warmup_cosine", "linear", "onecycle"])
def test_smooth_rates_reach_floor_and_hold_during_extension(policy):
    scheduler = schedule(policy, lr_warmup_fraction=0.1)
    rates = []
    for _ in range(130):
        rates.append(scheduler.rate)
        scheduler.batch_finished()
    assert max(rates) == pytest.approx(0.01)
    assert min(rates) == pytest.approx(0.001)
    assert rates[99:] == pytest.approx([0.001] * 31)
    if policy == "linear":
        assert rates[0] > rates[1] > rates[2]
    else:
        assert rates[0] < rates[1] < rates[10]
        assert rates[10] > rates[40] > rates[98]


def test_step_and_plateau_use_epoch_and_validation_boundaries():
    step = schedule("step", lr_step_epochs=2, lr_factor=0.5)
    step.batch_finished()
    step.epoch_finished(1)
    assert step.rate == 0.01
    step.epoch_finished(0.8)
    assert step.rate == 0.005
    plateau = schedule("plateau", lr_patience=1, lr_factor=0.5)
    for loss in (1.0, 0.8, 0.9):
        plateau.epoch_finished(loss)
    assert plateau.rate == 0.01
    plateau.epoch_finished(0.9)
    assert plateau.rate == 0.005
    for _ in range(20):
        plateau.epoch_finished(0.9)
    assert plateau.rate == pytest.approx(0.001)


def test_constant_and_one_update_runs_are_safe():
    fixed = schedule("constant")
    for _ in range(110):
        fixed.batch_finished()
        fixed.epoch_finished(1)
        assert fixed.rate == 0.01
    for policy in ("warmup_cosine", "linear", "onecycle"):
        config = TrainingConfig(coins=["BTC"], lr_scheduler=policy)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        tiny = LearningRateSchedule(optimizer, config, 1, 1)
        assert tiny.rate == 0.01
        tiny.batch_finished()
        assert tiny.last_used == 0.01 and tiny.rate == pytest.approx(0.001)


@pytest.mark.parametrize("prefix", ["", "kronos_tokenizer_"])
@pytest.mark.parametrize("field,value", [("lr_scheduler", "unknown"), ("lr_min_ratio", float("nan")),
    ("lr_min_ratio", 1), ("lr_warmup_fraction", -0.1), ("lr_factor", True),
    ("lr_factor", 0), ("lr_step_epochs", 1.5), ("lr_patience", -1)])
def test_invalid_policy_settings_fail_before_training(prefix, field, value):
    config = TrainingConfig(coins=["BTC"], model_type="kronos-base", **{prefix + field: value})
    with pytest.raises(ValueError, match=prefix + field):
        config.validate()


@pytest.mark.parametrize("model_type", ["kronos-small", "kronos-base"])
def test_kronos_predictor_and_tokenizer_record_independent_schedules(tmp_path, monkeypatch, model_type):
    model, tokenizer = tiny_modules()
    monkeypatch.setattr("crypto_forecast.kronos_backend.load_pretrained", lambda *_: (model, tokenizer))
    write_history(tmp_path / "data")
    config = replace(config_for(tmp_path), model_type=model_type, epochs=2, patience=3,
        lr_scheduler="linear", kronos_tokenizer_epochs=2, kronos_tokenizer_patience=3,
        kronos_tokenizer_lr_scheduler="constant")
    events = []
    summary = train(config, progress=lambda _: None, on_event=events.append)
    assert summary["history"][0]["learning_rate"] > summary["history"][1]["learning_rate"]
    assert summary["history"][-1]["lr_end"] == pytest.approx(config.learning_rate * config.lr_min_ratio)
    assert all(row["learning_rate"] == pytest.approx(config.kronos_tokenizer_learning_rate)
               for row in summary["tokenizer_training"]["history"])
    assert all("learning_rate" in event for event in events if event["kind"] in ("epoch", "tokenizer_epoch", "batch", "tokenizer_batch"))
    checkpoint = torch.load(tmp_path / "run/model.pt", weights_only=True)
    assert checkpoint["metadata"]["config"]["lr_scheduler"] == "linear"
    assert predict(tmp_path / "run", tmp_path / "data", allow_stale=True, update_report=False)
    report = (tmp_path / "run/report.html").read_text(encoding="utf-8")
    assert "Mean learning rate by epoch" in report and "Tokenizer learning rate" in report


@pytest.mark.parametrize("model_type", ["gru", "lstm"])
def test_recurrent_training_records_and_saves_scheduler_rates(tmp_path, model_type):
    directory = tmp_path / "data"
    directory.mkdir()
    _completed_candles("BTC").to_csv(directory / "BTC.csv", index=False)
    config = TrainingConfig(coins=["BTC"], input_mode="daily", model_type=model_type,
        start="2020-01-01", end="2025-10-01", data_dir=str(directory), run_dir=str(tmp_path / "run"),
        lookback=10, hidden_size=8, epochs=2, patience=3, batch_size=64, device="cpu", lr_scheduler="linear")
    summary = train(config, progress=lambda _: None)
    assert summary["history"][0]["learning_rate"] > summary["history"][1]["learning_rate"]
    saved = json.loads((tmp_path / "run/summary.json").read_text())
    assert saved["history"] == summary["history"]
    assert saved["history"][-1]["lr_end"] == pytest.approx(config.learning_rate * 0.1)


def test_dashboard_and_cli_accept_scheduler_settings(tmp_path, monkeypatch):
    config = _settings(parser().parse_args(["train", "--lr-scheduler", "plateau", "--lr-patience", "1",
        "--kronos-tokenizer-lr-scheduler", "linear"]))
    assert config.lr_scheduler == "plateau" and config.lr_patience == 1
    assert config.kronos_tokenizer_lr_scheduler == "linear"
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    try:
        with pytest.raises(DashboardError, match="lr_min_ratio"):
            manager.start_job({"operation": "train", "config": {"lr_min_ratio": float("nan")}})
        job = manager.start_job({"operation": "train", "config": {"lr_scheduler": "step", "lr_step_epochs": 2}})
        assert job["config"]["lr_scheduler"] == "step" and job["config"]["lr_step_epochs"] == 2
    finally:
        manager.close()


def test_worker_keeps_epoch_mean_separate_from_live_rate(tmp_path, monkeypatch):
    from crypto_forecast import dashboard_worker
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)

    def fake_train(config, *, on_event, **kwargs):
        on_event(dict(kind="tokenizer_epoch", stage="training_tokenizer", epoch=1,
            train_loss=1.0, validation_loss=0.8, learning_rate=0.001, lr_end=0.0008, lr_next=0.0007))
        on_event(dict(kind="epoch", stage="training", epoch=1,
            train_loss=1.0, validation_loss=0.8, learning_rate=0.0001, lr_end=0.00008, lr_next=0.00007))
        return {"best_epoch": 1}

    monkeypatch.setattr(dashboard_worker, "train", fake_train)
    try:
        job = manager.start_job({"operation": "train", "config": {"model_type": "kronos-base"}})
        assert dashboard_worker.run_job(manager._path(job["id"])) == 0
        saved = manager.get_job(job["id"])
        assert saved["history"][0]["learning_rate"] == 0.0001
        assert saved["tokenizer_history"][0]["learning_rate"] == 0.001
        assert saved["progress"]["learning_rate"] == 0.00008
    finally:
        manager.close()

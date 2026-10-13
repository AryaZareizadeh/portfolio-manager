"""Kronos dashboard settings and reports require no pretrained downloads."""

import json
from pathlib import Path

import pytest

from crypto_forecast.dashboard import DashboardError, JobManager
from crypto_forecast.report import write_report


@pytest.mark.parametrize("timeframe,lookback", [("4h", 333), ("1d", 483)])
def test_kronos_job_preserves_settings_and_selects_workspace_weights(tmp_path, monkeypatch, timeframe, lookback):
    root = tmp_path / "models" / "kronos-small"
    for part in ("model", "tokenizer"):
        (root / part).mkdir(parents=True)
    manager = JobManager(tmp_path)
    launched = []
    monkeypatch.setattr(manager, "_launch", lambda job: launched.append(job))
    try:
        job = manager.start_job({"operation": "train", "config": {
            "coins": ["BTC"], "model_type": "kronos-small", "kronos_timeframe": timeframe,
            "lookback": lookback, "kronos_sample_count": 2, "kronos_train_stride": 3,
            "kronos_preparation_workers": 2,
            "kronos_eval_stride": 6, "batch_size": 8, "learning_rate": 1e-5,
            "kronos_finetune_tokenizer": True, "kronos_tokenizer_epochs": 3,
            "kronos_tokenizer_patience": 1, "kronos_tokenizer_learning_rate": 0.0003,
        }})
        settings = job["config"]
        assert settings["model_type"] == "kronos-small"
        assert settings["kronos_timeframe"] == timeframe
        assert settings["kronos_sample_count"] == 2
        assert settings["kronos_train_stride"] == 3
        assert settings["kronos_preparation_workers"] == 2
        assert settings["kronos_eval_stride"] == 6
        assert settings["kronos_finetune_tokenizer"] is True
        assert settings["kronos_tokenizer_epochs"] == 3
        assert settings["kronos_tokenizer_patience"] == 1
        assert settings["kronos_tokenizer_learning_rate"] == 0.0003
        assert Path(settings["kronos_model_path"]) == root / "model"
        assert Path(settings["kronos_tokenizer_path"]) == root / "tokenizer"
        assert len(launched) == 1
    finally:
        manager.close()


@pytest.mark.parametrize("settings", [
    {"kronos_timeframe": "5m"}, {"lookback": 490},
    {"kronos_timeframe": "1d", "lookback": 490},
    {"kronos_sample_count": 0}, {"kronos_sample_count": True},
    {"kronos_train_stride": 0}, {"kronos_eval_stride": "6"}, {"kronos_preparation_workers": -1},
    {"kronos_finetune_tokenizer": "true"}, {"kronos_tokenizer_epochs": False},
    {"kronos_tokenizer_patience": 0}, {"kronos_tokenizer_learning_rate": "0.0002"},
    {"kronos_model_path": "../outside"}, {"kronos_tokenizer_path": "../outside"},
])
def test_dashboard_rejects_invalid_kronos_settings_and_arbitrary_weight_paths(tmp_path, monkeypatch, settings):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: pytest.fail("Invalid settings must not launch a worker."))
    try:
        with pytest.raises(DashboardError) as error:
            manager.start_job({"operation": "train", "config": {
                "coins": ["BTC"], "model_type": "kronos-small", **settings,
            }})
        assert error.value.status == 400
    finally:
        manager.close()


def test_dashboard_discovers_custom_single_timeframe_data_and_hub_fallback(tmp_path):
    candle = tmp_path / "data" / "4h" / "BTC.csv"
    candle.parent.mkdir(parents=True)
    candle.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    candle.with_suffix(".metadata.json").write_text(json.dumps({
        "rows": 30, "last_date": "2025-01-01T00:00:00+00:00",
    }), encoding="utf-8")
    manager = JobManager(tmp_path)
    try:
        status = manager.data_status()["coins"]
        assert len(status) == 1
        assert status[0]["coin"] == "BTC"
        assert status[0]["timeframes"] == ["4h"]
        assert status[0]["last_close"]["4h"] == "2025-01-01T04:00:00+00:00"
        assert manager.defaults()["kronos_model_path"] == "NeoQuasar/Kronos-small"
        assert manager.defaults()["kronos_tokenizer_path"] == "NeoQuasar/Kronos-Tokenizer-base"
    finally:
        manager.close()


def test_kronos_report_uses_recorded_single_stream_metadata_and_discloses_sampling(tmp_path):
    summary = {
        "model_type": "kronos-small", "coins": ["BTC"],
        "config": {"input_mode": "multiframe", "kronos_timeframe": "4h", "lookback": 60},
        "kronos": {"timeframe": "1d", "eval_stride": 7, "train_stride": 3, "max_context": 512},
        "metrics": {"test": [{"coin": "ALL", "model": "kronos-small", "horizon_days": 1}]},
    }
    document = write_report(tmp_path, summary, []).read_text(encoding="utf-8")
    assert "Kronos-small Transformer + frozen pretrained tokenizer" in document
    assert "one 1d OHLCV sequence" in document
    assert "every 7th eligible held-out origin" in document
    assert "Training selects every 3rd eligible origin" in document
    assert "upstream pretraining overlap with this dataset is unknown" in document
    assert "quote-turnover channel is set to zero" in document
    assert "Four Kronos-small encoders" not in document
    assert "Combined timeframes" not in document
    assert '<tr class="model-row">' in document


def test_kronos_report_escapes_backend_metadata(tmp_path):
    injection = '<img src=x onerror="alert(1)">'
    summary = {"model_type": "kronos-small", "kronos": dict.fromkeys(
        ("timeframe", "eval_stride", "train_stride", "max_context", "amount_note"), injection)}
    document = write_report(tmp_path, summary, []).read_text(encoding="utf-8")
    assert injection not in document
    assert "&lt;img" in document


def test_sequential_report_identifies_trained_tokenizer_and_separate_losses(tmp_path):
    summary = {"model_type": "kronos-small", "kronos": {"tokenizer_finetuned": True},
               "tokenizer_training": {"enabled": True, "best_epoch": 2, "history": [
                   {"epoch": 1, "train_loss": 1.2, "validation_loss": 0.4},
                   {"epoch": 2, "train_loss": 1.0, "validation_loss": 0.3}]}}
    document = write_report(tmp_path, summary, []).read_text(encoding="utf-8")
    assert "Kronos-small Transformer + fine-tuned tokenizer" in document
    assert "Tokenizer learning curve" in document
    assert "full reconstruction MSE" in document
    assert "Tokenizer selected epoch: 2" in document
    assert "frozen pretrained tokenizer" not in document


def test_worker_preserves_tokenizer_history_separately(tmp_path, monkeypatch):
    from crypto_forecast import dashboard_worker

    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)

    def train(config, *, on_event, **kwargs):
        on_event({"kind": "preparation", "stage": "preparing", "preparation_phase": "windows",
                  "preparation_completed": 50, "preparation_total": 100, "coin": "BTC",
                  "usable_samples": 45})
        saved = json.loads(next(manager.jobs_dir.glob("train-*.json")).read_text(encoding="utf-8"))
        assert saved["stage"] == "preparing"
        assert saved["progress"]["preparation_completed"] == 50
        assert saved["progress"]["preparation_total"] == 100
        assert saved["progress"]["coin"] == "BTC"
        on_event({"kind": "tokenizer_epoch", "stage": "training_tokenizer", "epoch": 1,
                  "train_loss": 0.8, "validation_loss": 0.2})
        on_event({"kind": "epoch", "stage": "training", "epoch": 1,
                  "train_loss": 3.5, "validation_loss": 3.4})
        return {"best_epoch": 1}

    monkeypatch.setattr(dashboard_worker, "train", train)
    try:
        job = manager.start_job({"operation": "train", "config": {"model_type": "kronos-small"}})
        assert dashboard_worker.run_job(manager.jobs_dir / f"{job['id']}.json") == 0
        finished = manager.get_job(job["id"])
        assert finished["tokenizer_history"] == [{"epoch": 1, "train_loss": 0.8, "validation_loss": 0.2}]
        assert finished["history"] == [{"epoch": 1, "train_loss": 3.5, "validation_loss": 3.4}]
    finally:
        manager.close()

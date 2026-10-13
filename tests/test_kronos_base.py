"""Base selection must choose separate pretrained files and preserve checkpoint identity."""

from dataclasses import replace
import json
import torch

from crypto_forecast.chart_forecast import prepare_chart, forecast_chart
from crypto_forecast.dashboard import JobManager
from crypto_forecast.dashboard_worker import run_job
from crypto_forecast.pipeline import train, predict
from test_chart_forecast import install_tiny, hashes, payload
from test_kronos import tiny_modules, write_history, config_for


def test_original_base_worker_uses_separate_files_and_preserves_both_variants(tmp_path, monkeypatch):
    torch.manual_seed(7)
    small = install_tiny(tmp_path)
    torch.manual_seed(19)
    base = install_tiny(tmp_path, "kronos-base")
    before = hashes(tmp_path / "models")
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    try:
        models = manager.chart_models()["pretrained"]
        assert [(model["id"], model["ready"]) for model in models] == [("original", True), ("original-base", True)]
        job = manager.start_chart_forecast({**payload(), "model": "original-base", "device": "cpu"})
        assert run_job(manager._path(job["id"])) == 0
        result = manager.chart_result(job["id"])
        assert result["model"] == "original-base" and result["model_label"].startswith("Kronos-base")
        value, chart = prepare_chart(tmp_path, {**payload(), "device": "cpu"})
        assert forecast_chart(tmp_path, value, chart)["forecast"] != result["forecast"]
        assert hashes(tmp_path / "models") == before
    finally:
        manager.close()


def test_base_finetune_checkpoint_roundtrip_and_chart_selection(tmp_path, monkeypatch):
    model, tokenizer = tiny_modules()
    monkeypatch.setattr("crypto_forecast.kronos_backend.load_pretrained", lambda *_: (model, tokenizer))
    write_history(tmp_path / "data")
    config = replace(config_for(tmp_path), model_type="kronos-base", kronos_finetune_tokenizer=False)
    summary = train(config, progress=lambda _: None)
    assert config.kronos_model_path == "NeoQuasar/Kronos-base"
    assert summary["model_type"] == "kronos-base"
    checkpoint = torch.load(tmp_path / "run/model.pt", weights_only=True)
    assert checkpoint["metadata"]["model_type"] == checkpoint["metadata"]["config"]["model_type"] == "kronos-base"
    rows = predict(tmp_path / "run", tmp_path / "data", allow_stale=True, update_report=False)
    assert rows
    value, chart = prepare_chart(tmp_path, {**payload(), "model": "base-test", "device": "cpu"})
    run = tmp_path / "runs/base-test"
    run.mkdir(parents=True)
    torch.save(checkpoint, run / "model.pt")
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "forecasts.json").write_text("[]")
    result = forecast_chart(tmp_path, value, chart)
    assert result["model_label"].startswith("Kronos-base")
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    try:
        assert manager.chart_models()["runs"][0]["id"] == "base-test"
        job = manager.start_chart_forecast({**payload(), "model": "base-test", "device": "cpu"})
        assert run_job(manager._path(job["id"])) == 0
    finally:
        manager.close()


def test_base_training_uses_local_base_paths(tmp_path, monkeypatch):
    root = install_tiny(tmp_path, "kronos-base")
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    try:
        job = manager.start_job({"operation": "train", "config": {"model_type": "kronos-base", "lookback": 8}})
        assert job["config"]["kronos_model_path"] == str(root / "model")
        assert job["config"]["kronos_tokenizer_path"] == str(root / "tokenizer")
    finally:
        manager.close()

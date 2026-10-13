"""Separate tokenizer inventory, confined selection, worker completion, and CLI."""

import json
import os
from pathlib import Path
import threading
import urllib.request

import pytest
import torch

from crypto_forecast.cli import parser, _settings
from crypto_forecast.dashboard import DashboardError, JobManager, create_server, read_json, utc_now, write_json


@pytest.fixture
def manager(tmp_path, monkeypatch):
    value = JobManager(tmp_path)
    monkeypatch.setattr(value, "_launch", lambda job: None)
    yield value
    value.close()


def save_tokenizer(workspace, identifier="tokenizer-run", complete=True):
    directory = workspace / "runs" / identifier
    directory.mkdir(parents=True)
    torch.save({"training_complete": complete, "tokenizer_state": {}}, directory / "tokenizer.pt")
    write_json(directory / "tokenizer_summary.json", {
        "artifact_type": "kronos-tokenizer", "created_at": utc_now(), "best_epoch": 2,
        "config": {"kronos_timeframe": "4h", "model_type": "kronos-small"},
        "tokenizer_training": {"best_validation_loss": 0.3,
                               "history": [{"epoch": 2, "validation_loss": 0.3}]}})
    return directory


def test_tokenizer_inventory_includes_stopped_combined_checkpoint(manager):
    directory = save_tokenizer(manager.workspace)
    now = utc_now()
    write_json(manager._path(directory.name), {"id": directory.name, "name": "Reusable tokenizer",
        "status": "stopped", "started_at": now, "finished_at": now})
    items = manager.list_tokenizers()
    assert len(items) == 1
    assert items[0]["id"] == directory.name
    assert items[0]["name"] == "Reusable tokenizer"
    assert items[0]["complete"] is True
    assert items[0]["best_epoch"] == 2
    assert items[0]["best_validation_loss"] == 0.3
    assert manager.list_runs() == []


def test_incomplete_tokenizer_checkpoint_remains_labeled(manager):
    save_tokenizer(manager.workspace, complete=False)
    assert manager.list_tokenizers()[0]["complete"] is False


@pytest.mark.parametrize("summary_state", ["missing", "stale", "corrupt"])
def test_inventory_recovers_authoritative_checkpoint_after_partial_publication(manager, summary_state):
    directory = save_tokenizer(manager.workspace)
    summary_path = directory / "tokenizer_summary.json"
    metadata = read_json(summary_path)
    metadata.update(best_epoch=4, training_complete=True)
    metadata["tokenizer_training"]["best_validation_loss"] = 0.1
    metadata["config"]["kronos_timeframe"] = "1h"
    torch.save({"metadata": metadata, "tokenizer_state_dict": {}}, directory / "tokenizer.pt")
    if summary_state == "missing":
        summary_path.unlink()
    elif summary_state == "stale":
        checkpoint_time = (directory / "tokenizer.pt").stat().st_mtime_ns
        os.utime(summary_path, ns=(checkpoint_time - 1000000000, checkpoint_time - 1000000000))
    else:
        summary_path.write_text("broken", encoding="utf-8")
    items = manager.list_tokenizers()
    assert len(items) == 1
    assert items[0]["best_epoch"] == 4
    assert items[0]["best_validation_loss"] == 0.1
    assert items[0]["complete"] is True
    assert items[0]["config"]["kronos_timeframe"] == "1h"


def test_predictor_can_select_durable_checkpoint_without_summary(manager):
    directory = save_tokenizer(manager.workspace)
    (directory / "tokenizer_summary.json").unlink()
    job = manager.start_job({"operation": "train", "config": {"model_type": "kronos-small",
        "kronos_training_stage": "predictor", "kronos_tokenizer_run_id": directory.name}})
    assert job["config"]["kronos_tokenizer_run_dir"] == str(directory.resolve())


def test_predictor_job_resolves_confined_tokenizer_id(manager):
    directory = save_tokenizer(manager.workspace)
    job = manager.start_job({"operation": "train", "config": {
        "model_type": "kronos-small", "coins": ["BTC"],
        "kronos_training_stage": "predictor", "kronos_tokenizer_run_id": directory.name,
        "kronos_tokenizer_batch_size": 4}})
    assert job["config"]["kronos_tokenizer_run_dir"] == str(directory.resolve())
    assert "kronos_tokenizer_run_id" not in job["config"]
    assert job["config"]["kronos_tokenizer_batch_size"] == 4


@pytest.mark.parametrize("source", ["../outside", "/outside", "missing", "", None, 3])
def test_invalid_tokenizer_source_is_rejected_without_job(manager, source):
    with pytest.raises(DashboardError) as error:
        manager.start_job({"operation": "train", "config": {"model_type": "kronos-small",
            "kronos_training_stage": "predictor", "kronos_tokenizer_run_id": source}})
    assert error.value.status == 400
    assert manager.list_jobs() == []


def test_browser_cannot_supply_tokenizer_filesystem_path(manager):
    with pytest.raises(DashboardError, match="Unknown training setting"):
        manager.start_job({"operation": "train", "config": {
            "kronos_tokenizer_run_dir": "C:/outside"}})


def test_tokenizer_symlink_artifact_is_rejected_as_bad_setting(manager, monkeypatch):
    directory = save_tokenizer(manager.workspace)
    actual = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == directory / "tokenizer.pt" or actual(path))
    with pytest.raises(DashboardError) as error:
        manager.start_job({"operation": "train", "config": {"model_type": "kronos-small",
            "kronos_training_stage": "predictor", "kronos_tokenizer_run_id": directory.name}})
    assert error.value.status == 400
    assert manager.list_tokenizers() == []


def test_corrupt_tokenizer_metadata_is_not_listed(manager):
    directory = save_tokenizer(manager.workspace)
    (directory / "tokenizer_summary.json").write_text("broken", encoding="utf-8")
    assert manager.list_tokenizers() == []


def test_corrupt_legacy_checkpoint_is_not_listed(manager):
    directory = save_tokenizer(manager.workspace)
    (directory / "tokenizer.pt").write_bytes(b"broken")
    assert manager.list_tokenizers() == []


def test_tokenizer_job_progress_uses_tokenizer_epoch_limit(manager):
    job = manager.start_job({"operation": "train", "config": {
        "model_type": "kronos-small", "kronos_training_stage": "tokenizer",
        "epochs": 100, "kronos_tokenizer_epochs": 7}})
    assert job["progress"]["max_epochs"] == 7


def test_cli_stage_source_and_batch_override_json(tmp_path):
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"kronos_training_stage": "sequential"}), encoding="utf-8")
    args = parser().parse_args(["train", "--config", str(config), "--kronos-training-stage", "predictor",
        "--kronos-tokenizer-run-dir", "runs/tokenizer-run", "--kronos-tokenizer-batch-size", "3"])
    settings = _settings(args)
    assert settings.kronos_training_stage == "predictor"
    assert settings.kronos_tokenizer_run_dir == "runs/tokenizer-run"
    assert settings.kronos_tokenizer_batch_size == 3


def test_tokenizer_endpoint_and_bootstrap(tmp_path):
    save_tokenizer(tmp_path)
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    address = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for endpoint in ("/api/tokenizers", "/api/bootstrap"):
            with opener.open(address + endpoint, timeout=5) as response:
                result = json.loads(response.read())
            assert result["tokenizers"][0]["id"] == "tokenizer-run"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_tokenizer_worker_finishes_without_evaluation_and_preserves_epoch_limit(manager, monkeypatch):
    from crypto_forecast import dashboard_worker

    job = manager.start_job({"operation": "train", "config": {"model_type": "kronos-small",
        "kronos_training_stage": "tokenizer", "epochs": 100, "kronos_tokenizer_epochs": 7}})
    def train(config, **callbacks):
        return {"artifact_type": "kronos-tokenizer", "best_epoch": 2, "tokenizer_training": {"history": []}}
    monkeypatch.setattr(dashboard_worker, "train", train)
    assert dashboard_worker.run_job(manager._path(job["id"])) == 0
    result = read_json(manager._path(job["id"]))
    assert result["status"] == "completed"
    assert result["progress"]["max_epochs"] == 7
    assert result["progress"]["best_epoch"] == 2
    assert "tokenizer" in result["logs"][-1]["message"].lower()
    assert "evaluation completed" not in result["logs"][-1]["message"]

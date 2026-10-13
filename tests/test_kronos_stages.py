"""Independent tokenizer checkpoints use real modules, not fabricated weights."""
from dataclasses import replace
import json

import pytest
import torch

from crypto_forecast import kronos_backend as backend
from crypto_forecast.pipeline import train
from test_kronos import config_for, tiny_modules, write_history


def tokenizer_run(tmp_path, monkeypatch, on_event=None, **changes):
    write_history(tmp_path / "data")
    config = replace(config_for(tmp_path), kronos_training_stage="tokenizer",
                     kronos_tokenizer_batch_size=4, **changes)
    _, tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_tokenizer_pretrained", lambda *args: tokenizer)
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: pytest.fail("Tokenizer-only loaded predictor"))
    summary = train(config, progress=lambda _: None, on_event=on_event)
    return config, summary, tokenizer


def test_tokenizer_only_saves_best_without_predictor_or_final_test(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "forecast_samples", lambda *args: pytest.fail("Tokenizer evaluated final test"))
    events = []
    config, summary, tokenizer = tokenizer_run(tmp_path, monkeypatch, on_event=events.append)
    checkpoint = torch.load(tmp_path / "run" / "tokenizer.pt", weights_only=True)
    assert summary["artifact_type"] == "kronos-tokenizer"
    assert summary["training_complete"] is True
    assert summary["tokenizer_training"]["batch_size"] == 4
    assert checkpoint["metadata"]["training_complete"] is True
    assert backend._state_hash(tokenizer) == checkpoint["metadata"]["tokenizer_sha256"]
    assert not (tmp_path / "run" / "model.pt").exists()
    assert not (tmp_path / "run" / "forecasts.json").exists()
    assert any(event.get("counts") == summary["splits"]["sampled_counts"] for event in events)


def test_predictor_reuses_frozen_tokenizer_with_independent_batch(tmp_path, monkeypatch):
    config, _, tokenizer = tokenizer_run(tmp_path, monkeypatch)
    original = backend._state_hash(tokenizer)
    model, base_tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: (model, base_tokenizer))
    monkeypatch.setattr(backend, "load_predictor_pretrained", lambda *args: model)
    monkeypatch.setattr(backend, "_train_tokenizer", lambda *args, **kw: pytest.fail("Tokenizer retrained"))
    predictor = replace(config, run_dir=str(tmp_path / "predictor"), kronos_training_stage="predictor",
                        kronos_tokenizer_run_dir=config.run_dir, batch_size=16)
    summary = train(predictor, progress=lambda _: None)
    saved = torch.load(tmp_path / "predictor" / "model.pt", weights_only=True)
    assert all(torch.equal(saved["tokenizer_state_dict"][k], v) for k, v in tokenizer.state_dict().items())
    assert summary["kronos"]["tokenizer_finetuned"] is True
    assert summary["kronos"]["tokenizer_source_run"] == config.run_dir
    assert summary["tokenizer_training"]["reused"] is True
    assert summary["config"]["batch_size"] == 16
    assert original == backend._state_hash(tokenizer)


@pytest.mark.parametrize("change", [{"train_fraction": 0.6}, {"lookback": 9}, {"kronos_train_stride": 16}])
def test_reuse_rejects_different_partitions_or_windows(tmp_path, monkeypatch, change):
    config, _, _ = tokenizer_run(tmp_path, monkeypatch)
    model, tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: (model, tokenizer))
    predictor = replace(config, run_dir=str(tmp_path / "predictor"), kronos_training_stage="predictor",
                        kronos_tokenizer_run_dir=config.run_dir, **change)
    with pytest.raises(ValueError, match="same candle data and chronological partitions"):
        train(predictor, progress=lambda _: None)


def test_best_tokenizer_survives_interruption_and_can_be_reused(tmp_path, monkeypatch):
    write_history(tmp_path / "data")
    config = replace(config_for(tmp_path), kronos_tokenizer_epochs=3,
                     kronos_training_stage="tokenizer")
    _, tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_tokenizer_pretrained", lambda *args: tokenizer)

    def stop_after_epoch(event):
        if event["kind"] == "tokenizer_epoch":
            raise RuntimeError("simulate worker interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        train(config, progress=lambda _: None, on_event=stop_after_epoch)
    saved = torch.load(tmp_path / "run" / "tokenizer.pt", weights_only=True)
    assert saved["metadata"]["training_complete"] is False
    assert saved["metadata"]["best_epoch"] == 1
    summary = json.loads((tmp_path / "run" / "tokenizer_summary.json").read_text())
    assert summary["training_complete"] is False
    model, base = tiny_modules()
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: (model, base))
    monkeypatch.setattr(backend, "load_predictor_pretrained", lambda *args: model)
    predictor = replace(config, kronos_training_stage="predictor", run_dir=str(tmp_path / "predictor"),
                        kronos_tokenizer_run_dir=config.run_dir)
    train(predictor, progress=lambda _: None)
    paired = torch.load(tmp_path / "predictor" / "model.pt", weights_only=True)
    assert all(torch.equal(v, paired["tokenizer_state_dict"][k]) for k, v in saved["tokenizer_state_dict"].items())


def test_sequential_saves_tokenizer_before_predictor_failure(tmp_path, monkeypatch):
    write_history(tmp_path / "data")
    model, tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: (model, tokenizer))
    monkeypatch.setattr(backend, "_token_dataset", lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("predictor OOM")))
    with pytest.raises(RuntimeError, match="predictor OOM"):
        train(config_for(tmp_path), progress=lambda _: None)
    saved = torch.load(tmp_path / "run" / "tokenizer.pt", weights_only=True)
    assert saved["metadata"]["training_complete"] is True
    assert not (tmp_path / "run" / "model.pt").exists()


def test_existing_tokenizer_cannot_be_overwritten_by_accidental_run(tmp_path, monkeypatch):
    config, _, _ = tokenizer_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="already contains"):
        train(config, progress=lambda _: None)


def test_saved_checkpoint_restores_best_epoch_instead_of_last_epoch(tmp_path, monkeypatch):
    states = []

    def validation(tokenizer, *args):
        states.append(backend._state_hash(tokenizer))
        return [0.2, 0.5][len(states) - 1]

    monkeypatch.setattr(backend, "_tokenizer_validation_loss", validation)
    config, summary, tokenizer = tokenizer_run(tmp_path, monkeypatch, kronos_tokenizer_epochs=3)
    saved = torch.load(tmp_path / "run" / "tokenizer.pt", weights_only=True)
    assert len(states) == 2 and states[0] != states[1]
    assert saved["metadata"]["tokenizer_sha256"] == states[0] == backend._state_hash(tokenizer)
    assert saved["metadata"]["best_epoch"] == summary["best_epoch"] == 1
    assert saved["metadata"]["tokenizer_training"]["epochs_completed"] == 2


def test_changed_candles_cannot_reuse_tokenizer(tmp_path, monkeypatch):
    config, _, _ = tokenizer_run(tmp_path, monkeypatch)
    path = tmp_path / "data" / "1d" / "BTC.csv"
    import pandas as pd
    import hashlib
    frame = pd.read_csv(path)
    frame.loc[10, "volume"] *= 2
    frame.to_csv(path, index=False)
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["data_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: tiny_modules())
    with pytest.raises(ValueError, match="same candle data and chronological partitions"):
        train(replace(config, kronos_training_stage="predictor", run_dir=str(tmp_path / "predictor"),
                      kronos_tokenizer_run_dir=config.run_dir), progress=lambda _: None)


def test_predictor_failure_keeps_source_checkpoint_bytes_unchanged(tmp_path, monkeypatch):
    config, _, _ = tokenizer_run(tmp_path, monkeypatch)
    source = tmp_path / "run" / "tokenizer.pt"
    original = source.read_bytes()
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: tiny_modules())
    monkeypatch.setattr(backend, "load_predictor_pretrained", lambda *args: tiny_modules()[0])
    monkeypatch.setattr(backend, "_batch_loss", lambda *args: (_ for _ in ()).throw(RuntimeError("predictor OOM")))
    with pytest.raises(RuntimeError, match="predictor OOM"):
        train(replace(config, kronos_training_stage="predictor", batch_size=16,
                      run_dir=str(tmp_path / "predictor"), kronos_tokenizer_run_dir=config.run_dir),
              progress=lambda _: None)
    assert source.read_bytes() == original


def test_report_discloses_reused_tokenizer_provenance_and_escapes_source(tmp_path):
    from crypto_forecast.report import write_report
    source = '<img src=x onerror="alert(1)">'
    summary = {"model_type": "kronos-small", "kronos": {"tokenizer_finetuned": True},
               "tokenizer_training": {"enabled": True, "reused": True, "source_run": source,
                                      "best_epoch": 2, "history": []}}
    document = write_report(tmp_path, summary, []).read_text(encoding="utf-8")
    assert "Previously tuned tokenizer reused" in document
    assert "Tokenizer source:" in document
    assert source not in document and "&lt;img" in document


def test_reuse_data_contract_accepts_coin_order_changes(tmp_path):
    from crypto_forecast.kronos_artifacts import data_signature
    config = replace(config_for(tmp_path), coins=["BTC", "ETH"])
    manifest = {coin: {"timeframes": {"1d": {"used_rows_sha256": coin}}} for coin in config.coins}
    splits = {"val_start": "2024-06-01", "test_start": "2024-08-01"}
    assert data_signature(config, manifest, splits) == data_signature(
        replace(config, coins=["ETH", "BTC"]), manifest, splits)


def test_saved_tokenizer_reuse_does_not_require_original_tokenizer_files(tmp_path, monkeypatch):
    config, _, _ = tokenizer_run(tmp_path, monkeypatch)
    model, _ = tiny_modules()
    model.save_pretrained(tmp_path / "base-predictor")
    monkeypatch.undo()
    summary = train(replace(config, kronos_training_stage="predictor",
        kronos_model_path=str(tmp_path / "base-predictor"),
        kronos_tokenizer_path=str(tmp_path / "missing-original-tokenizer"),
        kronos_tokenizer_run_dir=config.run_dir, run_dir=str(tmp_path / "predictor")),
        progress=lambda _: None)
    assert summary["tokenizer_training"]["reused"] is True

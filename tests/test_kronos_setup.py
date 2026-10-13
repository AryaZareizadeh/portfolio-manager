"""Local setup pins weight revisions and Kronos configuration stays explicit."""

import json
from types import SimpleNamespace

import pytest

from crypto_forecast.cli import _demo, _settings, parser
from crypto_forecast.kronos_setup import MODEL_ID, TOKENIZER_ID, download_pretrained
from crypto_forecast.pipeline import TrainingConfig


@pytest.mark.parametrize("variant", ["kronos-small", "kronos-base"])
def test_setup_records_and_downloads_exact_official_revisions(tmp_path, monkeypatch, variant):
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    class FakeApi:
        def model_info(self, repository):
            return SimpleNamespace(sha=f"pinned-{repository.split('/')[-1]}")

    def download(**kwargs):
        calls.append(kwargs)
        kwargs["local_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["local_dir"] / "config.json").write_text("{}")
        (kwargs["local_dir"] / "model.safetensors").write_bytes(b"test snapshot")

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "snapshot_download", download)
    manifest = download_pretrained(tmp_path, progress=lambda _: None, model=variant)
    assert [entry["repo_id"] for entry in calls] == ["NeoQuasar/" + variant.capitalize(), TOKENIZER_ID]
    assert all(call["revision"].startswith("pinned-") for call in calls)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert manifest["model"]["revision"] == calls[0]["revision"]
    assert manifest["tokenizer"]["revision"] == calls[1]["revision"]


def test_setup_does_not_publish_manifest_for_incomplete_weights(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", lambda: SimpleNamespace(model_info=lambda _: SimpleNamespace(sha="revision")))
    monkeypatch.setattr(hub, "snapshot_download", lambda **_: None)
    with pytest.raises(ValueError, match="Incomplete pretrained snapshot"):
        download_pretrained(tmp_path, progress=lambda _: None)
    assert not (tmp_path / "manifest.json").exists()


def test_failed_refresh_preserves_previous_snapshot_and_manifest(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    (tmp_path / "manifest.json").write_text('{"revision": "old"}')
    old_model = tmp_path / "model"
    old_model.mkdir()
    (old_model / "model.safetensors").write_bytes(b"previous weights")
    monkeypatch.setattr(hub, "HfApi", lambda: SimpleNamespace(model_info=lambda _: SimpleNamespace(sha="new")))

    def failing_download(**kwargs):
        if kwargs["repo_id"] == TOKENIZER_ID:
            raise OSError("disconnected")
        kwargs["local_dir"].mkdir(parents=True)
        (kwargs["local_dir"] / "config.json").write_text("{}")
        (kwargs["local_dir"] / "model.safetensors").write_bytes(b"new uncommitted weights")

    monkeypatch.setattr(hub, "snapshot_download", failing_download)
    with pytest.raises(ValueError, match="disconnected"):
        download_pretrained(tmp_path, progress=lambda _: None)
    assert (old_model / "model.safetensors").read_bytes() == b"previous weights"
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"revision": "old"}


@pytest.mark.parametrize("timeframe,maximum", [("4h", 333), ("1d", 483)])
def test_context_limit_includes_teacher_forcing_future(timeframe, maximum):
    TrainingConfig(coins=["BTC"], model_type="kronos-small", kronos_timeframe=timeframe,
                   lookback=maximum, kronos_prediction_length=None).validate()
    with pytest.raises(ValueError, match="512-token"):
        TrainingConfig(coins=["BTC"], model_type="kronos-small", kronos_timeframe=timeframe,
                       lookback=maximum + 1, kronos_prediction_length=None).validate()


def test_cli_selects_kronos_finetune_settings_and_setup():
    settings = _settings(parser().parse_args(["train", "--config", "config.kronos.example.json",
                                             "--kronos-timeframe", "1d", "--kronos-train-stride", "3"]))
    assert settings.model_type == "kronos-small"
    assert settings.kronos_timeframe == "1d"
    assert settings.kronos_train_stride == 3
    assert settings.batch_size == 1
    assert settings.learning_rate == 1e-5
    settings.validate()
    assert parser().parse_args(["kronos-setup"]).model_dir is None
    assert parser().parse_args(["kronos-setup", "--model", "kronos-base"]).model == "kronos-base"


def test_daily_kronos_demo_uses_daily_stream_and_finetune_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("crypto_forecast.cli.train", lambda config, **_: config)
    settings = _demo(parser().parse_args(["demo", "--model-type", "kronos-small",
                                         "--input-mode", "daily", "--run-dir", str(tmp_path)]))
    assert settings.kronos_timeframe == "1d"
    assert settings.batch_size == 8
    assert settings.learning_rate == 1e-5
    assert (tmp_path / "synthetic_data" / "BTC.csv").is_file()


def test_cli_controls_sequential_tokenizer_training():
    settings = _settings(parser().parse_args([
        "train", "--model-type", "kronos-small", "--kronos-tokenizer-epochs", "3",
        "--kronos-tokenizer-patience", "1", "--kronos-tokenizer-learning-rate", "0.0002",
    ]))
    assert settings.kronos_finetune_tokenizer is True
    assert settings.kronos_tokenizer_epochs == 3
    assert settings.kronos_tokenizer_patience == 1
    assert settings.kronos_tokenizer_learning_rate == 0.0002
    settings.validate()
    skipped = _settings(parser().parse_args(["train", "--no-kronos-finetune-tokenizer"]))
    assert skipped.kronos_finetune_tokenizer is False


@pytest.mark.parametrize("settings", [
    {"kronos_finetune_tokenizer": "false"}, {"kronos_finetune_tokenizer": 1},
    {"kronos_tokenizer_epochs": 0}, {"kronos_tokenizer_epochs": True},
    {"kronos_tokenizer_patience": 0}, {"kronos_tokenizer_patience": 1.5},
    {"kronos_tokenizer_learning_rate": 0}, {"kronos_tokenizer_learning_rate": True},
    {"kronos_tokenizer_learning_rate": "0.0002"}, {"kronos_tokenizer_learning_rate": float("nan")},
])
def test_tokenizer_training_settings_are_validated(settings):
    with pytest.raises(ValueError, match="kronos_"):
        TrainingConfig(coins=["BTC"], model_type="kronos-small", **settings).validate()

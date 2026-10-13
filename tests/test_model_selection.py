"""Selected recurrent architectures really train and reconstruct frozen forecasts."""

import hashlib
import json
import threading
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_forecast import HORIZONS
from crypto_forecast.cli import _settings, parser
from crypto_forecast.dashboard import DashboardError, JobManager, create_server
from crypto_forecast.data import aggregate_candles, load_multiframe_data
from crypto_forecast.features import FEATURE_NAMES, split_samples
from crypto_forecast.model import ForecastLSTM, MultiTimeframeLSTM, Normalizer
from crypto_forecast.multiframe import MTF_FEATURE_NAMES, build_multiframe_samples, inference_multiframe_sample
from crypto_forecast.pipeline import TrainingConfig, _predict, predict, train
from crypto_forecast.timeframes import TIMEFRAMES


@pytest.fixture(scope="module")
def model_history(tmp_path_factory):
    """A labeled synthetic market with enough history for 30-day purged splits."""
    directory = tmp_path_factory.mktemp("model-selection-history")
    data_dir = directory / "data"
    length = 340 * 96
    index = np.arange(length)
    close = 100 * np.exp(np.cumsum(0.00002 + 0.0004 * np.sin(index / 191)))
    base = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=length, freq="15min", tz="UTC"),
        "open": close * .999, "high": close * 1.005, "low": close * .995,
        "close": close, "volume": 1000 + 150 * np.sin(index / 37),
    })
    end = (base.timestamp.iloc[-1] + pd.Timedelta(minutes=15)).isoformat()
    for interval in TIMEFRAMES:
        path = data_dir / interval / "BTC.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        candles = aggregate_candles(base, interval, as_of=end)
        candles.to_csv(path, index=False)
        provenance = {
            "source": "synthetic_demo", "coin": "BTC", "quote_currency": "USDT",
            "interval": interval, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(".metadata.json").write_text(json.dumps(provenance), encoding="utf-8")
        if interval == "1d":
            daily = data_dir / "BTC.csv"
            daily.write_bytes(path.read_bytes())
            daily.with_suffix(".metadata.json").write_text(json.dumps(provenance), encoding="utf-8")
    return TrainingConfig(
        coins=["BTC"], start="2024-01-01", end=end, data_dir=str(data_dir),
        lookback=6, hidden_size=4, epochs=1, patience=1, batch_size=128,
        device="cpu", seed=7,
    ), directory


@pytest.fixture(scope="module")
def lstm_run(model_history):
    base, directory = model_history
    config = replace(base, model_type="lstm", run_dir=str(directory / "lstm-run"))
    summary = train(config, progress=lambda _: None)
    checkpoint = torch.load(Path(config.run_dir) / "model.pt", weights_only=True, map_location="cpu")
    return config, summary, checkpoint


@pytest.mark.parametrize("multiframe", [False, True], ids=["daily", "four-timeframes"])
def test_lstm_encoders_and_coin_embedding_receive_finite_gradients(multiframe):
    torch.manual_seed(11)
    model_class = MultiTimeframeLSTM if multiframe else ForecastLSTM
    n_features = len(MTF_FEATURE_NAMES if multiframe else FEATURE_NAMES)
    model = model_class(n_features, 2, hidden_size=4)
    x = torch.randn((3, 4, 6, n_features) if multiframe else (3, 6, n_features))
    output = model(x, torch.tensor([0, 1, 0]))
    assert output.shape == (3, len(HORIZONS))
    assert torch.isfinite(output).all()
    torch.nn.functional.mse_loss(output, torch.randn_like(output)).backward()
    encoders = model.encoders if multiframe else [model.encoder]
    for encoder in encoders:
        assert isinstance(encoder, torch.nn.LSTM)
        gradient = encoder.weight_ih_l0.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)
    embedding_gradient = model.coin_embedding.weight.grad
    assert embedding_gradient is not None and torch.isfinite(embedding_gradient).all()
    assert torch.count_nonzero(embedding_gradient)


def test_selected_lstm_learns_without_changing_calendar_splits(lstm_run):
    config, summary, checkpoint = lstm_run
    metadata = checkpoint["metadata"]
    assert summary["model_type"] == metadata["model_type"] == "lstm"
    assert summary["config"]["model_type"] == metadata["config"]["model_type"] == "lstm"
    assert summary["parameters_changed"] is True
    assert summary["data_kind"] == "synthetic_demo"
    assert len(summary["history"]) == 1 and summary["best_epoch"] == 1
    assert all(np.isfinite(value) for value in (
        summary["history"][0]["train_loss"], summary["history"][0]["validation_loss"],
    ))
    frames = load_multiframe_data(config.data_dir, config.coins, as_of=config.end)
    samples = build_multiframe_samples(frames, config.lookback, config.origin_interval)
    partitions, split_info = split_samples(samples, config.train_fraction, config.validation_fraction)
    assert summary["splits"]["counts"] == split_info["counts"]
    assert summary["splits"]["purged"] == split_info["purged"]
    assert max(s.label_end for s in partitions["train"]) < pd.Timestamp(split_info["val_start"])
    assert max(s.label_end for s in partitions["validation"]) < pd.Timestamp(split_info["test_start"])
    for rows in summary["metrics"].values():
        assert {row["model"] for row in rows} == {"lstm", "persistence", "drift"}
        assert all(row["interval_coverage"] is not None for row in rows if row["model"] == "lstm")
    torch.manual_seed(config.seed)
    initial = MultiTimeframeLSTM(len(MTF_FEATURE_NAMES), 1, config.hidden_size).state_dict()
    for branch in range(len(TIMEFRAMES)):
        name = f"encoders.{branch}.weight_ih_l0"
        assert not torch.equal(initial[name], checkpoint["state_dict"][name])


def test_frozen_lstm_reconstructs_identical_forecasts_and_exact_horizons(lstm_run):
    config, _, checkpoint = lstm_run
    metadata = checkpoint["metadata"]
    model = MultiTimeframeLSTM(len(MTF_FEATURE_NAMES), 1, config.hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    frames = load_multiframe_data(config.data_dir, config.coins, as_of=config.end)
    sample = inference_multiframe_sample("BTC", frames["BTC"], config.lookback, config.origin_interval)
    direct = _predict(model, [sample], {"BTC": 0}, Normalizer.from_dict(metadata["x_scaler"]),
                      Normalizer.from_dict(metadata["y_scaler"]), torch.device("cpu"))
    forecasts = predict(config.run_dir, config.data_dir, allow_stale=True, update_report=False)
    assert len(forecasts) == len(HORIZONS)
    for horizon_index, horizon in enumerate(HORIZONS):
        row = next(row for row in forecasts if row["horizon_days"] == horizon)
        assert row["forecast_price_usdt"] == float(sample.price * np.exp(direct[0, horizon_index]))
        assert row["change_pct"] == float(100 * np.expm1(direct[0, horizon_index]))
        assert pd.Timestamp(row["target_date"]) == sample.origin + pd.Timedelta(days=horizon)
    assert json.loads((Path(config.run_dir) / "forecasts.json").read_text()) == forecasts


@pytest.mark.parametrize("input_mode", ["daily", "multiframe"])
def test_legacy_gru_checkpoint_without_architecture_field_preserves_forecasts(model_history, input_mode):
    base, directory = model_history
    config = replace(base, model_type="gru", input_mode=input_mode,
                     run_dir=str(directory / f"legacy-{input_mode}"))
    train(config, progress=lambda _: None)
    expected = predict(config.run_dir, config.data_dir, allow_stale=True, update_report=False)
    path = Path(config.run_dir) / "model.pt"
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    checkpoint["metadata"].pop("model_type")
    checkpoint["metadata"]["config"].pop("model_type")
    torch.save(checkpoint, path)
    actual = predict(config.run_dir, config.data_dir, allow_stale=True, update_report=False)
    assert actual == expected


@pytest.mark.parametrize("model_type", ["transformer", "", None, []])
def test_invalid_architecture_fails_before_any_data_loading(monkeypatch, model_type):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Architecture validation must happen before opening training history.")

    monkeypatch.setattr("crypto_forecast.pipeline.load_multiframe_data", unexpected_load)
    monkeypatch.setattr("crypto_forecast.pipeline.load_data", unexpected_load)
    with pytest.raises(ValueError, match="model_type"):
        train(TrainingConfig(coins=["BTC"], model_type=model_type), progress=lambda _: None)


@pytest.mark.parametrize("model_type", ["gru", "lstm"])
def test_dashboard_stores_the_chosen_architecture_and_preserves_fixed_timeframes(tmp_path, monkeypatch, model_type):
    manager = JobManager(tmp_path)
    launches = []
    monkeypatch.setattr(manager, "_launch", lambda job: launches.append(job))
    try:
        job = manager.start_job({"operation": "train", "config": {"coins": ["BTC"], "model_type": model_type}})
        assert job["config"]["model_type"] == model_type
        assert job["config"]["input_mode"] == "multiframe"
        assert job["config"]["origin_interval"] == "4h"
        assert len(launches) == 1
        assert manager.get_job(job["id"])["config"]["model_type"] == model_type
    finally:
        manager.close()


def test_dashboard_rejects_unavailable_architecture_before_launch(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    launches = []
    monkeypatch.setattr(manager, "_launch", lambda job: launches.append(job))
    try:
        with pytest.raises(DashboardError, match="model_type") as error:
            manager.start_job({"operation": "train", "config": {"coins": ["BTC"], "model_type": "transformer"}})
        assert error.value.status == 400
        assert launches == []
    finally:
        manager.close()


def test_dashboard_bootstrap_advertises_trainable_models(tmp_path):
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{server.server_address[1]}/api/bootstrap", timeout=5) as response:
            bootstrap = json.load(response)
        assert {model["id"] for model in bootstrap["models"]} == {"gru", "lstm", "kronos-small", "kronos-base"}
        assert all(model["name"] and model["description"] for model in bootstrap["models"])
        assert bootstrap["defaults"]["model_type"] == "gru"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_architecture_override_and_legacy_default():
    assert _settings(parser().parse_args(["train"])).model_type == "gru"
    assert _settings(parser().parse_args(["train", "--model-type", "lstm"])).model_type == "lstm"
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["train", "--model-type", "transformer"])
    assert error.value.code == 2

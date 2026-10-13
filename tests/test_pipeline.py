"""End-to-end checks using labeled synthetic history, never market performance claims."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_forecast import HORIZONS
from crypto_forecast.data import load_data
from crypto_forecast.features import FEATURE_NAMES, build_samples, inference_sample, split_samples
from crypto_forecast.model import ForecastGRU, Normalizer
from crypto_forecast.pipeline import TrainingConfig, _predict, predict, train


def _completed_candles(coin, days=500):
    index = np.arange(days)
    rate = 0.001 + 0.004 * np.sin(index / 11 + (coin == "ETH"))
    close = (100 if coin == "BTC" else 30) * np.exp(np.cumsum(rate))
    # A large future change deliberately challenges any accidentally global scaler.
    close[index >= 360] *= 12
    volume = 1000 + 100 * np.sin(index / 9)
    volume[index >= 360] *= 100000
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(end="2025-09-30", periods=days, tz="UTC"),
            "open": close * 0.995,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("forecast-integration")
    data_dir = directory / "data"
    data_dir.mkdir()
    frames = {coin: _completed_candles(coin) for coin in ("BTC", "ETH")}
    for coin, frame in frames.items():
        frame.to_csv(data_dir / f"{coin}.csv", index=False)
    config = TrainingConfig(
        coins=list(frames),
        input_mode="daily",
        data_dir=str(data_dir),
        run_dir=str(directory / "run"),
        start="2020-01-01",
        end="2025-10-01",
        lookback=10,
        hidden_size=12,
        epochs=2,
        patience=2,
        batch_size=64,
        device="cpu",
        seed=7,
    )
    summary = train(config, data_kind="synthetic", progress=lambda _: None)
    return config, summary


def test_real_gradient_training_and_training_only_normalizers(trained_run):
    config, summary = trained_run
    checkpoint = torch.load(Path(config.run_dir) / "model.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    frames = load_data(config.data_dir, config.coins, as_of=pd.Timestamp(config.end, tz="UTC"))
    samples = build_samples(frames, config.lookback)
    splits, expected_split_info = split_samples(samples, config.train_fraction, config.validation_fraction)
    training_x = np.concatenate([sample.x for sample in splits["train"]])
    training_y = np.stack([sample.y for sample in splits["train"]])
    expected_x = Normalizer.fit(training_x)
    expected_y = Normalizer.fit(training_y)
    np.testing.assert_allclose(metadata["x_scaler"]["mean"], expected_x.mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(metadata["x_scaler"]["scale"], expected_x.scale, rtol=0, atol=1e-12)
    np.testing.assert_allclose(metadata["y_scaler"]["mean"], expected_y.mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(metadata["y_scaler"]["scale"], expected_y.scale, rtol=0, atol=1e-12)
    all_x = np.concatenate([sample.x for sample in samples])
    volume_column = FEATURE_NAMES.index("log_volume_change_1d")
    assert abs(float(all_x[:, volume_column].mean()) - expected_x.mean[volume_column]) > 0.01
    assert summary["parameters_changed"] is True
    torch.manual_seed(config.seed)
    initial_model = ForecastGRU(len(FEATURE_NAMES), len(config.coins), config.hidden_size)
    assert any(
        not torch.equal(checkpoint["state_dict"][name], initial_tensor)
        for name, initial_tensor in initial_model.state_dict().items()
    )
    assert len(summary["history"]) == 2
    assert summary["best_epoch"] in (1, 2)
    assert all(np.isfinite(item["train_loss"]) and np.isfinite(item["validation_loss"]) for item in summary["history"])
    assert summary["splits"]["counts"] == expected_split_info["counts"]
    assert summary["splits"]["purged"] == {"train": 60, "validation": 60, "total": 120}
    assert max(sample.label_end for sample in splits["train"]) < pd.Timestamp(summary["splits"]["val_start"])
    assert max(sample.label_end for sample in splits["validation"]) < pd.Timestamp(summary["splits"]["test_start"])


def test_pure_weights_artifact_reconstructs_identical_predictions(trained_run):
    config, _ = trained_run
    run_dir = Path(config.run_dir)
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    assert metadata["data_kind"] == "synthetic_demo"
    assert metadata["coins"] == config.coins
    assert all(torch.is_tensor(tensor) for tensor in checkpoint["state_dict"].values())
    frozen_model = ForecastGRU(len(FEATURE_NAMES), len(metadata["coins"]), metadata["config"]["hidden_size"])
    frozen_model.load_state_dict(checkpoint["state_dict"])
    frames = load_data(config.data_dir, config.coins, as_of=pd.Timestamp(config.end, tz="UTC"))
    samples = [inference_sample(coin, frames[coin], config.lookback) for coin in config.coins]
    mapping = {coin: index for index, coin in enumerate(metadata["coins"])}
    x_scaler = Normalizer.from_dict(metadata["x_scaler"])
    y_scaler = Normalizer.from_dict(metadata["y_scaler"])
    direct = _predict(frozen_model, samples, mapping, x_scaler, y_scaler, torch.device("cpu"))
    before = predict(run_dir, config.data_dir, allow_stale=True, update_report=False)
    after = predict(run_dir, config.data_dir, allow_stale=True, update_report=False)
    assert before == after
    for index, sample in enumerate(samples):
        for horizon_index, horizon in enumerate(HORIZONS):
            row = next(item for item in before if item["coin"] == sample.coin and item["horizon_days"] == horizon)
            assert row["forecast_price_usdt"] == float(sample.price * np.exp(direct[index, horizon_index]))
            assert row["change_pct"] == float(100 * np.expm1(direct[index, horizon_index]))
    assert json.loads((run_dir / "forecasts.json").read_text()) == after


def test_forecast_dates_prices_and_intervals_are_consistent(trained_run):
    config, _ = trained_run
    rows = predict(config.run_dir, config.data_dir, allow_stale=True, update_report=False)
    assert len(rows) == 2 * len(HORIZONS)
    assert all(row["data_kind"] == "synthetic_demo" for row in rows)
    assert all(row["is_stale"] is True for row in rows)
    for row in rows:
        origin = pd.Timestamp(row["as_of"])
        assert origin == pd.Timestamp("2025-10-01", tz="UTC")
        assert pd.Timestamp(row["target_date"]) == origin + pd.Timedelta(days=row["horizon_days"])
        assert row["forecast_price_usdt"] == pytest.approx(row["last_price_usdt"] * (1 + row["change_pct"] / 100))
        assert 0 < row["lower_price_usdt"] <= row["upper_price_usdt"]


def test_unknown_coin_and_stale_data_are_rejected(trained_run):
    config, _ = trained_run
    with pytest.raises(ValueError, match="trained coins"):
        predict(config.run_dir, config.data_dir, coins=["DOGE"], update_report=False)
    with pytest.raises(ValueError, match="Stale daily data"):
        predict(config.run_dir, config.data_dir, max_age_days=2, update_report=False)
    rows = predict(config.run_dir, config.data_dir, coins=["btc"], allow_stale=True, update_report=False)
    assert {row["coin"] for row in rows} == {"BTC"}
    assert {row["horizon_days"] for row in rows} == set(HORIZONS)


def test_residual_intervals_are_fitted_only_to_validation(trained_run):
    config, _ = trained_run
    run_dir = Path(config.run_dir)
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    validation = pd.read_csv(run_dir / "validation_predictions.csv")
    residuals = []
    for horizon in HORIZONS:
        block = validation[validation.horizon_days == horizon]
        actual = np.log(block.actual_price_usdt / block.last_price_usdt)
        predicted = np.log(block.forecast_price_usdt / block.last_price_usdt)
        residuals.append(np.quantile(actual - predicted, [0.1, 0.9]))
    expected_bounds = np.asarray(residuals).T
    np.testing.assert_allclose(checkpoint["metadata"]["residual_bounds"], expected_bounds, rtol=1e-7, atol=1e-7)
    test_rows = pd.read_csv(run_dir / "test_predictions.csv")
    for _, row in test_rows.iterrows():
        assert pd.Timestamp(row.target_date) == pd.Timestamp(row.as_of) + pd.Timedelta(days=int(row.horizon_days))
        assert row.actual_price_usdt == pytest.approx(row.last_price_usdt * (1 + row.actual_change_pct / 100))


def test_frozen_run_is_not_overwritten_without_explicit_flag(trained_run):
    config, _ = trained_run
    model_path = Path(config.run_dir) / "model.pt"
    before = model_path.read_bytes()
    with pytest.raises(ValueError, match="already contains a trained run"):
        train(config, data_kind="synthetic", progress=lambda _: None)
    assert model_path.read_bytes() == before


@pytest.mark.parametrize(
    "cutoff",
    ["2025-10-01", pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-01T03:30:00+03:30")],
)
def test_public_predict_normalizes_naive_and_aware_cutoffs(trained_run, cutoff):
    config, _ = trained_run
    expected = predict(
        config.run_dir,
        config.data_dir,
        allow_stale=True,
        as_of=pd.Timestamp("2025-10-01", tz="UTC"),
        update_report=False,
    )
    actual = predict(config.run_dir, config.data_dir, allow_stale=True, as_of=cutoff, update_report=False)
    assert actual == expected


def test_historical_prediction_cutoff_excludes_future_candles(trained_run):
    config, _ = trained_run
    rows = predict(config.run_dir, config.data_dir, allow_stale=True, as_of="2025-09-01", update_report=False)
    assert {row["as_of"] for row in rows} == {"2025-09-01T00:00:00+00:00"}
    frames = load_data(config.data_dir, config.coins, as_of="2025-09-01")
    for row in rows:
        assert row["last_price_usdt"] == frames[row["coin"]].close.iloc[-1]


@pytest.mark.parametrize(
    "changes",
    [
        {"coins": "BTC"},
        {"coins": []},
        {"coins": ["btc", "BTC"]},
        {"coins": ["USDT"]},
        {"coins": ["BTC/USDT"]},
        {"lookback": True},
        {"hidden_size": 1.5},
        {"epochs": 0},
        {"learning_rate": float("nan")},
        {"device": "invalid"},
        {"start": "2025-01-01", "end": "2024-01-01"},
        {"start": "2020-01-01T01:00:00Z"},
    ],
)
def test_invalid_training_configuration_fails_before_loading_data(changes):
    settings = {"coins": ["BTC"], "start": "2020-01-01", "end": "2025-01-01", **changes}
    with pytest.raises(ValueError):
        TrainingConfig(**settings).validate()


@pytest.mark.parametrize("overwrite", [False, True])
def test_report_failure_does_not_publish_new_or_change_existing_run(trained_run, tmp_path, monkeypatch, overwrite):
    config, _ = trained_run
    destination = Path(config.run_dir) if overwrite else tmp_path / "failed-new-run"
    settings = replace(config, run_dir=str(destination), epochs=1)
    before = {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } if overwrite else None

    def fail_report(*args, **kwargs):
        raise RuntimeError("Injected report preparation failure")

    monkeypatch.setattr("crypto_forecast.pipeline.write_report", fail_report)
    with pytest.raises(RuntimeError, match="Injected report preparation failure"):
        train(settings, overwrite=overwrite, data_kind="synthetic", progress=lambda _: None)
    if overwrite:
        after = {
            path.relative_to(destination): path.read_bytes()
            for path in destination.rglob("*")
            if path.is_file()
        }
        assert after == before
    else:
        assert not destination.exists()


def test_latest_history_gap_excludes_coin_and_keeps_eligible_coin(trained_run, tmp_path):
    config, _ = trained_run
    data_dir = tmp_path / "gapped-data"
    data_dir.mkdir()
    for coin in config.coins:
        frame = _completed_candles(coin)
        if coin == "BTC":
            frame = frame.drop(index=480)
        frame.to_csv(data_dir / f"{coin}.csv", index=False)
    settings = replace(config, data_dir=str(data_dir), run_dir=str(tmp_path / "gapped-run"), epochs=1)
    summary = train(settings, data_kind="synthetic", progress=lambda _: None)
    assert summary["coins"] == ["ETH"]
    assert "Latest forecasting window unavailable" in summary["excluded_coins"]["BTC"]
    checkpoint = torch.load(Path(settings.run_dir) / "model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["coins"] == ["ETH"]
    forecasts = json.loads((Path(settings.run_dir) / "forecasts.json").read_text())
    assert {row["coin"] for row in forecasts} == {"ETH"}


def test_synthetic_provenance_and_original_data_hash_survive_source_changes(trained_run, tmp_path):
    config, _ = trained_run
    data_dir = tmp_path / "synthetic-source"
    data_dir.mkdir()
    originals = {}
    provenance = {}
    for coin in config.coins:
        frame = _completed_candles(coin)
        csv_path = data_dir / f"{coin}.csv"
        frame.to_csv(csv_path, index=False)
        originals[coin] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        provenance[coin] = {
            "coin": coin,
            "source": "Synthetic integration test data",
            "quote_currency": "USDT",
            "interval": "1d",
            "data_sha256": originals[coin],
        }
        (data_dir / f"{coin}.metadata.json").write_text(json.dumps(provenance[coin]))
    initial_frames = load_data(data_dir, config.coins, as_of=config.end)
    settings = replace(config, data_dir=str(data_dir), run_dir=str(tmp_path / "source-change-run"), epochs=1)
    mutated = False

    def mutate_sources_after_load(message):
        nonlocal mutated
        if mutated:
            return
        mutated = True
        for coin in config.coins:
            changed = _completed_candles(coin)
            changed.loc[:, ["open", "high", "low", "close"]] *= 2
            csv_path = data_dir / f"{coin}.csv"
            changed.to_csv(csv_path, index=False)
            replacement = {
                **provenance[coin],
                "source": "Changed after training began",
                "data_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            }
            (data_dir / f"{coin}.metadata.json").write_text(json.dumps(replacement))

    # The ordinary training API defaults to historical. Loaded source metadata
    # must still identify synthetic data and must describe exactly the input used.
    summary = train(settings, progress=mutate_sources_after_load)
    assert mutated
    assert summary["data_kind"] == "synthetic_demo"
    run_dir = Path(settings.run_dir)
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["data_kind"] == "synthetic_demo"
    assert "Synthetic demonstration" in (run_dir / "report.html").read_text(encoding="utf-8")
    for coin in config.coins:
        manifest = summary["data_manifest"][coin]
        assert manifest["csv_sha256"] == originals[coin]
        assert manifest["csv_sha256"] != hashlib.sha256((data_dir / f"{coin}.csv").read_bytes()).hexdigest()
        assert manifest["used_rows_sha256"] == hashlib.sha256(initial_frames[coin].to_csv(index=False).encode()).hexdigest()
        assert manifest["provenance"] == provenance[coin]
    forecasts = json.loads((run_dir / "forecasts.json").read_text())
    assert {row["data_kind"] for row in forecasts} == {"synthetic_demo"}
    for row in forecasts:
        assert row["last_price_usdt"] == initial_frames[row["coin"]].close.iloc[-1]

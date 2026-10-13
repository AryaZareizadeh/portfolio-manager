"""Real branch learning and frozen multi-resolution forecast reconstruction."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_forecast.data import aggregate_candles, load_multiframe_data
from crypto_forecast.features import split_samples
from crypto_forecast.model import MultiTimeframeGRU, Normalizer
from crypto_forecast.multiframe import MTF_FEATURE_NAMES, build_multiframe_samples, inference_multiframe_sample
from crypto_forecast.pipeline import TrainingConfig, _predict, predict, train
from crypto_forecast.timeframes import TIMEFRAMES


@pytest.fixture(scope="module")
def multiframe_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("multiframe-integration")
    data_dir = directory / "data"
    length = 400 * 96
    index = np.arange(length)
    timestamps = pd.date_range("2024-01-01", periods=length, freq="15min", tz="UTC")
    for coin in ("BTC", "ETH"):
        close = (100 if coin == "BTC" else 30) * np.exp(np.cumsum(
            0.00001 + 0.0002*np.sin(index/170 + (coin == "ETH"))))
        volume = 1000 + 200*np.sin(index/37)
        volume[index >= 300*96] *= 100
        base = pd.DataFrame({"timestamp": timestamps, "open": close*.999,
                             "high": close*1.005, "low": close*.995,
                             "close": close, "volume": volume})
        for interval in TIMEFRAMES:
            path = data_dir / interval / f"{coin}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            aggregate_candles(base, interval, as_of="2025-02-04").to_csv(path, index=False)
            path.with_suffix(".metadata.json").write_text(json.dumps({
                "source": "synthetic_demo", "coin": coin, "quote_currency": "USDT",
                "interval": interval, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            }), encoding="utf-8")
    config = TrainingConfig(coins=["BTC", "ETH"], start="2024-01-01", end="2025-02-04",
                            data_dir=str(data_dir), run_dir=str(directory / "run"),
                            lookback=10, hidden_size=8, epochs=1, patience=1,
                            batch_size=128, device="cpu", seed=7)
    summary = train(config, progress=lambda _: None)
    checkpoint = torch.load(Path(config.run_dir) / "model.pt", weights_only=True, map_location="cpu")
    frames = load_multiframe_data(data_dir, config.coins, as_of=config.end)
    return config, summary, checkpoint, frames


def test_each_encoder_learns_and_branch_scalers_use_only_training(multiframe_run):
    config, summary, checkpoint, frames = multiframe_run
    samples = build_multiframe_samples(frames, config.lookback)
    splits, _ = split_samples(samples, config.train_fraction, config.validation_fraction)
    values = np.stack([sample.x for sample in splits["train"]]).astype(np.float64)
    metadata = checkpoint["metadata"]
    np.testing.assert_allclose(metadata["x_scaler"]["mean"], values.mean(axis=(0, 2))[:, None], atol=1e-12)
    np.testing.assert_allclose(metadata["x_scaler"]["scale"],
                               np.maximum(values.std(axis=(0, 2)), 1e-6)[:, None], atol=1e-12)
    # The volume regime change occurs only after the training boundary.
    all_volume = np.stack([s.x[..., -1] for s in samples]).mean(axis=(0, 2))
    assert np.all(all_volume - values[..., -1].mean(axis=(0, 2)) > .2)
    torch.manual_seed(config.seed)
    initial = MultiTimeframeGRU(len(MTF_FEATURE_NAMES), 2, config.hidden_size).state_dict()
    for branch in range(4):
        name = f"encoders.{branch}.weight_ih_l0"
        assert not torch.equal(initial[name], checkpoint["state_dict"][name])
    assert summary["parameters_changed"]
    assert summary["data_kind"] == "synthetic_demo"
    assert metadata["format_version"] == 2
    assert metadata["timeframes"] == list(TIMEFRAMES)
    assert metadata["origin_interval"] == "4h"
    assert len(metadata["features"]) == 11


def test_frozen_combined_model_roundtrips_and_uses_volume_in_every_branch(multiframe_run):
    config, _, checkpoint, frames = multiframe_run
    metadata = checkpoint["metadata"]
    model = MultiTimeframeGRU(len(MTF_FEATURE_NAMES), 2, config.hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    samples = [inference_multiframe_sample(coin, frames[coin], config.lookback) for coin in config.coins]
    x_scaler = Normalizer.from_dict(metadata["x_scaler"])
    y_scaler = Normalizer.from_dict(metadata["y_scaler"])
    estimates = _predict(model, samples, {"BTC": 0, "ETH": 1}, x_scaler, y_scaler, torch.device("cpu"))
    rows = predict(config.run_dir, config.data_dir, allow_stale=True, update_report=False)
    for i, sample in enumerate(samples):
        for j, horizon in enumerate((1, 7, 30)):
            row = next(r for r in rows if r["coin"] == sample.coin and r["horizon_days"] == horizon)
            assert row["forecast_price_usdt"] == float(sample.price*np.exp(estimates[i, j]))
            assert pd.Timestamp(row["target_date"]) == sample.origin + pd.Timedelta(days=horizon)
            assert sample.origin.hour % 4 == 0
            assert max(sample.input_ends) <= sample.origin
    model.eval()
    x = torch.from_numpy(x_scaler.transform(np.stack([samples[0].x])))
    with torch.inference_mode():
        original = model(x, torch.tensor([0]))
        for branch in range(4):
            changed = x.clone()
            changed[:, branch, :, 7:] += 1
            assert not torch.allclose(original, model(changed, torch.tensor([0])), rtol=1e-6, atol=1e-7)


def test_combined_freshness_and_historical_cutoff(multiframe_run):
    config, _, _, _ = multiframe_run
    with pytest.raises(ValueError, match="Stale multiframe"):
        predict(config.run_dir, config.data_dir, update_report=False)
    rows = predict(config.run_dir, config.data_dir, as_of="2025-02-03T10:10:00Z",
                   allow_stale=True, update_report=False)
    assert {r["as_of"] for r in rows} == {"2025-02-03T08:00:00+00:00"}

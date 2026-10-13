import json

import numpy as np
import pandas as pd
import pytest

from crypto_forecast import HORIZONS
from crypto_forecast.features import (
    FEATURE_NAMES,
    build_samples,
    inference_sample,
    split_samples,
)


def candles(days=500, rate=0.01, offset=0):
    close = 100 * np.exp(rate * np.arange(days))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=days, tz="UTC")
            + pd.Timedelta(days=offset),
            "open": close / np.exp(rate),
            "high": close * 1.02,
            "low": close / np.exp(rate) * 0.98,
            "close": close,
            "volume": np.full(days, 1000.0),
        }
    )


def test_exact_calendar_targets_and_analytical_causal_features():
    frame = candles(120)
    samples = build_samples({"BTC": frame}, lookback=60)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.x.shape == (60, len(FEATURE_NAMES))
    assert sample.x.dtype == np.float32
    assert sample.y.dtype == np.float32
    assert sample.origin == frame.timestamp.iloc[89] + pd.Timedelta(days=1)
    assert sample.label_end == frame.timestamp.iloc[119] + pd.Timedelta(days=1)
    assert sample.price == frame.close.iloc[89]
    np.testing.assert_allclose(sample.y, np.asarray(HORIZONS) * 0.01, atol=1e-7)
    np.testing.assert_allclose(sample.drift, sample.y, atol=1e-7)
    for name, expected in (
        ("log_return_1d", 0.01),
        ("log_close_open", 0.01),
        ("log_high_low", np.log(1.02 / 0.98) + 0.01),
        ("log_return_7d", 0.07),
        ("log_return_30d", 0.30),
        ("volatility_7d", 0),
        ("volatility_30d", 0),
        ("log_volume_change_1d", 0),
        ("log_volume_vs_7d", 0),
        ("log_volume_vs_30d", 0),
    ):
        np.testing.assert_allclose(sample.x[:, FEATURE_NAMES.index(name)], expected, atol=1e-7)


def test_gaps_remove_whole_calendar_windows_and_never_shift_targets():
    frame = candles(400)
    missing_date = frame.timestamp.iloc[170]
    gapped = frame.drop(index=170).reset_index(drop=True)
    samples = build_samples({"BTC": gapped}, 60)
    assert samples
    for sample in samples:
        context_start = sample.origin - pd.Timedelta(days=90)
        final_target_open = sample.label_end - pd.Timedelta(days=1)
        assert not context_start <= missing_date <= final_target_open
        target_open = sample.origin + pd.Timedelta(days=6)
        target_price = float(gapped.loc[gapped.timestamp == target_open, "close"].iloc[0])
        assert float(sample.y[1]) == pytest.approx(np.log(target_price / sample.price), abs=1e-7)
    assert len(samples) < len(build_samples({"BTC": frame}, 60))


def test_future_changes_do_not_change_inputs_or_drift():
    original = candles(300)
    changed = original.copy()
    changed.loc[151:, ["open", "high", "low", "close"]] *= 3
    changed.loc[151:, "volume"] *= 5
    old = {sample.origin: sample for sample in build_samples({"BTC": original}, 60)}
    new = {sample.origin: sample for sample in build_samples({"BTC": changed}, 60)}
    boundary = original.timestamp.iloc[151]
    for origin in old:
        if origin <= boundary:
            np.testing.assert_array_equal(old[origin].x, new[origin].x)
            np.testing.assert_array_equal(old[origin].drift, new[origin].drift)
    assert any(not np.array_equal(old[origin].y, new[origin].y) for origin in old if origin <= boundary)


def test_inference_uses_latest_candle_and_matches_training_features():
    frame = candles(250)
    samples = build_samples({"BTC": frame}, 60)
    expected = samples[-1]
    history = frame[frame.timestamp < expected.origin]
    inferred = inference_sample("BTC", history, 60)
    assert inferred.origin == history.timestamp.iloc[-1] + pd.Timedelta(days=1)
    assert inferred.price == history.close.iloc[-1]
    assert np.isnan(inferred.y).all()
    np.testing.assert_array_equal(inferred.x, expected.x)
    np.testing.assert_array_equal(inferred.drift, expected.drift)


def test_inference_rejects_latest_gap_instead_of_using_older_sequence():
    frame = candles(250).drop(index=230)
    with pytest.raises(ValueError, match="missing calendar days"):
        inference_sample("BTC", frame, 60)
    with pytest.raises(ValueError, match="at least 90"):
        inference_sample("BTC", candles(89), 60)
    # Earlier gaps outside the latest lookback and warmup need not block inference.
    inferred = inference_sample("BTC", candles(250).drop(index=100), 60)
    assert inferred.origin == pd.Timestamp("2020-09-07", tz="UTC")


def test_global_split_boundaries_purge_overlapping_labels_for_all_coins():
    samples = build_samples({"BTC": candles(900), "ETH": candles(780, offset=120)}, 60)
    splits, metadata = split_samples(samples)
    val_start = pd.Timestamp(metadata["val_start"])
    test_start = pd.Timestamp(metadata["test_start"])
    assert all(sample.origin < val_start and sample.label_end < val_start for sample in splits["train"])
    assert all(val_start <= sample.origin < test_start and sample.label_end < test_start for sample in splits["validation"])
    assert all(sample.origin >= test_start for sample in splits["test"])
    for name in splits:
        assert {sample.coin for sample in splits[name]} == {"BTC", "ETH"}
        assert splits[name] == sorted(splits[name], key=lambda sample: (sample.origin, sample.coin))
    assert metadata["purged"] == {"train": 60, "validation": 60, "total": 120}
    assert sum(metadata["counts"].values()) + metadata["purged"]["total"] == len(samples)
    json.dumps(metadata)


def test_same_dates_stay_in_same_split_across_coins():
    splits, _ = split_samples(build_samples({"BTC": candles(600), "ETH": candles(600, rate=0.02)}, 30))
    assignment = {}
    for name, items in splits.items():
        for item in items:
            assert item.origin not in assignment or assignment[item.origin] == name
            assignment[item.origin] = name


@pytest.mark.parametrize("lookback", [0, -1, 1.5, True])
def test_rejects_invalid_lookback(lookback):
    with pytest.raises(ValueError, match="positive integer"):
        build_samples({"BTC": candles()}, lookback)


def test_duplicate_and_non_daily_timestamps_are_rejected():
    frame = candles()
    frame.loc[50, "timestamp"] = frame.timestamp.iloc[49]
    with pytest.raises(ValueError, match="Duplicate"):
        build_samples({"BTC": frame}, 60)
    frame = candles()
    frame.loc[50, "timestamp"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="UTC midnight"):
        inference_sample("BTC", frame, 60)


def test_split_requires_valid_fractions_and_multiple_origins():
    with pytest.raises(ValueError, match="unique daily origins"):
        split_samples([])
    samples = build_samples({"BTC": candles()}, 60)
    for fractions in ((0.9, 0.2), (0, 0.15), (0.7, 0), (float("nan"), 0.15)):
        with pytest.raises(ValueError, match="Split fractions"):
            split_samples(samples, *fractions)

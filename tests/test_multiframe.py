"""Multi-timeframe candle alignment, volume features, and calendar boundaries."""

import numpy as np
import pandas as pd
import pytest

from crypto_forecast import HORIZONS
from crypto_forecast.features import split_samples
from crypto_forecast.multiframe import (
    MTF_FEATURE_NAMES,
    build_multiframe_samples,
    inference_multiframe_sample,
)
from crypto_forecast.timeframes import TIMEFRAMES, interval_delta


def candles(days=150, rate=0.005):
    count = days * 96
    close = 100 * np.exp(rate * np.arange(count) / 96)
    fine = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=count, freq="15min", tz="UTC"),
        "open": close / np.exp(rate / 96),
        "high": close * 1.002,
        "low": close / np.exp(rate / 96) * 0.998,
        "close": close,
        "volume": np.full(count, 10.0),
    })
    frames = {"15m": fine}
    for interval in TIMEFRAMES[1:]:
        frames[interval] = fine.set_index("timestamp").resample(interval_delta(interval)).agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).reset_index()
    return frames


def through(frames, origin):
    return {
        interval: frame[frame.timestamp + interval_delta(interval) <= origin].copy()
        for interval, frame in frames.items()
    }


def test_analytical_prices_four_hour_cadence_and_exact_calendar_targets():
    frames = candles(days=126)
    samples = build_multiframe_samples({"BTC": frames}, lookback=60)
    assert len(samples) == 37
    first = samples[0]
    assert first.origin == pd.Timestamp("2020-03-31", tz="UTC")
    assert first.x.shape == (4, 60, len(MTF_FEATURE_NAMES))
    assert first.x.dtype == first.y.dtype == np.float32
    assert first.label_end == first.origin + pd.Timedelta(days=30)
    assert first.price == frames["15m"].loc[frames["15m"].timestamp == first.origin - pd.Timedelta(minutes=15), "close"].iloc[0]
    assert all(b.origin - a.origin == pd.Timedelta(hours=4) for a, b in zip(samples, samples[1:]))
    for sample in samples:
        np.testing.assert_allclose(sample.y, np.asarray(HORIZONS) * 0.005, atol=1e-7)
        np.testing.assert_allclose(sample.drift, sample.y, atol=1e-7)
        for branch, interval in enumerate(TIMEFRAMES):
            rate = 0.005 * interval_delta(interval) / pd.Timedelta(days=1)
            for name, expected in (("log_return_1bar", rate), ("log_return_7bar", 7 * rate),
                                   ("log_return_30bar", 30 * rate), ("volatility_7bar", 0),
                                   ("volatility_30bar", 0), ("log_volume_change_1bar", 0),
                                   ("log_volume_vs_7bar", 0), ("log_volume_vs_30bar", 0)):
                np.testing.assert_allclose(sample.x[branch, :, MTF_FEATURE_NAMES.index(name)], expected, atol=1e-7)


def test_unclosed_daily_and_future_candles_never_enter_inputs():
    frames = candles(days=130)
    boundary = pd.Timestamp("2020-04-05 04:00", tz="UTC")
    original = {sample.origin: sample for sample in build_multiframe_samples({"BTC": frames}, 60)}[boundary]
    changed = {interval: frame.copy() for interval, frame in frames.items()}
    for interval, frame in changed.items():
        future = frame.timestamp + interval_delta(interval) > boundary
        frame.loc[future, ["open", "high", "low", "close"]] *= 3
        frame.loc[future, "volume"] *= 5
    modified = {sample.origin: sample for sample in build_multiframe_samples({"BTC": changed}, 60)}[boundary]
    np.testing.assert_array_equal(original.x, modified.x)
    np.testing.assert_array_equal(original.drift, modified.drift)
    assert not np.array_equal(original.y, modified.y)
    assert original.input_ends == (boundary, boundary, boundary, boundary.normalize())
    inferred = inference_multiframe_sample("BTC", through(frames, boundary), 60)
    np.testing.assert_array_equal(inferred.x, original.x)
    np.testing.assert_array_equal(inferred.drift, original.drift)
    assert inferred.origin == boundary
    assert np.isnan(inferred.y).all()


@pytest.mark.parametrize("interval", TIMEFRAMES)
def test_each_branch_uses_its_observed_aggregated_volume(interval):
    boundary = pd.Timestamp("2020-04-05 04:00", tz="UTC")
    frames = through(candles(), boundary)
    original = inference_multiframe_sample("BTC", frames, 60)
    frame = frames[interval]
    previous_volume = frame.volume.iloc[-1]
    # Aggregated candles carry the sum of their constituent 15-minute volumes.
    assert previous_volume == 10 * interval_delta(interval) / pd.Timedelta(minutes=15)
    frame.loc[frame.index[-1], "volume"] *= 50
    changed = inference_multiframe_sample("BTC", frames, 60)
    branch = TIMEFRAMES.index(interval)
    np.testing.assert_array_equal(changed.x[:, :, :7], original.x[:, :, :7])
    other_branches = [index for index in range(4) if index != branch]
    np.testing.assert_array_equal(changed.x[other_branches], original.x[other_branches])
    expected = (
        np.log1p(previous_volume * 50) - np.log1p(previous_volume),
        np.log1p(previous_volume * 50) - np.log1p(frame.volume.iloc[-7:].mean()),
        np.log1p(previous_volume * 50) - np.log1p(frame.volume.iloc[-30:].mean()),
        np.log1p(previous_volume * 50),
    )
    np.testing.assert_allclose(changed.x[branch, -1, 7:], expected, atol=1e-6)


@pytest.mark.parametrize("interval", TIMEFRAMES)
def test_missing_context_candle_rejects_latest_inference_and_affected_training(interval):
    boundary = pd.Timestamp("2020-04-05 04:00", tz="UTC")
    frames = candles()
    frame = frames[interval]
    expected_end = boundary.floor(interval_delta(interval))
    missing_open = expected_end - 3 * interval_delta(interval)
    frames[interval] = frame.loc[frame.timestamp != missing_open].copy()
    assert boundary not in {sample.origin for sample in build_multiframe_samples({"BTC": frames}, 60)}
    with pytest.raises(ValueError, match=f"latest {interval} inference history"):
        inference_multiframe_sample("BTC", through(frames, boundary), 60)


def test_reference_gaps_remove_future_windows_without_shifting_target_dates():
    frames = candles(days=230)
    missing_open = pd.Timestamp("2020-04-20 02:15", tz="UTC")
    frames["15m"] = frames["15m"].loc[frames["15m"].timestamp != missing_open].copy()
    samples = build_multiframe_samples({"BTC": frames}, 10)
    assert samples
    missing_end = missing_open + pd.Timedelta(minutes=15)
    for sample in samples:
        assert not sample.origin - pd.Timedelta(days=7) <= missing_end <= sample.label_end
        future_open = sample.origin + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)
        future_price = frames["15m"].loc[frames["15m"].timestamp == future_open, "close"].iloc[0]
        assert sample.y[1] == pytest.approx(np.log(future_price / sample.price), abs=1e-7)


def test_stale_higher_branch_cannot_silently_roll_back_forecast_origin():
    frames = through(candles(), pd.Timestamp("2020-04-05 04:00", tz="UTC"))
    frames["1d"] = frames["1d"].iloc[:-1]
    with pytest.raises(ValueError, match="latest 1d inference history"):
        inference_multiframe_sample("BTC", frames, 60)


def test_latest_inference_uses_latest_four_hour_boundary_and_excludes_later_bars():
    frames = candles()
    cutoff = pd.Timestamp("2020-04-05 07:45", tz="UTC")
    expected = pd.Timestamp("2020-04-05 04:00", tz="UTC")
    sample = inference_multiframe_sample("BTC", through(frames, cutoff), 60)
    anchored = inference_multiframe_sample("BTC", through(frames, expected), 60)
    assert sample.origin == expected
    np.testing.assert_array_equal(sample.x, anchored.x)
    np.testing.assert_array_equal(sample.drift, anchored.drift)


def test_seven_day_drift_requires_full_reference_history_even_with_short_lookback():
    boundary = pd.Timestamp("2020-04-05 04:00", tz="UTC")
    frames = through(candles(), boundary)
    missing_open = boundary - pd.Timedelta(days=5)
    frames["15m"] = frames["15m"].loc[frames["15m"].timestamp != missing_open]
    with pytest.raises(ValueError, match="seven continuous calendar days"):
        inference_multiframe_sample("BTC", frames, 1)


def test_global_four_hour_split_purges_overlapping_thirty_day_labels():
    samples = build_multiframe_samples({"BTC": candles(days=400), "ETH": candles(days=400, rate=0.003)}, 10)
    splits, metadata = split_samples(samples)
    validation_start = pd.Timestamp(metadata["val_start"])
    test_start = pd.Timestamp(metadata["test_start"])
    assert all(sample.label_end < validation_start for sample in splits["train"])
    assert all(validation_start <= sample.origin and sample.label_end < test_start for sample in splits["validation"])
    assert all(sample.origin >= test_start for sample in splits["test"])
    assert metadata["purged"] == {"train": 360, "validation": 360, "total": 720}
    assert all({sample.coin for sample in partition} == {"BTC", "ETH"} for partition in splits.values())


def test_timestamp_units_sorting_and_zero_volume_are_supported():
    frames = candles()
    expected = inference_multiframe_sample("BTC", frames, 60)
    for interval, frame in frames.items():
        frame["timestamp"] = frame.timestamp.astype("datetime64[us, UTC]")
        frame["volume"] = 0.0
        frames[interval] = frame.iloc[::-1]
    actual = inference_multiframe_sample("BTC", frames, 60)
    assert actual.origin == expected.origin
    assert actual.input_ends == expected.input_ends
    np.testing.assert_array_equal(actual.x[:, :, :10], expected.x[:, :, :10])
    np.testing.assert_array_equal(actual.x[:, :, MTF_FEATURE_NAMES.index("log_volume")], 0)


def test_rejects_missing_branches_duplicate_unaligned_and_invalid_candles():
    frames = candles()
    with pytest.raises(ValueError, match="Missing candle timeframes: 1h"):
        build_multiframe_samples({"BTC": {key: value for key, value in frames.items() if key != "1h"}}, 60)
    bad = frames["1h"].copy()
    bad.loc[1, "timestamp"] = bad.timestamp.iloc[0]
    with pytest.raises(ValueError, match="Duplicate 1h"):
        inference_multiframe_sample("BTC", {**frames, "1h": bad}, 60)
    bad.loc[1, "timestamp"] += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="aligned UTC"):
        inference_multiframe_sample("BTC", {**frames, "1h": bad}, 60)
    bad = frames["4h"].copy()
    bad.loc[0, "volume"] = -1
    with pytest.raises(ValueError, match="volume nonnegative"):
        inference_multiframe_sample("BTC", {**frames, "4h": bad}, 60)


@pytest.mark.parametrize("lookback", [0, -1, True, 1.5])
def test_rejects_invalid_lookbacks(lookback):
    with pytest.raises(ValueError, match="positive integer"):
        build_multiframe_samples({}, lookback)
    with pytest.raises(ValueError, match="positive integer"):
        inference_multiframe_sample("BTC", {}, lookback)


def test_insufficient_history_and_empty_frames_do_not_produce_samples():
    frames = candles(days=89)
    assert build_multiframe_samples({"BTC": frames}, 60) == []
    with pytest.raises(ValueError, match="at least 90"):
        inference_multiframe_sample("BTC", frames, 60)
    empty = {interval: frame.iloc[:0] for interval, frame in frames.items()}
    assert build_multiframe_samples({"BTC": empty}, 60) == []
    with pytest.raises(ValueError, match="needs 15m candle history"):
        inference_multiframe_sample("BTC", empty, 60)


def test_supported_cadence_is_validated():
    with pytest.raises(ValueError, match="Unsupported candle interval"):
        build_multiframe_samples({}, 60, origin_interval="2h")

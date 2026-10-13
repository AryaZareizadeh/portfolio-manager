"""Analytical checks for reported errors, coin weighting, and sampling."""

import math

import numpy as np
import pandas as pd
import pytest

from crypto_forecast.features import Sample
from crypto_forecast.pipeline import _metrics


def _sample(coin, offset, returns, drift=(0.0, 0.0, 0.0)):
    origin = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=offset)
    return Sample(
        x=np.zeros((1, 1)), y=np.log1p(returns), coin=coin, origin=origin,
        label_end=origin + pd.Timedelta(days=30), price=100.0, drift=np.log1p(drift),
    )


def _row(rows, coin, horizon, model="gru"):
    return next(row for row in rows if (row["coin"], row["horizon_days"], row["model"]) == (coin, horizon, model))


def test_errors_are_cumulative_return_percentage_points_with_comparable_baselines():
    samples = [
        _sample("BTC", 0, [0.10, 0.20, -0.10], drift=[0.02, 0.03, 0.06]),
        _sample("BTC", 1, [-0.05, -0.10, 0.30], drift=[0.01, 0.02, 0.05]),
    ]
    predictions = np.log1p([[0.08, 0.30, -0.08], [-0.02, -0.07, 0.26]])
    rows = _metrics(samples, predictions, np.zeros((2, 3)))
    # GRU return errors in pp: (-2, +3), (+10, +3), (+2, -4).
    for horizon, mae, rmse in ((1, 2.5, math.sqrt(6.5)), (7, 6.5, math.sqrt(54.5)), (30, 3.0, math.sqrt(10))):
        row = _row(rows, "BTC", horizon)
        assert row["samples"] == 2
        assert row["return_mae_pct"] == pytest.approx(mae)
        assert row["return_rmse_pct"] == pytest.approx(rmse)
    expected_log_mae = (math.log(1.10 / 1.08) + math.log(0.98 / 0.95)) / 2
    assert _row(rows, "BTC", 1)["log_return_mae"] == pytest.approx(expected_log_mae)
    unchanged = _row(rows, "BTC", 1, "persistence")
    assert unchanged["return_mae_pct"] == pytest.approx(7.5)
    assert unchanged["return_rmse_pct"] == pytest.approx(math.sqrt(62.5))
    drift = _row(rows, "BTC", 1, "drift")
    assert drift["return_mae_pct"] == pytest.approx(7.0)
    assert drift["return_rmse_pct"] == pytest.approx(math.sqrt(50.0))


def test_macro_metrics_give_shorter_history_coins_equal_weight():
    samples = [_sample("BTC", day, [0, 0, 0]) for day in range(3)]
    samples.append(_sample("ETH", 0, [0, 0, 0]))
    predictions = np.log1p([[0.02] * 3] * 3 + [[0.10] * 3])
    rows = _metrics(samples, predictions, np.zeros((2, 3)))
    for horizon in (1, 7, 30):
        macro = _row(rows, "ALL", horizon)
        assert _row(rows, "BTC", horizon)["return_mae_pct"] == pytest.approx(2.0)
        assert _row(rows, "ETH", horizon)["return_mae_pct"] == pytest.approx(10.0)
        assert macro["samples"] == 4
        assert macro["return_mae_pct"] == pytest.approx(6.0)
        assert macro["return_rmse_pct"] == pytest.approx(6.0)
        # A pooled sample-weighted MAE would be 4 pp, which is not the contract.
        assert macro["return_mae_pct"] != pytest.approx(4.0)


def test_direction_accuracy_excludes_zero_returns_and_ranges_count_actual_coverage():
    samples = [_sample("BTC", day, [change] * 3) for day, change in enumerate((0.10, -0.10, 0.0, 0.20))]
    predictions = np.log1p([[change] * 3 for change in (0.05, 0.05, 0.10, 0.0)])
    # A gross forecast return is multiplied by 0.95 and 1.10 for the bounds.
    bounds = np.array([[math.log(0.95)] * 3, [math.log(1.10)] * 3])
    rows = _metrics(samples, predictions, bounds)
    row = _row(rows, "BTC", 1)
    assert row["direction_accuracy"] == pytest.approx(0.5)
    assert row["interval_coverage"] == pytest.approx(0.25)
    assert row["interval_width_pct"] == pytest.approx(15.75)
    baseline = _row(rows, "BTC", 1, "persistence")
    assert baseline["direction_accuracy"] is None
    assert baseline["interval_coverage"] is None
    assert baseline["interval_width_pct"] is None


def test_non_overlapping_origins_are_spaced_for_each_coin_and_horizon():
    # Offset listing histories and reverse ordering challenge global or row-based strides.
    samples = [_sample("BTC", day, [0.01] * 3) for day in range(61)]
    samples += [_sample("ETH", day, [0.01] * 3) for day in range(3, 64)]
    samples.reverse()
    predictions = np.zeros((len(samples), 3))
    full = _metrics(samples, predictions, np.zeros((2, 3)))
    independent = _metrics(samples, predictions, np.zeros((2, 3)), non_overlapping=True)
    for horizon, expected in ((1, 61), (7, 9), (30, 3)):
        for model in ("gru", "persistence", "drift"):
            for coin in ("BTC", "ETH"):
                assert _row(full, coin, horizon, model)["samples"] == 61
                assert _row(independent, coin, horizon, model)["samples"] == expected
            assert _row(independent, "ALL", horizon, model)["samples"] == 2 * expected

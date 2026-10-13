"""Historical chart selection must never supply subsequent observations to Kronos."""

import pandas as pd
import pytest

from crypto_forecast.chart_forecast import testing_chart as full_chart, prepare_testing, forecast_chart
from crypto_forecast.dashboard_worker import run_job
from test_chart_forecast import install_tiny, hashes
from test_dashboard import dashboard, _post
from test_kronos import candles


def save(workspace, frame):
    path = workspace / "data" / "1d" / "BTC.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def selection(**values):
    return {"coin": "BTC", "timeframe": "1d", "lookback": 8, "horizon": 3,
            "anchor": "2024-01-10T00:00:00Z", "device": "cpu", **values}


def test_full_chart_is_not_limited_to_model_context_and_can_display_gaps(tmp_path):
    save(tmp_path, candles(700).drop(index=100))
    chart = full_chart(tmp_path, {"coin": "BTC", "timeframe": "1d", "lookback": 512})
    assert len(chart["history"]) == chart["available_candles"] == 699
    assert chart["history"][0]["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert chart["forecast"] == []
    save(tmp_path, candles(5))
    assert len(full_chart(tmp_path, {"timeframe": "1d"})["history"]) == 5


def test_selected_candle_is_last_input_and_actual_future_is_separate(tmp_path):
    save(tmp_path, candles(20))
    value, chart = prepare_testing(tmp_path, selection())
    assert len(chart["history"]) == 8
    assert chart["history"][0]["timestamp"] == "2024-01-03T00:00:00+00:00"
    assert chart["history"][-1]["timestamp"] == chart["selected_candle"] == "2024-01-10T00:00:00+00:00"
    assert value["end"] == chart["origin"] == "2024-01-11T00:00:00+00:00"
    assert len(chart["actual"]) == 3 and chart["actual"][0]["timestamp"] == chart["origin"]
    _, last = prepare_testing(tmp_path, selection(anchor="2024-01-20"))
    assert last["actual"] == []


@pytest.mark.parametrize("changed, message", [
    ({"anchor": "2024-01-02"}, "Need 8"),
    ({"anchor": "2024-01-10T12:00:00Z"}, "not present"),
    ({"anchor": "2099-01-10"}, "not present"),
    ({"anchor": "NaT"}, "valid candle"),
    ({"anchor": None}, "Select a completed"),
    ({"end": "2024-01-15"}, "clicked chart point"),
])
def test_selection_rejects_invalid_or_insufficient_context(tmp_path, changed, message):
    save(tmp_path, candles(20))
    with pytest.raises(ValueError, match=message):
        prepare_testing(tmp_path, selection(**changed))


def test_selected_lookback_rejects_gaps(tmp_path):
    save(tmp_path, candles(20).drop(index=8))
    with pytest.raises(ValueError, match="gaps"):
        prepare_testing(tmp_path, selection())


def test_observed_future_does_not_change_prediction_or_weight_files(tmp_path):
    root = install_tiny(tmp_path)
    before = hashes(root)
    frame = candles(20)
    path = save(tmp_path, frame)
    value, chart = prepare_testing(tmp_path, selection())
    first = forecast_chart(tmp_path, value, chart)
    # Alter every observed candle after the clicked anchor, including its volume.
    frame.loc[10:, ["open", "high", "low", "close", "volume"]] *= 9
    frame.to_csv(path, index=False)
    second_value, second_chart = prepare_testing(tmp_path, selection())
    second = forecast_chart(tmp_path, second_value, second_chart)
    assert chart["history"] == second_chart["history"]
    assert first["actual"] != second["actual"]
    assert first["forecast"] == second["forecast"]
    assert hashes(root) == before


def test_testing_http_job_and_saved_comparison(dashboard):
    server, request, bootstrap, launches, address, workspace = dashboard
    save(workspace, candles(700))
    status, _, _ = request("/api/testing/chart", method="POST", value={"timeframe": "1d"})
    assert status == 403
    status, chart, _ = _post(dashboard, {"timeframe": "1d"}, path="/api/testing/chart")
    assert status == 200 and len(chart["history"]) == 700 and not launches
    status, _, _ = request("/api/testing/forecast", method="POST", value=selection())
    assert status == 403
    root = install_tiny(workspace)
    before = hashes(root)
    status, job, _ = _post(dashboard, selection(), path="/api/testing/forecast")
    assert status == 201 and job["mode"] == "testing" and job["operation"] == "forecast"
    assert job["anchor"] == "2024-01-10T00:00:00+00:00" and len(launches) == 1
    assert run_job(server.manager._path(job["id"])) == 0
    status, result, _ = request("/api/chart/forecasts/" + job["id"])
    assert status == 200 and result["testing"] and len(result["actual"]) == len(result["forecast"]) == 3
    assert hashes(root) == before
    status, script, _ = request("/testing.js")
    assert status == 200 and "/api/testing/forecast" in script
    status, page, _ = request("/")
    assert 'id="testing-tab-button"' in page and 'id="test-lookback"' in page
    assert 'id="test-duration"' in page and 'id="test-duration-unit"' in page
    assert 'id="test-duration-note"' in page and 'id="test-horizon"' not in page

"""Progress reports finished real work; fake only external HTTP bytes."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from threading import Event
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from crypto_forecast import data
from test_exchange_selection import install_markets
from test_multiframe_data import base_candles, install_archive, save_frames
from test_recent_refresh import rest_payload


def phase(events, name):
    return [event for event in events if event["phase"] == name]


def counts(events, name):
    return [(event["completed"], event["total"]) for event in phase(events, name)]


def test_download_reports_checked_archives_and_serialized_timeframes(tmp_path, monkeypatch):
    install_archive(monkeypatch)
    events = []
    report = data.download_multiframe_history(["btc"], "2025-01-01", "2025-01-03", tmp_path,
                                             on_progress=events.append)
    assert report["BTC"]["status"] == "downloaded"
    assert counts(events, "monthly_archives") == [(0, 1), (1, 1)]
    assert counts(events, "daily_archives") == [(0, 0)]
    assert counts(events, "timeframes") == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    assert phase(events, "checking_source") and phase(events, "verifying")
    assert all(event["coin"] == "BTC" for event in events)
    json.dumps(events)  # Cross-thread/UI payloads must contain only primitives.

    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: pytest.fail("cache requested network"))
    cached = []
    data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path,
                                     on_progress=cached.append)
    assert counts(cached, "monthly_archives") == [(0, 1), (1, 1)]
    assert phase(cached, "monthly_archives")[-1]["detail"] == "cached"


def test_parallel_archives_report_completion_before_slow_first_item(tmp_path, monkeypatch):
    release_first, fast_finished = Event(), Event()
    events = []

    def request(url):
        if "2025-01.zip" in url:
            assert release_first.wait(5), "progress waited for the first submitted archive"
        else:
            fast_finished.set()
        raise HTTPError(url, 404, "missing", {}, None)

    def progress(event):
        events.append(event)
        if event["phase"] == "monthly_archives" and event["completed"] == 1:
            assert fast_finished.is_set()
            release_first.set()

    monkeypatch.setattr(data, "_request", request)
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = data._binance_history("BTC", data._utc("2025-01-01"), data._utc("2025-03-01"),
                                       data._utc("2025-04-01"), "15m", tmp_path, False, executor,
                                       on_progress=progress)
    assert counts(events, "monthly_archives") == [(0, 2), (1, 2), (2, 2)]
    assert len(result["missing_archives"]) >= 2


def test_daily_fallback_counts_missing_and_error_responses(tmp_path, monkeypatch):
    def request(url):
        if "2025-03-02.zip" in url:
            raise ValueError("invalid archive")
        raise HTTPError(url, 404, "missing", {}, None)

    monkeypatch.setattr(data, "_request", request)
    events = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = data._binance_history("BTC", data._utc("2025-03-01"), data._utc("2025-03-04"),
                                       data._utc("2025-04-01"), "15m", tmp_path, False, executor,
                                       on_progress=events.append)
    assert counts(events, "monthly_archives") == [(0, 1), (1, 1)]
    assert counts(events, "daily_archives") == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert {event["detail"] for event in phase(events, "daily_archives")[1:]} == {"missing", "error"}
    assert len(result["errors"]) == 1


def test_kucoin_pages_count_empty_and_error_windows(tmp_path, monkeypatch):
    def request(url):
        query = parse_qs(urlparse(url).query)
        if query["startAt"] == ["1735689600"]:
            return b'{"code":"200000","data":[]}'
        raise ValueError("invalid response")

    monkeypatch.setattr(data, "_request", request)
    events = []
    result = data._kucoin_history("BTC", data._utc("2025-01-01"), data._utc("2025-01-18"),
                                 data._utc("2025-02-01"), "15m", tmp_path, on_progress=events.append)
    assert counts(events, "history_pages") == [(0, 2), (1, 2), (2, 2)]
    assert {event["detail"] for event in phase(events, "history_pages")[1:]} == {"checked", "error"}
    assert len(result["historical_rest"]) == 1 and len(result["errors"]) == 1


def test_source_fallback_identifies_exchange_and_uses_page_counts(tmp_path, monkeypatch):
    install_markets(monkeypatch, base_candles().drop(index=5), base_candles())
    events = []
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path,
                                             on_progress=events.append)
    assert report["BTC"]["source"] == "KuCoin spot"
    assert counts(events, "history_pages") == [(0, 1), (1, 1)]
    assert phase(events, "monthly_archives")[0]["source"] == "Binance spot"
    assert phase(events, "history_pages")[0]["source"] == "KuCoin spot"
    assert phase(events, "timeframes")[0]["source"] == "KuCoin spot"


@pytest.mark.parametrize("failure_phase", ["checking_source", "monthly_archives", "history_pages", "verifying", "timeframes"])
def test_callback_errors_propagate_without_becoming_download_errors(tmp_path, monkeypatch, failure_phase):
    install_markets(monkeypatch, base_candles().drop(index=5), base_candles())

    def progress(event):
        if event["phase"] == failure_phase:
            raise ValueError("callback failed")

    with pytest.raises(ValueError, match="callback failed"):
        data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path,
                                         on_progress=progress)
    assert not (tmp_path / "15m/BTC.csv").exists()


def test_recent_reports_pages_and_zero_work_when_current(tmp_path, monkeypatch):
    save_frames(tmp_path)
    monkeypatch.setattr(data, "_request", lambda _: rest_payload(base_candles(16, "2025-01-03")))
    events = []
    result = data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-03T04:00Z",
                                           on_progress=events.append)
    assert result["BTC"]["added_rows"] == 16
    pages = phase(events, "recent_pages")
    assert pages[0]["completed"] == 0 and pages[-1]["completed"] == pages[-1]["total"] == 1
    assert counts(events, "timeframes") == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    current = []
    data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-03T04:00Z", on_progress=current.append)
    assert counts(current, "recent_pages") == [(0, 0)]


def test_recent_invalid_page_is_checked_without_replacing_files(tmp_path, monkeypatch):
    save_frames(tmp_path)
    before = (tmp_path / "15m/BTC.csv").read_bytes()
    monkeypatch.setattr(data, "_request", lambda _: b"[]")
    events = []
    with pytest.raises(ValueError, match="did not cover"):
        data.refresh_multiframe_recent(tmp_path, ["BTC"], as_of="2025-01-03T04:00Z", on_progress=events.append)
    assert phase(events, "recent_pages")[-1]["completed"] == 1
    assert (tmp_path / "15m/BTC.csv").read_bytes() == before

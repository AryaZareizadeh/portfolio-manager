"""Parallel fetching overlaps waits while publication remains coordinated."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import queue
import threading
from urllib.parse import parse_qs, urlparse

import pytest

from crypto_forecast import data, dashboard_worker as worker
from crypto_forecast.dashboard import JobManager, read_json
from test_incremental_download import saved_history, candle_http
from test_multiframe_data import base_candles, save_frames


@pytest.mark.parametrize("source", ["Binance spot", "KuCoin spot"])
def test_missing_pages_overlap_and_report_completed_work(tmp_path, monkeypatch, source):
    full = base_candles()
    saved_history(tmp_path, full.drop(index=[5, 12, 20, 30]), source)
    requested = candle_http(monkeypatch, full, source)
    original_open = data.urlopen
    release, started = threading.Event(), threading.Event()
    events = []
    caller = threading.get_ident()

    def open_url(request, timeout=0):
        query = parse_qs(urlparse(request.full_url).query)
        first = (query.get("startAt") == ["1735694100"] or query.get("startTime") == ["1735694100000"])
        if first:
            started.set()
            assert release.wait(3), "later missing pages did not run while the first request waited"
        return original_open(request, timeout=timeout)

    def progress(event):
        assert threading.get_ident() == caller
        if event["phase"] == "history_pages":
            events.append((event["completed"], event["total"]))
            if event["completed"] == 1:
                assert started.wait(3)
                release.set()

    monkeypatch.setattr(data, "urlopen", open_url)
    try:
        report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path / "stage",
                                                 existing_data_dir=tmp_path, on_progress=progress)["BTC"]
        assert report["rows"] == 192
        assert len(requested) == 4
        assert events == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    finally:
        release.set()


def test_all_http_pools_share_four_request_slots(monkeypatch):
    mutex = threading.Lock()
    release, four_open, excess = threading.Event(), threading.Event(), threading.Event()
    ready = threading.Barrier(8)
    active, peak = 0, 0

    class Response(io.BytesIO):
        def close(self):
            nonlocal active
            if not self.closed:
                with mutex:
                    active -= 1
            super().close()

    def open_url(request, timeout=0):
        nonlocal active, peak
        with mutex:
            active += 1
            peak = max(peak, active)
            if active == 4:
                four_open.set()
            if active > 4:
                excess.set()
        assert release.wait(5)
        return Response(b"ok")

    def request(index):
        ready.wait(timeout=5)
        return data._request(f"https://example.invalid/{index}")

    monkeypatch.setattr(data, "urlopen", open_url)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(request, index) for index in range(8)]
        try:
            assert four_open.wait(5)
            assert not excess.wait(0.2), "more than four HTTP requests entered the exchange at once"
        finally:
            release.set()
        assert [future.result() for future in futures] == [b"ok"] * 8
    assert peak == 4
    assert active == 0


def test_fast_coin_is_saved_while_slow_coin_fetches_without_snapshot_overwrites(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {
        "coins": ["BTC", "ETH"], "start": "2025-01-01", "end": "2025-01-03"}})
    release_slow, slow_progress = threading.Event(), threading.Event()
    commands = queue.Queue()
    exit_codes, published = [], []

    def download(coins, start, end, directory, *, on_progress, **kwargs):
        coin = coins[0]
        on_progress({"coin": coin, "phase": "history_pages", "completed": 0, "total": 1})
        if coin == "BTC":
            assert release_slow.wait(5), "ETH could not finish while BTC was fetching"
            slow_progress.set()
        on_progress({"coin": coin, "phase": "history_pages", "completed": 1, "total": 1})
        save_frames(Path(directory), coin=coin)
        return {coin: {"status": "downloaded", "rows": 192}}

    monkeypatch.setattr(worker, "download_multiframe_history", download)
    thread = threading.Thread(target=lambda: exit_codes.append(worker.run_job(manager._path(job["id"]), commands)))

    class Pipe:
        def write(self, payload):
            command = json.loads(payload)
            coin = command["coin"]
            assert command["published"] is True
            published.append(coin)
            if coin == "ETH":
                assert (tmp_path / "data/15m/ETH.csv").exists()
                release_slow.set()
                assert slow_progress.wait(3)
                # Other fetchers must not overwrite the parent's Saved state
                # while the coordinator is waiting for this acknowledgement.
                assert manager.get_job(job["id"])["progress"]["coin_progress"]["ETH"]["status"] == "saved"
            commands.put(command)
        def flush(self):
            pass
        def close(self):
            pass

    class Process:
        stdin = Pipe()
        def poll(self):
            return exit_codes[0] if exit_codes else None
        def wait(self):
            thread.join(timeout=5)
            assert not thread.is_alive()
            return exit_codes[0]

    try:
        thread.start()
        manager._monitor(job["id"], Process())
        assert exit_codes == [0]
        assert published == ["ETH", "BTC"]
        saved = manager.get_job(job["id"])
        assert saved["status"] == "completed"
        assert saved["progress"]["completed_coins"] == 2
        assert all(row["status"] == "saved" for row in saved["progress"]["coin_progress"].values())
        assert set(read_json(tmp_path / "data/download_report.json")) == {"BTC", "ETH"}
    finally:
        release_slow.set()
        for coin in ("BTC", "ETH"):
            commands.put({"coin": coin, "published": False, "error": "test cleanup"})
        thread.join(timeout=5)
        manager.close()

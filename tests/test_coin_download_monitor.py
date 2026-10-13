"""Per-coin progress persists while work runs, rather than just at coin boundaries."""
import pytest
from pathlib import Path
from test_multiframe_data import save_frames

from crypto_forecast.dashboard import JobManager, read_json
from crypto_forecast import dashboard_worker as worker


def test_worker_persists_real_progress_for_each_coin(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {
        "coins": ["BTC", "ETH"], "start": "2024-01-01", "end": "2024-02-01"}})
    path = manager._path(job["id"])

    def download(coins, *args, on_progress=None, **kwargs):
        assert callable(on_progress)
        coin = coins[0]
        on_progress({"coin": coin, "phase": "monthly_archives", "completed": 1, "total": 3,
                     "source": "Binance spot"})
        saved = read_json(path)["progress"]["coin_progress"]
        assert saved[coin]["completed"] == 1 and saved[coin]["total"] == 3
        assert saved[coin]["status"] == "running"
        on_progress({"coin": coin, "phase": "monthly_archives", "completed": 3, "total": 3})
        assert read_json(path)["progress"]["coin_progress"][coin]["completed"] == 3
        save_frames(Path(args[2]), coin=coin)
        return {coin: {"status": "downloaded", "rows": 192, "source": "Binance spot"}}

    monkeypatch.setattr(worker, "download_multiframe_history", download)
    try:
        assert worker.run_job(path) == 0
        saved = read_json(path)
        assert saved["stage"] == "completed"
        assert set(saved["progress"]["coin_progress"]) == {"BTC", "ETH"}
        assert all(row["status"] == "saved" for row in saved["progress"]["coin_progress"].values())
    finally:
        manager.close()


def test_worker_marks_exception_on_affected_coin(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {
        "coins": ["BTC", "ETH"], "end": "2024-02-01"}})

    def download(coins, *args, on_progress=None, **kwargs):
        on_progress({"coin": coins[0], "phase": "history_pages", "completed": 2, "total": 5})
        raise ValueError("network unavailable")

    monkeypatch.setattr(worker, "download_multiframe_history", download)
    try:
        assert worker.run_job(manager._path(job["id"])) == 1
        saved = manager.get_job(job["id"])
        assert saved["progress"]["coin_progress"]["BTC"]["status"] == "failed"
        assert saved["progress"]["coin_progress"]["BTC"]["error"] == "network unavailable"
        assert saved["progress"]["coin_progress"]["ETH"]["status"] == "failed"
    finally:
        manager.close()


def test_recent_refresh_persists_known_totals_for_each_coin(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    monkeypatch.setattr(worker.time, "monotonic", lambda: 0.0)
    job = manager.start_job({"operation": "download", "config": {"coins": ["BTC", "ETH"]}})
    path = manager._path(job["id"])
    cutoffs = []

    def download(coins, *args, **kwargs):
        save_frames(Path(args[2]), coin=coins[0])
        return {coins[0]: {"status": "downloaded", "rows": 192}}

    def refresh(directory, coins, *, as_of, on_progress):
        assert len(coins) == 1
        coin = coins[0]
        cutoffs.append(as_of)
        on_progress({"coin": coin, "phase": "recent_pages", "completed": 0, "total": 2})
        row = read_json(path)["progress"]["coin_progress"][coin]
        assert row["total"] == 2 and row["completed"] == 0
        on_progress({"coin": coin, "phase": "recent_pages", "completed": 2, "total": 2})
        return {coin: {"status": "refreshed", "added_rows": 20}}

    monkeypatch.setattr(worker, "download_multiframe_history", download)
    monkeypatch.setattr(worker, "refresh_multiframe_recent", refresh)
    try:
        assert worker.run_job(path) == 0
        assert len(cutoffs) == 2 and cutoffs[0] == cutoffs[1]
        assert all(row["status"] == "saved" for row in read_json(path)["progress"]["coin_progress"].values())
    finally:
        manager.close()

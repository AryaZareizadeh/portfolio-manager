"""Saved candle coverage controls network requests and per-coin publication."""
import io
import json
from pathlib import Path
import queue
import threading
from urllib.parse import parse_qs, urlparse
from urllib.error import URLError

import pandas as pd
import pytest

from crypto_forecast import data, dashboard_worker, dashboard
from crypto_forecast.dashboard import JobManager, read_json
from test_multiframe_data import base_candles, save_frames
from test_exchange_selection import install_markets
from test_dashboard import _download_files, _data_snapshot


def saved_history(directory, frame=None, source="Binance spot"):
    frame = base_candles() if frame is None else frame
    save_frames(directory, frame)
    for interval in data.TIMEFRAMES:
        path = directory / interval / "BTC.metadata.json"
        meta = read_json(path)
        meta.update(source=source, symbol="BTC-USDT" if source == "KuCoin spot" else "BTCUSDT",
                    requested_start="2025-01-01T00:00:00+00:00",
                    exchange_comparison={source: {"first_date": "2025-01-01T00:00:00+00:00"}})
        path.write_text(json.dumps(meta), encoding="utf-8")


def candle_http(monkeypatch, frame, source="Binance spot"):
    ranges = []

    def open_url(request, timeout=0):
        query = parse_qs(urlparse(request.full_url).query)
        if source == "Binance spot":
            assert "data-api.binance.vision/api/v3/klines" in request.full_url
            left = pd.Timestamp(int(query["startTime"][0]), unit="ms", tz="UTC")
            right = pd.Timestamp(int(query["endTime"][0]) + 1, unit="ms", tz="UTC")
        else:
            assert "api.kucoin.com/api/v1/market/candles" in request.full_url
            left = pd.Timestamp(int(query["startAt"][0]), unit="s", tz="UTC")
            right = pd.Timestamp(int(query["endAt"][0]) + 1, unit="s", tz="UTC")
        ranges.append((left.isoformat(), right.isoformat()))
        piece = frame.loc[(frame.timestamp >= left) & (frame.timestamp < right)]
        if source == "Binance spot":
            rows = [[int(row.timestamp.timestamp() * 1000), row.open, row.high,
                     row.low, row.close, row.volume, 0, 0, 0, 0, 0, 0] for row in piece.itertuples()]
            payload = rows
        else:
            rows = [[str(int(row.timestamp.timestamp())), str(row.open), str(row.close),
                     str(row.high), str(row.low), str(row.volume), "0"] for row in piece.itertuples()]
            payload = {"code": "200000", "data": rows}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(data, "urlopen", open_url)
    return ranges


def test_complete_saved_range_uses_no_network_including_prelisting_dates(tmp_path, monkeypatch):
    saved_history(tmp_path)
    before = (tmp_path / "15m/BTC.csv").read_bytes()
    ranges = candle_http(monkeypatch, base_candles())
    report = data.download_multiframe_history(["BTC"], "2017-01-01", "2025-01-03", tmp_path / "stage",
                                             existing_data_dir=tmp_path)["BTC"]
    assert ranges == []
    assert report["rows"] == 192
    assert report["reused_rows"] == 192
    assert (tmp_path / "15m/BTC.csv").read_bytes() == before
    assert len(data.load_multiframe_data(tmp_path / "stage", ["BTC"])["BTC"]["1d"]) == 2


@pytest.mark.parametrize("source", ["Binance spot", "KuCoin spot"])
def test_only_missing_internal_slot_and_tail_are_fetched(tmp_path, monkeypatch, source):
    full = base_candles(208)
    saved_history(tmp_path, full.iloc[:192].drop(index=5), source)
    ranges = candle_http(monkeypatch, full, source)
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03T04:00Z", tmp_path / "stage",
                                             existing_data_dir=tmp_path)["BTC"]
    assert sorted(ranges) == [("2025-01-01T01:15:00+00:00", "2025-01-01T01:30:00+00:00"),
                      ("2025-01-03T00:00:00+00:00", "2025-01-03T04:00:00+00:00")]
    assert report["rows"] == 208
    assert report["source"] == source
    frames = data.load_multiframe_data(tmp_path / "stage", ["BTC"])["BTC"]
    assert len(frames["15m"]) == 208
    assert len(frames["1d"]) == 2
    assert frames["15m"].timestamp.is_unique


def test_shorter_requested_range_preserves_newer_saved_candles(tmp_path, monkeypatch):
    saved_history(tmp_path)
    ranges = candle_http(monkeypatch, base_candles())
    report = data.download_multiframe_history(["BTC"], "2025-01-02", "2025-01-03", tmp_path / "stage",
                                             existing_data_dir=tmp_path)["BTC"]
    assert ranges == []
    assert report["rows"] == 192
    assert report["timeframes"]["15m"]["first_date"] == "2025-01-01T00:00:00+00:00"


def test_missing_derived_file_is_rebuilt_without_refetching_base(tmp_path, monkeypatch):
    saved_history(tmp_path)
    (tmp_path / "4h/BTC.csv").unlink()
    ranges = candle_http(monkeypatch, base_candles())
    data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path / "stage",
                                    existing_data_dir=tmp_path)
    assert ranges == []
    assert len(data.load_multiframe_data(tmp_path / "stage", ["BTC"])["BTC"]["4h"]) == 12


def test_corrupted_saved_base_is_not_trusted(tmp_path, monkeypatch):
    saved_history(tmp_path)
    (tmp_path / "15m/BTC.csv").write_text("corrupted", encoding="utf-8")
    ranges = candle_http(monkeypatch, base_candles())
    with pytest.raises(ValueError, match="hash|provenance"):
        data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path / "stage",
                                        existing_data_dir=tmp_path)
    assert ranges == []


def test_missing_prefix_is_fetched_without_discarding_newer_history(tmp_path, monkeypatch):
    full = base_candles(288)
    saved_history(tmp_path, full.iloc[96:])
    ranges = candle_http(monkeypatch, full)
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-03", tmp_path / "stage",
                                             existing_data_dir=tmp_path)["BTC"]
    assert ranges == [("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00")]
    assert report["rows"] == 288
    assert report["timeframes"]["15m"]["last_date"] == "2025-01-03T23:45:00+00:00"


def test_missing_tail_is_paginated_without_requesting_saved_slots(tmp_path, monkeypatch):
    saved_history(tmp_path)
    ranges = candle_http(monkeypatch, base_candles(2304))
    report = data.download_multiframe_history(["BTC"], "2025-01-01", "2025-01-25", tmp_path / "stage",
                                             existing_data_dir=tmp_path)["BTC"]
    assert sorted(ranges) == [("2025-01-03T00:00:00+00:00", "2025-01-13T10:00:00+00:00"),
                      ("2025-01-13T10:00:00+00:00", "2025-01-23T20:00:00+00:00"),
                      ("2025-01-23T20:00:00+00:00", "2025-01-25T00:00:00+00:00")]
    assert report["rows"] == 2304


def test_worker_saves_healthy_coin_when_parallel_coin_fails(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {
        "coins": ["BTC", "ETH"], "start": "2025-01-01", "end": "2025-01-03"}})
    install_markets(monkeypatch, base_candles(), base_candles())
    original_open = data.urlopen

    def open_url(request, timeout=0):
        if "ETH" in request.full_url:
            raise URLError("ETH network unavailable")
        return original_open(request, timeout=timeout)

    monkeypatch.setattr(data, "urlopen", open_url)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    try:
        assert dashboard_worker.run_job(manager._path(job["id"])) == 1
        saved = manager.get_job(job["id"])
        assert saved["progress"]["coin_progress"]["BTC"]["status"] == "saved"
        assert saved["progress"]["coin_progress"]["ETH"]["status"] == "failed"
        assert saved["progress"]["completed_coins"] == 1
        assert len(data.load_multiframe_data(tmp_path / "data", ["BTC"])["BTC"]["15m"]) == 192
    finally:
        manager.close()


def test_per_coin_publication_preserves_unselected_report_entries(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {"coins": ["BTC", "ETH"]}})
    destination = tmp_path / "data"
    _download_files(destination, label="original")
    eth = (destination / "15m/ETH.csv").read_bytes()
    stage = manager.jobs_dir / (job["id"] + "-data")
    _download_files(stage, label="replacement")
    try:
        manager._publish_download(job, coins=["BTC"])
        assert (destination / "15m/ETH.csv").read_bytes() == eth
        report = read_json(destination / "download_report.json")
        assert report["BTC"]["label"] == "replacement"
        assert report["ETH"]["label"] == "original"
    finally:
        manager.close()


def test_parent_and_worker_handshake_saves_each_parallel_coin(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {
        "coins": ["BTC", "ETH"], "start": "2025-01-01", "end": "2025-01-03"}})
    install_markets(monkeypatch, base_candles(), base_candles())
    commands = queue.Queue()
    exit_codes = []
    thread = threading.Thread(target=lambda: exit_codes.append(dashboard_worker.run_job(manager._path(job["id"]), commands)))

    class Pipe:
        def write(self, payload):
            commands.put(json.loads(payload))
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
        saved = manager.get_job(job["id"])
        assert saved["status"] == "completed"
        assert saved["progress"]["completed_coins"] == 2
        assert set(read_json(tmp_path / "data/download_report.json")) == {"BTC", "ETH"}
        assert len(data.load_multiframe_data(tmp_path / "data", ["ETH"])["ETH"]["15m"]) == 192
    finally:
        for coin in ("BTC", "ETH"):
            commands.put({"coin": coin, "published": False, "error": "test cleanup"})
        thread.join(timeout=5)
        manager.close()


def test_stop_keeps_saved_coin_and_prevents_pending_coin_publication(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {"coins": ["BTC", "ETH"]}})
    destination = tmp_path / "data"
    _download_files(destination, label="original")
    stage = manager.jobs_dir / (job["id"] + "-data")
    _download_files(stage, label="replacement")
    progress = {"coin_progress": {coin: {"status": "ready"} for coin in ("BTC", "ETH")}}
    manager._update_job(job["id"], status="running", stage="publishing_coin", publishing_coin="BTC", progress=progress)

    class Process:
        stdin = io.BytesIO()

    process = Process()
    try:
        manager._publish_pending_coin(job["id"], process)
        assert json.loads(process.stdin.getvalue()) == {"coin": "BTC", "published": True}
        assert manager.get_job(job["id"])["progress"]["coin_progress"]["BTC"]["status"] == "saved"
        before = _data_snapshot(destination)
        manager._update_job(job["id"], stage="publishing_coin", publishing_coin="ETH")
        manager.stop_job(job["id"])
        manager._publish_pending_coin(job["id"], process)
        assert manager.get_job(job["id"])["status"] == "stopped"
        assert _data_snapshot(destination) == before
        assert read_json(destination / "download_report.json")["ETH"]["label"] == "original"
    finally:
        manager.close()


def test_parent_command_reader_accepts_publication_acknowledgements(monkeypatch):
    commands = queue.Queue()
    chunks = iter((b'{"coin":"BTC","published":true}\n{"coin":"ETH","published":false,"error":"disk full"}\n', b""))
    monkeypatch.setattr(dashboard_worker, "_read_parent_block", lambda fd, size: next(chunks))
    def parent_lost(code):
        raise SystemExit(code)
    monkeypatch.setattr(dashboard_worker.os, "_exit", parent_lost)
    with pytest.raises(SystemExit):
        dashboard_worker.read_parent_commands(commands, fd=123)
    assert commands.get_nowait() == {"coin": "BTC", "published": True}
    assert commands.get_nowait() == {"coin": "ETH", "published": False, "error": "disk full"}
    assert commands.empty()


def test_publication_monitor_can_read_progress_during_windows_snapshot_replacement(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text('{"status":"running"}', encoding="utf-8")
    original = dashboard._open_snapshot
    attempts = []
    def snapshot(filename, flags):
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError("temporary Windows sharing conflict")
            error.winerror = 32
            raise error
        return original(filename, flags)
    monkeypatch.setattr(dashboard, "_open_snapshot", snapshot)
    assert read_json(path) == {"status": "running"}
    assert len(attempts) == 3

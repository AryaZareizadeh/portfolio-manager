"""Snapshot read races must not become false 404s or orphan waiting workers."""
import io
from pathlib import Path

import pytest

from crypto_forecast import dashboard
from crypto_forecast.dashboard import JobManager


@pytest.mark.parametrize("failure", ["missing", "invalid_json"])
def test_get_job_retries_transient_missing_or_invalid_snapshot(tmp_path, monkeypatch, failure):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {"coins": ["BTC"]}})
    path = manager._path(job["id"])
    invalid = tmp_path / "incomplete.json"
    invalid.write_text('{"status":', encoding="utf-8")
    original = dashboard._open_snapshot
    attempts = []

    def snapshot(filename, flags):
        if Path(filename) == path:
            attempts.append(1)
            if len(attempts) < 3:
                if failure == "missing":
                    raise FileNotFoundError("snapshot temporarily unavailable during replacement")
                return original(str(invalid), flags)
        return original(filename, flags)

    monkeypatch.setattr(dashboard, "_open_snapshot", snapshot)
    try:
        assert manager.get_job(job["id"])["status"] == "queued"
        assert len(attempts) == 3
    finally:
        monkeypatch.setattr(dashboard, "_open_snapshot", original)
        manager.close()


def test_monitor_closes_worker_pipe_when_job_is_permanently_missing(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {"coins": ["BTC"]}})

    class Process:
        stdin = io.BytesIO()
        returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
        def wait(self, timeout=None):
            return self.returncode

    process = Process()
    manager.processes[job["id"]] = process
    manager._path(job["id"]).unlink()
    try:
        manager._monitor(job["id"], process)
        assert process.stdin.closed
        assert job["id"] not in manager.processes
        assert process.poll() is not None
    finally:
        manager.processes.pop(job["id"], None)
        manager.close()


def test_stop_during_monitor_read_failure_stays_stopped(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda _: None)
    job = manager.start_job({"operation": "download", "config": {"coins": ["BTC"]}})

    class Process:
        stdin = io.BytesIO()
        returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
        def wait(self, timeout=None):
            return self.returncode

    process = Process()
    manager.processes[job["id"]] = process
    original = manager.get_job
    failures = []

    def snapshot(identifier):
        if not failures:
            failures.append(True)
            # Stop wins the lock between a read failure and monitor cleanup.
            manager.stop_job(identifier)
            raise dashboard.DashboardError(404, "Job not found.")
        return original(identifier)

    monkeypatch.setattr(manager, "get_job", snapshot)
    try:
        manager._monitor(job["id"], process)
        stopped = original(job["id"])
        assert stopped["status"] == "stopped"
        assert stopped["error"] is None
        assert process.stdin.closed
        assert job["id"] not in manager.processes
    finally:
        manager.close()

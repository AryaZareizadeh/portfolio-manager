"""Local dashboard HTTP and job-control boundaries; no market download or training."""

import hashlib
import io
import json
import os
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from crypto_forecast.dashboard import DashboardError, JobManager, create_server, read_json


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    server = create_server(tmp_path, port=0)
    launches = []
    monkeypatch.setattr(server.manager, "_launch", lambda job: launches.append(job))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = "http://127.0.0.1:" + str(server.server_address[1])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, *, method="GET", value=None, raw=None, headers=None):
        if raw is None and value is not None:
            raw = json.dumps(value).encode()
        supplied = dict(headers or {})
        if raw is not None:
            supplied.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(address + path, data=raw, method=method, headers=supplied)
        try:
            response = opener.open(req, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
            value = json.loads(body) if "application/json" in content_type else body.decode()
            return response.status, value, response.headers

    try:
        status, bootstrap, _ = request("/api/bootstrap")
        assert status == 200
        yield server, request, bootstrap, launches, address, tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _payload(**config):
    return {"operation": "train", "name": "Browser test", "config": {"coins": ["BTC", "ETH"], **config}}


def _post(dashboard, payload=None, path="/api/jobs", **kwargs):
    _, request, bootstrap, _, address, _ = dashboard
    headers = {"X-CSRF-Token": bootstrap["csrf_token"], "Origin": address}
    headers.update(kwargs.pop("headers", {}))
    return request(path, method="POST", value=payload, headers=headers, **kwargs)


def test_bootstrap_and_static_assets(dashboard):
    _, request, bootstrap, _, _, _ = dashboard
    assert {"csrf_token", "defaults", "data", "jobs", "runs"} <= bootstrap.keys()
    assert bootstrap["csrf_token"]
    assert bootstrap["defaults"]["input_mode"] == "multiframe"
    assert bootstrap["defaults"]["origin_interval"] == "4h"
    status, page, headers = request("/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert "<script" in page
    assert "Train" in page or "train" in page
    for path, mime in (("/app.js", "javascript"), ("/styles.css", "text/css")):
        status, body, headers = request(path)
        assert status == 200 and body
        assert mime in headers["Content-Type"]


def test_job_settings_are_confined_and_job_is_queryable(dashboard):
    server, request, _, launches, _, workspace = dashboard
    status, job, _ = _post(dashboard, _payload(epochs=3, device="cpu"))
    assert status == 201
    assert len(launches) == 1
    assert job["operation"] == "train"
    assert job["config"]["input_mode"] == "multiframe"
    assert job["config"]["origin_interval"] == "4h"
    assert Path(job["config"]["data_dir"]) == workspace / "data"
    run_dir = Path(job["config"]["run_dir"])
    assert run_dir.parent == workspace / "runs"
    assert run_dir.name
    assert job["config"]["epochs"] == 3
    assert server.manager.get_job(job["id"])["id"] == job["id"]
    status, fetched, _ = request("/api/jobs/" + job["id"])
    assert status == 200 and fetched["id"] == job["id"]
    status, listing, _ = request("/api/jobs")
    assert status == 200
    assert any(item["id"] == job["id"] for item in listing["jobs"])
    assert listing["active_job"]


def test_download_ignores_training_settings_and_tokenizer_selection(dashboard):
    _, _, _, launches, _, _ = dashboard
    status, job, _ = _post(dashboard, {"operation": "download", "config": {
        "coins": ["BTC", "ETH"], "start": "2024-01-01", "end": "2024-02-01",
        "epochs": 9000, "lookback": 9000, "batch_size": 9000,
        "train_fraction": 0.9, "validation_fraction": 0.9, "model_type": "kronos-small",
        "kronos_tokenizer_run_id": "unavailable", "kronos_tokenizer_batch_size": 9000}})
    assert status == 201
    assert len(launches) == 1
    assert job["operation"] == "download" and job["config"]["coins"] == ["BTC", "ETH"]
    assert job["config"]["start"] == "2024-01-01" and job["config"]["end"] == "2024-02-01"
    assert job["config"]["kronos_tokenizer_run_dir"] is None


@pytest.mark.parametrize("operation", ["download", "train"])
def test_coin_limit_error_reports_actual_count_and_limit(dashboard, operation):
    status, body, _ = _post(dashboard, {"operation": operation,
        "config": {"coins": [f"COIN{i}" for i in range(101)]}})
    assert status == 400
    assert "101" in body["error"] and "100" in body["error"] and "coin" in body["error"]


@pytest.mark.parametrize("operation, stage", [("download", "tokenizer"), ("train", "tokenizer"),
                                             ("train", "predictor")])
def test_dashboard_accepts_one_hundred_coins(dashboard, operation, stage):
    status, job, _ = _post(dashboard, {"operation": operation, "config": {
        "coins": [f"COIN{i}" for i in range(100)], "model_type": "kronos-small",
        "kronos_training_stage": stage}})
    assert status == 201
    assert len(job["config"]["coins"]) == 100


def test_training_limit_error_identifies_offending_setting(dashboard):
    status, body, _ = _post(dashboard, _payload(batch_size=9000))
    assert status == 400
    assert "batch_size" in body["error"] and "9000" in body["error"] and "8192" in body["error"]


def test_only_one_job_can_run_and_stop_is_idempotent(dashboard):
    _, _, _, launches, _, _ = dashboard
    status, first, _ = _post(dashboard, _payload())
    assert status == 201
    status, _, _ = _post(dashboard, _payload())
    assert status == 409
    assert len(launches) == 1
    status, stopped, _ = _post(dashboard, path="/api/jobs/" + first["id"] + "/stop", payload={})
    assert status == 200
    assert stopped["status"] == "stopped"
    status, again, _ = _post(dashboard, path="/api/jobs/" + first["id"] + "/stop", payload={})
    assert status == 200
    assert again["status"] == "stopped"


@pytest.mark.parametrize("config", [
    {"coins": []}, {"coins": None}, {"coins": "BTC"}, {"coins": [None]},
    {"coins": ["BTC", "btc"]}, {"coins": ["../BTC"]}, {"coins": ["USDT"]},
    {"epochs": 0}, {"epochs": True}, {"epochs": "10"}, {"batch_size": -1},
    {"lookback": 0}, {"learning_rate": 0}, {"learning_rate": None},
    {"learning_rate": "fast"}, {"learning_rate": float("nan")},
    {"learning_rate": float("inf")}, {"learning_rate": True}, {"device": "shell"},
    {"start": "2026-02-30"}, {"start": "2026-10-10", "end": "2020-01-01"},
    {"train_fraction": 1.0}, {"validation_fraction": 0.9}, {"train_fraction": None},
    {"validation_fraction": float("nan")},
    {"data_dir": "C:/outside/data"}, {"run_dir": "C:/outside/run"},
    {"unexpected_setting": "value"},
])
def test_invalid_settings_do_not_launch_worker(dashboard, config):
    _, _, _, launches, _, _ = dashboard
    status, body, _ = _post(dashboard, _payload(**config))
    assert status == 400
    assert "error" in body
    assert launches == []


@pytest.mark.parametrize("payload", [
    [], None, {"operation": "execute", "config": {"coins": ["BTC"]}},
    {"operation": "train", "config": []},
])
def test_bad_job_payloads_do_not_launch_worker(dashboard, payload):
    _, _, _, launches, _, _ = dashboard
    status, body, _ = _post(dashboard, raw=json.dumps(payload).encode())
    assert status == 400
    assert "error" in body
    assert launches == []


def test_malformed_json_and_unknown_jobs(dashboard):
    _, request, _, launches, _, _ = dashboard
    status, _, _ = _post(dashboard, raw=b'{"config":')
    assert status == 400
    assert launches == []
    assert request("/api/jobs/unknown")[0] == 404
    assert _post(dashboard, {}, path="/api/jobs/unknown/stop")[0] == 404


@pytest.mark.parametrize("headers", [
    {"X-CSRF-Token": "wrong"}, {"Origin": "https://attacker.invalid"},
    {"Host": "attacker.invalid"},
])
def test_mutations_require_local_origin_host_and_csrf(dashboard, headers):
    _, _, _, launches, _, _ = dashboard
    assert _post(dashboard, _payload(), headers=headers)[0] == 403
    assert launches == []


def test_mutation_without_csrf_is_denied(dashboard):
    _, request, _, launches, address, _ = dashboard
    status, _, _ = request("/api/jobs", method="POST", value=_payload(), headers={"Origin": address})
    assert status == 403
    assert launches == []


def test_reads_reject_foreign_host(dashboard):
    _, request, _, _, _, _ = dashboard
    assert request("/api/bootstrap", headers={"Host": "attacker.invalid"})[0] == 403


def test_concurrent_start_requests_launch_exactly_one_job(dashboard):
    _, _, _, launches, _, _ = dashboard
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _post(dashboard, _payload()), range(2)))
    assert sorted(result[0] for result in results) == [201, 409]
    assert len(launches) == 1


def test_process_launch_failure_is_saved_and_allows_retry(dashboard, monkeypatch):
    server, request, _, _, _, _ = dashboard

    def fail_launch(job):
        raise OSError("Test launch unavailable")

    monkeypatch.setattr(server.manager, "_launch", fail_launch)
    assert _post(dashboard, _payload())[0] == 500
    status, listing, _ = request("/api/jobs")
    assert status == 200 and listing["active_job"] is None
    assert listing["jobs"][0]["status"] == "failed"
    assert "Test launch unavailable" in listing["jobs"][0]["error"]
    monkeypatch.setattr(server.manager, "_launch", lambda job: None)
    assert _post(dashboard, _payload())[0] == 201


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com"])
def test_server_refuses_non_loopback_bindings(tmp_path, host):
    with pytest.raises(ValueError, match="(?i)loopback|local"):
        create_server(tmp_path, host=host, port=0)


def _saved_run(workspace, name="saved-run"):
    run = workspace / "runs" / name
    run.mkdir(parents=True)
    summary = {"created_at": "2026-10-10T12:00:00+00:00", "data_kind": "historical",
               "best_epoch": 2, "config": {"coins": ["BTC"], "input_mode": "multiframe",
               "origin_interval": "4h"}, "history": [{"epoch": 1, "train_loss": 0.4,
               "validation_loss": 0.7}], "metrics": {}}
    forecasts = [{"coin": "BTC", "horizon_days": 1, "forecast_price_usdt": 12.5}]
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "forecasts.json").write_text(json.dumps(forecasts), encoding="utf-8")
    (run / "report.html").write_text("<!doctype html><p>Saved report</p>", encoding="utf-8")
    (run / "private.csv").write_text("private data", encoding="utf-8")
    return run, summary, forecasts


def test_saved_runs_return_forecasts_report_and_confined_artifacts(dashboard):
    _, request, _, _, _, workspace = dashboard
    run, summary, forecasts = _saved_run(workspace)
    status, listing, _ = request("/api/runs")
    assert status == 200
    assert any(item["id"] == run.name for item in listing["runs"])
    status, detail, _ = request("/api/runs/" + run.name)
    assert status == 200
    assert detail["summary"] == summary
    assert detail["forecasts"] == forecasts
    status, report, headers = request(detail["report_url"])
    assert status == 200 and "Saved report" in report
    assert "text/html" in headers["Content-Type"]
    assert request("/runs/" + run.name + "/summary.json")[1] == summary
    assert request("/runs/" + run.name + "/forecasts.json")[1] == forecasts
    (run / "model.pt").write_bytes(b"private model checkpoint")
    assert request("/runs/" + run.name + "/model.pt")[0] == 404
    assert request("/runs/" + run.name + "/private.csv")[0] == 404
    assert request("/data/BTC.csv")[0] == 404


def test_full_report_artifact_links_download_the_saved_files(dashboard):
    from crypto_forecast.report import write_report

    _, request, _, _, address, workspace = dashboard
    run, summary, forecasts = _saved_run(workspace)
    csv_names = ("forecasts.csv", "validation_predictions.csv", "test_predictions.csv")
    for filename in csv_names:
        (run / filename).write_text("coin,value\nBTC,12.5\n", encoding="utf-8")
    write_report(run, summary, forecasts)
    report_path = "/runs/" + run.name + "/report"
    status, report, _ = request(report_path)
    assert status == 200
    for filename in ("summary.json", "forecasts.json", *csv_names):
        assert 'href="' + filename + '"' in report
        linked = urllib.parse.urljoin(address + report_path, filename)
        status, body, headers = request(urllib.parse.urlsplit(linked).path)
        assert status == 200
        if filename.endswith(".json"):
            assert body == json.loads((run / filename).read_text(encoding="utf-8"))
            assert "application/json" in headers["Content-Type"]
        else:
            assert body == (run / filename).read_bytes().decode("utf-8")
            assert "text/csv" in headers["Content-Type"]


@pytest.mark.parametrize("path", [
    "/api/runs/..", "/api/runs/%2e%2e", "/runs/%2e%2e/report",
    "/runs/saved-run/../report", "/runs/saved-run/%2e%2e/report",
    "/api/runs/%2e%2e%2fsecret", "/api/runs/%2e%2e%5csecret",
])
def test_run_routes_reject_traversal(dashboard, path):
    _, request, _, _, _, workspace = dashboard
    _saved_run(workspace)
    (workspace / "secret").write_text("must not expose", encoding="utf-8")
    status, body, _ = request(path)
    assert status in {400, 404}
    assert "must not expose" not in str(body)


def test_jobs_are_persisted_and_unfinished_jobs_become_interrupted(tmp_path, monkeypatch):
    first = create_server(tmp_path, port=0)
    monkeypatch.setattr(first.manager, "_launch", lambda job: None)
    job = first.manager.start_job(_payload())
    saved = [path for path in (tmp_path / ".dashboard").rglob("*.json")
             if json.loads(path.read_text(encoding="utf-8")).get("id") == job["id"]]
    assert len(saved) == 1
    first.manager.close()
    first.server_close()
    persisted = json.loads(saved[0].read_text(encoding="utf-8"))
    persisted["status"] = "running"
    saved[0].write_text(json.dumps(persisted), encoding="utf-8")
    second = create_server(tmp_path, port=0)
    try:
        restored = second.manager.get_job(job["id"])
        assert restored["status"] == "interrupted"
        assert any(item["id"] == job["id"] for item in second.manager.list_jobs())
    finally:
        second.manager.close()
        second.server_close()


def test_existing_runs_are_never_overwritten(dashboard):
    _, _, _, launches, _, workspace = dashboard
    run, _, _ = _saved_run(workspace)
    original = (run / "report.html").read_bytes()
    status, job, _ = _post(dashboard, {**_payload(), "name": run.name})
    assert status == 201
    assert len(launches) == 1
    assert Path(job["config"]["run_dir"]) != run
    assert (run / "report.html").read_bytes() == original


@pytest.mark.parametrize("filename", ["summary.json", "forecasts.json", "report.html", "forecasts.csv",
                                      "validation_predictions.csv", "test_predictions.csv"])
def test_run_artifact_symlinks_cannot_escape_workspace(dashboard, filename, monkeypatch):
    _, request, _, _, _, workspace = dashboard
    run, summary, forecasts = _saved_run(workspace)
    target = workspace.parent / ("outside-" + filename)
    content = json.dumps(summary if filename == "summary.json" else forecasts) if filename.endswith(".json") else "outside report"
    target.write_text(content, encoding="utf-8")
    artifact = run / filename
    artifact.unlink(missing_ok=True)
    try:
        artifact.symlink_to(target)
    except OSError:
        # Windows may deny symlink creation. Model the same resolved path boundary
        # while still exercising the actual HTTP route and artifact checks.
        artifact.write_text(content, encoding="utf-8")
        resolve = Path.resolve

        def redirected(path, *args, **kwargs):
            return target if path == artifact else resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", redirected)
    path = "/runs/saved-run/report" if filename == "report.html" else "/runs/saved-run/" + filename
    assert request(path)[0] == 404
    if filename in {"summary.json", "forecasts.json"}:
        assert request("/api/runs/saved-run")[0] == 404


def test_stop_terminates_and_kills_unresponsive_worker(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job(_payload())

    class Process:
        terminated = killed = waits = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated += 1

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("test worker", timeout)
            return 1

        def kill(self):
            self.killed += 1

    process = Process()
    manager.processes[job["id"]] = process
    stopped = manager.stop_job(job["id"])
    assert stopped["status"] == "stopped"
    assert (process.terminated, process.killed, process.waits) == (1, 1, 2)
    assert manager.stop_job(job["id"])["status"] == "stopped"
    assert process.terminated == 1
    manager.close()


@pytest.mark.parametrize("status, code", [("running", 9), ("completed", 0), ("stopped", 1)])
def test_monitor_marks_unexpected_worker_exit_without_overwriting_terminal_state(tmp_path, monkeypatch, status, code):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job(_payload())
    manager._update_job(job["id"], status=status)
    (manager.jobs_dir / (job["id"] + ".log")).write_text("worker diagnostic", encoding="utf-8")

    class Process:
        def wait(self):
            return code

    manager._monitor(job["id"], Process())
    updated = manager.get_job(job["id"])
    assert updated["status"] == ("failed" if status == "running" else status)
    if status == "running":
        assert "Worker exited (9)" in updated["error"]
        assert "worker diagnostic" in updated["error"]
    manager.close()


def test_worker_saves_training_events_and_completion(tmp_path, monkeypatch):
    from crypto_forecast import dashboard_worker

    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job(_payload(epochs=2))

    def train(config, *, progress, on_event):
        assert config.input_mode == "multiframe" and config.origin_interval == "4h"
        progress("Prepared closed candles")
        on_event({"kind": "stage", "stage": "training", "train_samples": 123})
        for epoch in (1, 2):
            on_event({"kind": "epoch", "stage": "training", "epoch": epoch,
                      "train_loss": 0.5 / epoch, "validation_loss": 0.8 / epoch})
        return {"best_epoch": 2}

    monkeypatch.setattr(dashboard_worker, "train", train)
    assert dashboard_worker.run_job(manager.jobs_dir / (job["id"] + ".json")) == 0
    completed = manager.get_job(job["id"])
    assert completed["status"] == "completed"
    assert completed["finished_at"]
    assert [point["epoch"] for point in completed["history"]] == [1, 2]
    assert completed["progress"]["train_samples"] == 123
    assert completed["progress"]["best_epoch"] == 2
    assert any("Prepared closed candles" in line["message"] for line in completed["logs"])
    manager.close()


def test_worker_saves_training_failure_and_error(tmp_path, monkeypatch):
    from crypto_forecast import dashboard_worker

    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job(_payload())

    def train(*args, **kwargs):
        raise ValueError("Missing completed volume history")

    monkeypatch.setattr(dashboard_worker, "train", train)
    assert dashboard_worker.run_job(manager.jobs_dir / (job["id"] + ".json")) == 1
    failed = manager.get_job(job["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "Missing completed volume history"
    assert failed["finished_at"]
    manager.close()


def test_worker_download_stages_each_coin_and_publishes_its_files(tmp_path, monkeypatch):
    from crypto_forecast import dashboard_worker

    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job({**_payload(end="2026-10-10"), "operation": "download"})
    requests = []

    def download(coins, start, end, data_dir, *, archive_root, existing_data_dir, on_progress=None):
        from test_multiframe_data import save_frames
        assert Path(existing_data_dir) == tmp_path / "data"
        requests.append((coins, start, end, data_dir, archive_root))
        save_frames(Path(data_dir), coin=coins[0])
        return {coins[0]: {"status": "downloaded", "rows": 192}}

    monkeypatch.setattr(dashboard_worker, "download_multiframe_history", download)
    assert dashboard_worker.run_job(manager.jobs_dir / (job["id"] + ".json")) == 0
    assert sorted(item[0] for item in requests) == [["BTC"], ["ETH"]]
    waiting = manager.get_job(job["id"])
    assert waiting["status"] == "completed" and waiting["stage"] == "completed"
    assert waiting["finished_at"] is not None
    assert waiting["progress"]["completed_coins"] == 2
    staging = manager.jobs_dir / (job["id"] + "-data")
    assert all(Path(item[3]) == staging for item in requests)
    assert all(Path(item[4]) == tmp_path / "data" / "_archives" for item in requests)
    report = json.loads((staging / "download_report.json").read_text(encoding="utf-8"))
    assert set(report) == {"BTC", "ETH"}
    assert set(read_json(tmp_path / "data" / "download_report.json")) == {"BTC", "ETH"}
    manager.close()


def _download_files(directory, *, coins=("BTC", "ETH"), label="downloaded"):
    directory = Path(directory)
    for interval in ("15m", "1h", "4h", "1d"):
        branch = directory / interval
        branch.mkdir(parents=True, exist_ok=True)
        for coin in coins:
            close = sum((label + interval + coin).encode())
            payload = f"timestamp,open,high,low,close,volume\n2026-01-01T00:00:00Z,1,{close},1,{close},3\n".encode()
            (branch / f"{coin}.csv").write_bytes(payload)
            (branch / f"{coin}.metadata.json").write_text(
                json.dumps({"coin": coin, "interval": interval,
                            "data_sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8")
    (directory / "download_report.json").write_text(
        json.dumps({coin: {"status": "downloaded", "label": label} for coin in coins}), encoding="utf-8")


def _data_snapshot(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


class _ExitedProcess:
    def __init__(self, code):
        self.code = code

    def wait(self):
        return self.code


def test_monitor_publishes_complete_download_after_successful_worker_exit(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job({**_payload(end="2026-10-10"), "operation": "download"})
    destination = tmp_path / "data"
    _download_files(destination, label="original")
    staging = manager.jobs_dir / (job["id"] + "-data")
    _download_files(staging, label="replacement")
    manager._update_job(job["id"], status="running", stage="publishing")
    manager._monitor(job["id"], _ExitedProcess(0))
    published = manager.get_job(job["id"])
    assert published["status"] == "completed" and published["stage"] == "completed"
    assert published["finished_at"]
    for relative, payload in _data_snapshot(destination).items():
        assert payload == (staging / relative).read_bytes()
    assert json.loads((destination / "download_report.json").read_text())["BTC"]["label"] == "replacement"
    assert any("Candle history is ready" in line["message"] for line in published["logs"])
    manager.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_or_cancelled_download_never_replaces_original_data(tmp_path, monkeypatch, cancelled):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job({**_payload(end="2026-10-10"), "operation": "download"})
    destination = tmp_path / "data"
    _download_files(destination, label="original")
    before = _data_snapshot(destination)
    staging = manager.jobs_dir / (job["id"] + "-data")
    _download_files(staging, label="replacement")
    manager._update_job(job["id"], status="running", stage="publishing")
    if cancelled:
        manager.stop_job(job["id"])
    manager._monitor(job["id"], _ExitedProcess(0 if cancelled else 1))
    assert manager.get_job(job["id"])["status"] == ("stopped" if cancelled else "failed")
    assert _data_snapshot(destination) == before
    manager.close()


def test_download_publication_error_rolls_back_existing_and_new_files(tmp_path, monkeypatch):
    from crypto_forecast import dashboard as dashboard_module

    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job({**_payload(end="2026-10-10"), "operation": "download"})
    destination = tmp_path / "data"
    _download_files(destination, coins=("BTC",), label="original")
    before = _data_snapshot(destination)
    staging = manager.jobs_dir / (job["id"] + "-data")
    _download_files(staging, label="replacement")
    manager._update_job(job["id"], status="running", stage="publishing")
    write = dashboard_module.write_bytes
    failed = []

    def fail_one_publication(path, payload):
        if Path(path) == destination / "1h" / "BTC.metadata.json" and not failed:
            failed.append(True)
            # Earlier writes have replaced an existing pair and created a new
            # ETH pair, so both restoration and removal are exercised.
            assert (destination / "15m" / "BTC.csv").read_bytes() != before[str(Path("15m") / "BTC.csv")]
            assert (destination / "15m" / "ETH.csv").exists()
            raise OSError("Injected publication write failure")
        write(path, payload)

    monkeypatch.setattr(dashboard_module, "write_bytes", fail_one_publication)
    manager._monitor(job["id"], _ExitedProcess(0))
    failed_job = manager.get_job(job["id"])
    assert failed_job["status"] == "failed"
    assert "Injected publication write failure" in failed_job["error"]
    assert failed == [True]
    assert _data_snapshot(destination) == before
    manager.close()


@pytest.mark.parametrize("corruption", ["missing", "hash_mismatch"])
def test_download_publication_rejects_incomplete_or_mismatched_staging(tmp_path, monkeypatch, corruption):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job({**_payload(end="2026-10-10"), "operation": "download"})
    destination = tmp_path / "data"
    _download_files(destination, label="original")
    before = _data_snapshot(destination)
    staging = manager.jobs_dir / (job["id"] + "-data")
    _download_files(staging, label="replacement")
    if corruption == "missing":
        (staging / "4h" / "ETH.csv").unlink()
    else:
        (staging / "4h" / "ETH.csv").write_text("corrupted staged payload", encoding="utf-8")
    manager._update_job(job["id"], status="running", stage="publishing")
    manager._monitor(job["id"], _ExitedProcess(0))
    failed_job = manager.get_job(job["id"])
    assert failed_job["status"] == "failed"
    if corruption == "hash_mismatch":
        assert "does not match its metadata" in failed_job["error"]
    assert _data_snapshot(destination) == before
    manager.close()


def test_closing_manager_stops_jobs_and_prevents_new_work(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    launched = []
    monkeypatch.setattr(manager, "_launch", lambda job: launched.append(job))
    job = manager.start_job(_payload())
    manager.close()
    assert manager.get_job(job["id"])["status"] == "stopped"
    with pytest.raises(DashboardError) as error:
        manager.start_job(_payload())
    assert error.value.status == 409
    assert len(launched) == 1


def test_failed_download_is_not_marked_ready(tmp_path, monkeypatch):
    from crypto_forecast import dashboard_worker

    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job({**_payload(end="2026-10-10"), "operation": "download"})
    def download(coins, *args, **kwargs):
        from test_multiframe_data import save_frames
        if coins[0] == "ETH":
            return {"ETH": {"status": "failed", "rows": 0}}
        save_frames(Path(args[2]))
        return {"BTC": {"status": "downloaded", "rows": 192}}
    monkeypatch.setattr(dashboard_worker, "download_multiframe_history", download)
    assert dashboard_worker.run_job(manager.jobs_dir / (job["id"] + ".json")) == 1
    failed = manager.get_job(job["id"])
    assert failed["status"] == "failed"
    assert "History download failed for ETH" in failed["error"]
    manager.close()


def test_one_dashboard_owns_each_workspace(tmp_path):
    first = create_server(tmp_path, port=0)
    second = None
    try:
        with pytest.raises(ValueError, match="(?i)workspace|already|dashboard|project|running"):
            second = create_server(tmp_path, port=0)
    finally:
        if second is not None:
            second.server_close()
        first.server_close()
    # A graceful shutdown releases ownership and permits a new dashboard.
    replacement = create_server(tmp_path, port=0)
    replacement.server_close()


@pytest.mark.parametrize("job_status", ["queued", "running", "stopped", "failed"])
def test_unfinished_or_cancelled_training_artifacts_are_not_results(dashboard, job_status):
    server, request, _, _, _, workspace = dashboard
    status, job, _ = _post(dashboard, _payload())
    assert status == 201
    _saved_run(workspace, job["id"])
    server.manager._update_job(job["id"], status=job_status)
    status, listing, _ = request("/api/runs")
    assert status == 200
    assert all(item["id"] != job["id"] for item in listing["runs"])
    assert request("/api/runs/" + job["id"])[0] == 404
    assert request("/runs/" + job["id"] + "/report")[0] == 404


def test_completed_training_artifacts_are_visible(dashboard):
    server, request, _, _, _, workspace = dashboard
    status, job, _ = _post(dashboard, _payload())
    assert status == 201
    _saved_run(workspace, job["id"])
    server.manager._update_job(job["id"], status="completed")
    status, listing, _ = request("/api/runs")
    assert status == 200
    assert any(item["id"] == job["id"] for item in listing["runs"])
    assert request("/api/runs/" + job["id"])[0] == 200
    assert request("/runs/" + job["id"] + "/report")[0] == 200


def test_progress_snapshot_retries_windows_sharing_conflicts(tmp_path, monkeypatch):
    from crypto_forecast import dashboard
    path = tmp_path / "job.json"
    path.write_text('{"status":"running"}', encoding="utf8")
    original_replace = dashboard._replace_snapshot
    attempts = []
    def replace(source, target):
        attempts.append(1)
        if len(attempts) < 3:
            assert json.loads(path.read_text())["status"] == "running"
            raise PermissionError("Windows reader temporarily holds the progress file")
        original_replace(source, target)
    monkeypatch.setattr(dashboard, "_replace_snapshot", replace)
    monkeypatch.setattr(dashboard.time, "sleep", lambda _: None)
    dashboard.write_json(path, {"status": "completed"})
    assert len(attempts) == 3
    assert json.loads(path.read_text())["status"] == "completed"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_dashboard_poll_allows_worker_to_replace_open_snapshot(tmp_path, monkeypatch):
    from crypto_forecast import dashboard

    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job(_payload())
    path = manager._path(job["id"])
    original_open = io.open
    replacements = []

    @contextmanager
    def open_during_update(file, *args, **kwargs):
        with original_open(file, *args, **kwargs) as snapshot:
            if Path(file) == path and not replacements:
                replacements.append(True)
                # The worker must publish while a dashboard poll still holds
                # the previous snapshot, even for longer than the retry budget.
                dashboard.write_json(path, {**job, "status": "completed"})
            yield snapshot

    monkeypatch.setattr(io, "open", open_during_update)
    monkeypatch.setattr(dashboard.time, "sleep", lambda _: None)
    try:
        assert manager.get_job(job["id"])["status"] == "queued"
        assert manager.get_job(job["id"])["status"] == "completed"
        assert list(manager.jobs_dir.glob("*.tmp")) == []
    finally:
        manager.close()

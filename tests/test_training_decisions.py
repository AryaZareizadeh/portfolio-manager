"""Early-stop choices preserve real training state and require an explicit owner choice."""

import io
import json
import math
import queue
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_forecast import pipeline
from crypto_forecast.dashboard import DashboardError, JobManager, create_server
from crypto_forecast.pipeline import TrainingConfig, predict, train


def _daily_config(tmp_path, **changes):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index = np.arange(500)
    close = 100 * np.exp(np.cumsum(0.001 + 0.004 * np.sin(index / 11)))
    pd.DataFrame({
        "timestamp": pd.date_range(end="2025-09-30", periods=len(index), tz="UTC"),
        "open": close * 0.995, "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": 1000 + 100 * np.sin(index / 9),
    }).to_csv(data_dir / "BTC.csv", index=False)
    return TrainingConfig(coins=["BTC"], input_mode="daily", start="2020-01-01",
                          end="2025-10-01", data_dir=str(data_dir),
                          run_dir=str(tmp_path / "run"), lookback=5,
                          hidden_size=4, epochs=6, patience=1,
                          batch_size=256, device="cpu", seed=7, **changes)


@pytest.mark.parametrize("model_type", ["gru", "lstm"])
def test_repeated_choices_train_exactly_ten_epochs_and_save_global_best(tmp_path, monkeypatch, model_type):
    config = _daily_config(tmp_path, model_type=model_type)
    events, choices, snapshots, models, optimizers = [], [], [], [], []
    actual_predict, actual_adam = pipeline._predict, torch.optim.AdamW
    state = {"stage": None, "validation_epoch": 0}

    # Only the validation score is controlled. Data loading, recurrent forward
    # passes, minibatches, backpropagation, Adam updates and inference remain real.
    def validation_score(*args):
        epoch = state["validation_epoch"]
        return 0.5 if epoch == 13 else (1.0 if epoch == 1 else 2.0)

    def capture_prediction(model, *args, **kwargs):
        models.append(model)
        if state["stage"] == "training":
            state["validation_epoch"] += 1
            snapshots.append({key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()})
        return actual_predict(model, *args, **kwargs)

    class TrackedAdam(actual_adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.completed_steps = 0
            optimizers.append(self)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            self.completed_steps += 1
            return result

    def event(value):
        events.append(value)
        if "stage" in value:
            state["stage"] = value["stage"]

    def decide(details):
        choices.append(dict(details))
        epochs = [event["epoch"] for event in events if event["kind"] == "epoch"]
        assert epochs == list(range(1, details["epoch"] + 1))
        assert state["stage"] == "awaiting_decision"
        assert not any(event.get("stage") in {"evaluating", "saving"} for event in events)
        assert not Path(config.run_dir).exists()
        assert details["extension_epochs"] == 10
        assert np.isfinite(details["train_loss"])
        return len(choices) <= 2

    monkeypatch.setattr(pipeline, "_macro_loss", validation_score)
    monkeypatch.setattr(pipeline, "_predict", capture_prediction)
    monkeypatch.setattr(torch.optim, "AdamW", TrackedAdam)
    summary = train(config, data_kind="synthetic_demo", progress=lambda _: None,
                    on_event=event, on_training_decision=decide)

    assert [choice["epoch"] for choice in choices] == [2, 12, 22]
    assert [choice["reason"] for choice in choices] == ["early_stop", "extension_complete", "extension_complete"]
    assert [choice["decision_id"] for choice in choices] == [1, 2, 3]
    assert [choice["max_epochs"] for choice in choices] == [6, 12, 22]
    assert [choice["best_epoch"] for choice in choices] == [1, 1, 13]
    assert [choice["best_validation_loss"] for choice in choices] == [1.0, 1.0, 0.5]
    assert summary["best_epoch"] == 13
    assert [point["epoch"] for point in summary["history"]] == list(range(1, 23))
    assert config.epochs == summary["config"]["epochs"] == 6
    control = summary["training_control"]
    assert {key: control[key] for key in ("initial_epochs", "epoch_limit", "epochs_completed", "extension_epochs")} == {
        "initial_epochs": 6, "epoch_limit": 22, "epochs_completed": 22, "extension_epochs": 10,
    }
    assert control["decisions"] == [
        {**choice, "action": action} for choice, action in zip(choices, ("continue", "continue", "finish"))
    ]
    assert len(optimizers) == len({id(model) for model in models}) == 1
    batches = math.ceil(sum(summary["splits"]["eligible_counts"]["train"].values()) / config.batch_size)
    assert optimizers[0].completed_steps == batches * 22
    assert {int(value["step"]) for value in optimizers[0].state.values()} == {batches * 22}
    assert len(snapshots) == 22
    for first, last in ((0, 1), (11, 12), (12, 21)):
        assert any(not torch.equal(snapshots[first][key], snapshots[last][key]) for key in snapshots[first])
    checkpoint = torch.load(Path(config.run_dir) / "model.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(checkpoint["state_dict"][key], value) for key, value in snapshots[12].items())
    assert checkpoint["metadata"]["config"]["epochs"] == 6
    forecasts = json.loads((Path(config.run_dir) / "forecasts.json").read_text(encoding="utf-8"))
    assert predict(config.run_dir, config.data_dir, allow_stale=True, update_report=False) == forecasts
    assert [event["stage"] for event in events if event["kind"] == "stage"].count("evaluating") == 1
    assert [event["stage"] for event in events if event["kind"] == "stage"].count("saving") == 1
    epoch_events = [event for event in events if event["kind"] == "epoch"]
    assert [event["max_epochs"] for event in epoch_events] == [6] * 2 + [12] * 10 + [22] * 10


@pytest.mark.parametrize("interactive", [False, True])
def test_declining_or_automatic_early_stop_keeps_original_behavior(tmp_path, monkeypatch, interactive):
    config = _daily_config(tmp_path)
    scores = iter((1.0, 2.0))
    monkeypatch.setattr(pipeline, "_macro_loss", lambda *args: next(scores))
    events, choices = [], []

    def decline(details):
        choices.append(details)
        return False

    summary = train(config, data_kind="synthetic_demo", progress=lambda _: None,
                    on_event=events.append, on_training_decision=decline if interactive else None)
    assert summary["best_epoch"] == 1
    assert len(summary["history"]) == summary["training_control"]["epochs_completed"] == 2
    assert summary["training_control"]["initial_epochs"] == summary["training_control"]["epoch_limit"] == 6
    assert len(choices) == int(interactive)
    assert len([event for event in events if event["kind"] == "decision"]) == int(interactive)
    assert len(summary["training_control"]["decisions"]) == int(interactive)


def test_ordinary_epoch_limit_finishes_without_requesting_patience_choice(tmp_path, monkeypatch):
    config = _daily_config(tmp_path)
    scores = iter(1 / epoch for epoch in range(1, config.epochs + 1))
    monkeypatch.setattr(pipeline, "_macro_loss", lambda *args: next(scores))

    def unexpected_choice(details):
        pytest.fail("No patience threshold was reached")

    summary = train(config, data_kind="synthetic_demo", progress=lambda _: None,
                    on_training_decision=unexpected_choice)
    assert len(summary["history"]) == summary["best_epoch"] == 6
    assert summary["training_control"]["decisions"] == []


@pytest.mark.parametrize("answer", [None, 1, "continue"])
def test_invalid_decision_answer_never_evaluates_or_publishes(tmp_path, monkeypatch, answer):
    config = _daily_config(tmp_path)
    scores = iter((1.0, 2.0))
    monkeypatch.setattr(pipeline, "_macro_loss", lambda *args: next(scores))
    events = []
    with pytest.raises(ValueError, match="True.*False"):
        train(config, data_kind="synthetic_demo", progress=lambda _: None,
              on_event=events.append, on_training_decision=lambda details: answer)
    assert not Path(config.run_dir).exists()
    assert not any(event.get("stage") in {"evaluating", "saving"} for event in events)


def _payload(operation="train"):
    return {"operation": operation, "config": {"coins": ["BTC"], "epochs": 6, "patience": 1}}


class _LiveProcess:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.terminated = False

    def poll(self):
        return 1 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 1


@pytest.fixture
def decision_manager(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job(_payload())
    process = _LiveProcess()
    manager.processes[job["id"]] = process
    details = {"decision_id": 1, "reason": "early_stop", "epoch": 2,
               "max_epochs": 6, "best_epoch": 1, "best_validation_loss": 1.0,
               "train_loss": 0.5, "validation_loss": 2.0, "extension_epochs": 10}
    manager._update_job(job["id"], status="awaiting_decision", stage="awaiting_decision",
                        pending_decision=details)
    try:
        yield manager, job["id"], process, details
    finally:
        manager.close()


@pytest.mark.parametrize("action", ["continue", "finish"])
def test_manager_sends_one_matching_choice_and_rejects_duplicates(decision_manager, action):
    manager, identifier, process, details = decision_manager
    before = manager.get_job(identifier)
    manager.decide_training(identifier, {"decision_id": 1, "action": action})
    command = json.loads(process.stdin.getvalue())
    assert command["decision_id"] == 1 and command["action"] == action
    assert process.stdin.getvalue().endswith(b"\n")
    # The worker owns the state transition; a parent response cannot clobber a
    # worker's epoch or completion snapshot while sending its choice.
    after = manager.get_job(identifier)
    assert {key: value for key, value in after.items() if key != "elapsed_seconds"} == {
        key: value for key, value in before.items() if key != "elapsed_seconds"
    }
    with pytest.raises(DashboardError) as error:
        manager.decide_training(identifier, {"decision_id": 1, "action": action})
    assert error.value.status == 409
    assert process.stdin.getvalue().count(b"\n") == 1
    manager._update_job(identifier, pending_decision={**details, "decision_id": 2})
    manager.decide_training(identifier, {"decision_id": 2, "action": "finish"})
    assert len(process.stdin.getvalue().splitlines()) == 2


@pytest.mark.parametrize("payload", [
    None, [], {}, {"decision_id": 1}, {"action": "continue"},
    {"decision_id": True, "action": "continue"},
    {"decision_id": 0, "action": "continue"},
    {"decision_id": "1", "action": "continue"},
    {"decision_id": 1, "action": "extend"},
    {"decision_id": 1, "action": "continue", "epochs": 100},
])
def test_manager_rejects_invalid_choice_without_writing_pipe(decision_manager, payload):
    manager, identifier, process, _ = decision_manager
    with pytest.raises(DashboardError) as error:
        manager.decide_training(identifier, payload)
    assert error.value.status == 400
    assert process.stdin.getvalue() == b""


@pytest.mark.parametrize("condition", ["stale", "running", "queued", "completed", "download", "dead", "stopping"])
def test_manager_rejects_choices_for_unavailable_or_wrong_request(decision_manager, condition):
    manager, identifier, process, _ = decision_manager
    decision_id = 2 if condition == "stale" else 1
    if condition in {"running", "queued", "completed"}:
        manager._update_job(identifier, status=condition)
    elif condition == "download":
        manager._update_job(identifier, operation="download")
    elif condition == "dead":
        process.terminated = True
    elif condition == "stopping":
        manager.stopping.add(identifier)
    with pytest.raises(DashboardError) as error:
        manager.decide_training(identifier, {"decision_id": decision_id, "action": "continue"})
    assert error.value.status == 409
    assert process.stdin.getvalue() == b""


def test_awaiting_choice_remains_active_and_stop_discards_the_prompt(decision_manager):
    manager, identifier, process, _ = decision_manager
    with pytest.raises(DashboardError) as error:
        manager.start_job(_payload())
    assert error.value.status == 409
    stopped = manager.stop_job(identifier)
    assert stopped["status"] == "stopped"
    assert not stopped.get("pending_decision")
    assert process.terminated
    with pytest.raises(DashboardError) as error:
        manager.decide_training(identifier, {"decision_id": 1, "action": "continue"})
    assert error.value.status == 409


def test_concurrent_choices_send_exactly_one_command(decision_manager):
    manager, identifier, process, _ = decision_manager

    def send(_):
        try:
            manager.decide_training(identifier, {"decision_id": 1, "action": "continue"})
        except DashboardError as error:
            return error.status
        return 200

    with ThreadPoolExecutor(max_workers=6) as executor:
        statuses = list(executor.map(send, range(6)))
    assert statuses.count(200) == 1
    assert statuses.count(409) == 5
    assert process.stdin.getvalue().count(b"\n") == 1


@pytest.fixture
def decision_http(tmp_path, monkeypatch):
    server = create_server(tmp_path, port=0)
    monkeypatch.setattr(server.manager, "_launch", lambda job: None)
    job = server.manager.start_job(_payload())
    process = _LiveProcess()
    server.manager.processes[job["id"]] = process
    server.manager._update_job(job["id"], status="awaiting_decision", stage="awaiting_decision",
                               pending_decision={"decision_id": 1, "epoch": 2, "extension_epochs": 10})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = "http://127.0.0.1:" + str(server.server_address[1])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, *, value=None, headers=None):
        req = urllib.request.Request(address + path, data=None if value is None else json.dumps(value).encode(),
                                     method="GET" if value is None else "POST", headers=headers or {})
        try:
            response = opener.open(req, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    try:
        yield server, job["id"], process, address, request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("bad_header", ["csrf", "origin", "host"])
def test_http_decision_is_protected_by_same_origin_host_and_csrf(decision_http, bad_header):
    server, identifier, process, address, request = decision_http
    headers = {"Content-Type": "application/json", "X-CSRF-Token": server.csrf_token, "Origin": address}
    headers[{"csrf": "X-CSRF-Token", "origin": "Origin", "host": "Host"}[bad_header]] = "https://attacker.invalid"
    status, _ = request(f"/api/jobs/{identifier}/decision", value={"decision_id": 1, "action": "continue"}, headers=headers)
    assert status == 403
    assert process.stdin.getvalue() == b""


def test_http_reconnect_exposes_waiting_choice_and_matching_post_is_forwarded(decision_http):
    server, identifier, process, address, request = decision_http
    status, listing = request("/api/jobs")
    assert status == 200
    assert listing["active_job"]["id"] == identifier
    assert listing["active_job"]["pending_decision"]["decision_id"] == 1
    assert request(f"/api/jobs/{identifier}")[1]["status"] == "awaiting_decision"
    headers = {"Content-Type": "application/json", "X-CSRF-Token": server.csrf_token, "Origin": address}
    status, _ = request(f"/api/jobs/{identifier}/decision", value={"decision_id": 1, "action": "continue"}, headers=headers)
    assert status == 200
    assert json.loads(process.stdin.getvalue())["action"] == "continue"
    assert request("/api/jobs/unknown/decision", value={"decision_id": 1, "action": "finish"}, headers=headers)[0] == 404


def test_worker_waits_for_exact_request_and_persists_repeated_choices(tmp_path, monkeypatch):
    from crypto_forecast import dashboard_worker

    manager = JobManager(tmp_path)
    monkeypatch.setattr(manager, "_launch", lambda job: None)
    job = manager.start_job(_payload())
    path = manager.jobs_dir / (job["id"] + ".json")
    commands = queue.Queue()
    entered = threading.Event()
    finished = threading.Event()
    choices = []
    errors = []

    def fake_train(config, *, progress, on_event, on_training_decision):
        details = {"reason": "early_stop", "decision_id": 1, "epoch": 2,
                   "max_epochs": 6, "best_epoch": 1, "best_validation_loss": 1.0,
                   "train_loss": 0.5, "validation_loss": 2.0, "extension_epochs": 10}
        entered.set()
        choices.append(on_training_decision(details))
        assert choices == [True]
        current = json.loads(path.read_text(encoding="utf-8"))
        assert current["status"] == "running" and not current.get("pending_decision")
        choices.append(on_training_decision({**details, "decision_id": 2, "reason": "extension_complete", "epoch": 12}))
        return {"best_epoch": 1, "training_control": {"epoch_limit": 12, "epochs_completed": 12}}

    monkeypatch.setattr(dashboard_worker, "train", fake_train)

    def run():
        try:
            assert dashboard_worker.run_job(path, commands=commands) == 0
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        # A command for another prompt must never grant the current extension.
        commands.put({"decision_id": 99, "action": "continue"})
        assert not finished.wait(timeout=0.1)
        current = manager.get_job(job["id"])
        assert current["status"] == "awaiting_decision"
        assert current["pending_decision"]["decision_id"] == 1
        commands.put({"decision_id": 1, "action": "continue"})
        # FIFO makes an obsolete duplicate harmless before the second choice.
        commands.put({"decision_id": 1, "action": "continue"})
        commands.put({"decision_id": 2, "action": "finish"})
        assert finished.wait(timeout=5)
        assert not errors
        assert choices == [True, False]
        completed = manager.get_job(job["id"])
        assert completed["status"] == "completed" and not completed.get("pending_decision")
        assert completed["progress"]["max_epochs"] == 12
    finally:
        # Unblock a failed assertion without leaving a test-owned worker waiting.
        commands.put({"decision_id": 1, "action": "finish"})
        commands.put({"decision_id": 2, "action": "finish"})
        thread.join(timeout=5)
        manager.close()


def test_parent_command_reader_accepts_json_lines_and_preserves_eof_cancellation(monkeypatch):
    from crypto_forecast import dashboard_worker

    commands = queue.Queue(maxsize=8)
    chunks = iter((b'{"decision_id":1,"action":"cont',
                   b'inue"}\nnot json\n{"decision_id":true,"action":"continue"}\n',
                   b'{"decision_id":2,"action":"finish"}\n', b""))
    monkeypatch.setattr(dashboard_worker, "_read_parent_block", lambda fd, size: next(chunks))

    def parent_lost(code):
        raise SystemExit(code)

    monkeypatch.setattr(dashboard_worker.os, "_exit", parent_lost)
    with pytest.raises(SystemExit) as error:
        dashboard_worker.read_parent_commands(commands, fd=123)
    assert error.value.code == 3
    assert commands.get_nowait() == {"decision_id": 1, "action": "continue"}
    assert commands.get_nowait() == {"decision_id": 2, "action": "finish"}
    assert commands.empty()

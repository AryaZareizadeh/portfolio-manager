"""Optimized preparation must retain the original samples and split metadata."""

from dataclasses import replace
import os
import subprocess
import sys

import numpy as np
import pytest

from crypto_forecast import kronos_backend as backend
from crypto_forecast.features import split_samples
from crypto_forecast.pipeline import TrainingConfig
from test_kronos import candles


def reference(frames, config):
    samples = backend.build_samples(frames, config.lookback, config.kronos_timeframe,
                                    prediction_length=config.kronos_prediction_length)
    partitions, info = split_samples(samples, config.train_fraction, config.validation_fraction)
    excluded, latest = {}, {}
    for coin in config.coins:
        counts = {name: sum(s.coin == coin for s in values) for name, values in partitions.items()}
        if counts["train"] < 32 or counts["validation"] < 5 or counts["test"] < 5:
            excluded[coin] = f"Insufficient contiguous samples in purged splits: {counts}; need 32/5/5."
        else:
            try:
                latest[coin] = backend.inference_sample(coin, frames[coin], config.lookback,
                                                       config.kronos_timeframe,
                                                       prediction_length=config.kronos_prediction_length)
            except ValueError as error:
                excluded[coin] = str(error)
    coins = [coin for coin in config.coins if coin not in excluded]
    partitions = {name: [s for s in values if s.coin in coins] for name, values in partitions.items()}
    info["eligible_counts"] = {name: {coin: sum(s.coin == coin for s in values) for coin in coins}
                               for name, values in partitions.items()}
    selected = {name: backend.select_stride(values, config.kronos_train_stride if name == "train"
                                            else config.kronos_eval_stride)
                for name, values in partitions.items()}
    info["sampled_counts"] = {name: len(values) for name, values in selected.items()}
    info["sampling"] = {"after_split": True, "per_coin": True,
                        "train_stride": config.kronos_train_stride, "eval_stride": config.kronos_eval_stride}
    return selected, info, excluded, latest


def assert_sample(actual, expected):
    for name in ("coin", "origin", "label_end", "price"):
        assert getattr(actual, name) == getattr(expected, name)
    for name in ("x", "future", "y", "drift"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.x_timestamps.equals(expected.x_timestamps)
    assert actual.future_timestamps.equals(expected.future_timestamps)


@pytest.mark.parametrize("timeframe,length,prediction", [
    ("15m", 600, 3), ("1h", 2400, 168), ("4h", 600, 3), ("1d", 600, 3),
    ("4h", 1600, None), ("1d", 600, None),
])
def test_preparation_preserves_gaps_splits_exclusions_and_stride(timeframe, length, prediction):
    frame = candles(length, timeframe)
    frames = {"ETH": frame.drop(index=[180, length - 3]).reset_index(drop=True),
              "BTC": frame.drop(index=[120, 121]).reset_index(drop=True),
              "SHORT": frame.iloc[:16].copy()}
    config = TrainingConfig(coins=["BTC", "SHORT", "ETH"], model_type="kronos-small",
                            kronos_timeframe=timeframe, lookback=8,
                            kronos_prediction_length=prediction,
                            train_fraction=0.7, validation_fraction=0.2,
                            kronos_train_stride=7, kronos_eval_stride=3)
    expected, info, excluded, latest = reference(frames, config)
    actual, actual_info, actual_excluded, actual_latest = backend.prepare_training_samples(frames, config)
    assert actual_info == info
    assert actual_excluded == excluded
    assert actual_latest.keys() == latest.keys()
    for coin in latest:
        assert_sample(actual_latest[coin], latest[coin])
    for name in expected:
        assert len(actual[name]) == len(expected[name])
        for left, right in zip(actual[name], expected[name]):
            assert_sample(left, right)


def test_parallel_preparation_matches_serial_and_reports_parent_progress():
    frames = {coin: candles(600, "1h") for coin in ("ETH", "BTC")}
    config = TrainingConfig(coins=["BTC", "ETH"], model_type="kronos-small",
                            kronos_timeframe="1h", lookback=8, kronos_prediction_length=3,
                            kronos_train_stride=24, kronos_eval_stride=24)
    serial, info, excluded, _ = backend.prepare_training_samples(
        frames, replace(config, kronos_preparation_workers=1))
    events = []
    parallel, parallel_info, parallel_excluded, _ = backend.prepare_training_samples(
        frames, replace(config, kronos_preparation_workers=2),
        emit=lambda kind, **value: events.append({"kind": kind, **value}))
    assert parallel_info == info and parallel_excluded == excluded
    for name in serial:
        for left, right in zip(parallel[name], serial[name]):
            assert_sample(left, right)
        assert len(parallel[name]) == len(serial[name])
    materializing = [e for e in events if e.get("preparation_phase") == "materializing"]
    assert materializing[0]["preparation_completed"] == 0
    assert materializing[-1]["preparation_completed"] == sum(info["sampled_counts"].values())
    assert materializing[-1]["preparation_workers"] == 2
    assert [e["preparation_completed"] for e in materializing] == sorted(
        e["preparation_completed"] for e in materializing)


@pytest.mark.skipif(os.name != "nt", reason="Windows command-pipe/spawn interaction")
def test_dashboard_command_reader_does_not_block_preparation_process_startup():
    # The command pipe stays open and idle during preparation. A blocking stdin
    # read must not prevent spawned workers from duplicating their startup pipe.
    script = '''
import multiprocessing, os, queue, threading
from concurrent.futures import ProcessPoolExecutor
from crypto_forecast.dashboard_worker import read_parent_commands
from crypto_forecast.kronos_preparation import watch_parent
threading.Thread(target=read_parent_commands, args=(queue.Queue(maxsize=1),), daemon=True).start()
with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=watch_parent) as pool:
    print(pool.submit(os.getpid).result(timeout=8), flush=True)
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", script], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        # communicate() closes stdin and would mask the deadlock being tested.
        try:
            code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail("Preparation worker startup blocked while the dashboard command pipe was idle.")
        assert code == 0
        assert int(process.stdout.readline()) > 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def test_dashboard_command_pipe_delivers_commands_and_cancels_on_parent_eof():
    script = '''
import json, queue, threading
from crypto_forecast.dashboard_worker import read_parent_commands
commands = queue.Queue(maxsize=2)
threading.Thread(target=read_parent_commands, args=(commands,), daemon=True).start()
print(json.dumps(commands.get(timeout=5)), flush=True)
print(json.dumps(commands.get(timeout=5)), flush=True)
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", script], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        process.stdin.write(b'{"decision_id":1,"action":"cont')
        process.stdin.flush()
        process.stdin.write(b'inue"}\n{"coin":"LTC","published":true}\n')
        process.stdin.flush()
        assert process.wait(timeout=10) == 0
        import json
        assert json.loads(process.stdout.readline()) == {"decision_id": 1, "action": "continue"}
        assert json.loads(process.stdout.readline()) == {"coin": "LTC", "published": True}
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)

    script = '''
import queue
from crypto_forecast.dashboard_worker import read_parent_commands
read_parent_commands(queue.Queue())
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", script], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 3
        assert stderr == b""
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.parametrize("workers", [-1, 5, True, 1.5, "2"])
def test_preparation_worker_count_is_validated(workers):
    with pytest.raises(ValueError, match="kronos_preparation_workers"):
        TrainingConfig(coins=["BTC"], model_type="kronos-small",
                       kronos_preparation_workers=workers).validate()


def test_preparation_children_exit_when_training_parent_is_terminated():
    script = '''
import multiprocessing, os, time
from concurrent.futures import ProcessPoolExecutor
from crypto_forecast.kronos_preparation import watch_parent
pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=watch_parent)
print(pool.submit(os.getpid).result(timeout=15), flush=True)
pool.submit(time.sleep, 30)
time.sleep(.3)
os._exit(0)
'''
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        worker_pid = int(parent.stdout.readline())
        assert parent.wait(timeout=15) == 0
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x00100000, False, worker_pid)
            if handle:
                try:
                    assert kernel.WaitForSingleObject(handle, 5000) == 0
                finally:
                    kernel.CloseHandle(handle)
            else:
                assert ctypes.get_last_error() == 87  # Process has already exited.
        else:
            import time
            for _ in range(50):
                try:
                    os.kill(worker_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(.1)
            else:
                pytest.fail("Preparation worker survived its training parent.")
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout=15)


def test_cli_passes_preparation_worker_choice_to_training():
    from crypto_forecast.cli import _settings, parser
    config = _settings(parser().parse_args(["train", "--model-type", "kronos-small",
                                            "--kronos-preparation-workers", "3"]))
    config.validate()
    assert config.kronos_preparation_workers == 3


def test_automatic_small_preparation_avoids_process_overhead():
    frames = {coin: candles(600, "1h") for coin in ("BTC", "ETH")}
    config = TrainingConfig(coins=["BTC", "ETH"], model_type="kronos-small",
                            kronos_timeframe="1h", lookback=8, kronos_prediction_length=3)
    events = []
    backend.prepare_training_samples(frames, config, emit=lambda kind, **value: events.append(value))
    assert events[-1]["preparation_workers"] == 1

"""Live training events describe the same run that the pipeline publishes."""

import math

import numpy as np
import pandas as pd
import pytest

from crypto_forecast.pipeline import TrainingConfig, train


def test_progress_events_match_real_training_and_saved_summary(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index = np.arange(500)
    close = 100 * np.exp(np.cumsum(0.001 + 0.004 * np.sin(index / 11)))
    frame = pd.DataFrame({
        "timestamp": pd.date_range(end="2025-09-30", periods=len(index), tz="UTC"),
        "open": close * 0.995, "high": close * 1.02, "low": close * 0.98,
        "close": close, "volume": 1000 + 100 * np.sin(index / 9),
    })
    frame.to_csv(data_dir / "BTC.csv", index=False)
    config = TrainingConfig(coins=["BTC"], input_mode="daily", start="2020-01-01",
                            end="2025-10-01", data_dir=str(data_dir),
                            run_dir=str(tmp_path / "run"), lookback=10,
                            hidden_size=8, epochs=1, batch_size=32, device="cpu")
    events, logs = [], []
    summary = train(config, data_kind="synthetic_demo", progress=logs.append,
                    on_event=events.append)

    stages = [event for event in events if event["kind"] == "stage"]
    assert [event["stage"] for event in stages] == [
        "loading", "preparing", "training", "evaluating", "saving",
    ]
    training = next(event for event in stages if event["stage"] == "training")
    expected_counts = {part: sum(counts.values()) for part, counts in
                       summary["splits"]["eligible_counts"].items()}
    assert training["counts"] == expected_counts
    assert training["device"] == summary["device"] == "cpu"
    epoch, = [event for event in events if event["kind"] == "epoch"]
    assert {key: epoch[key] for key in summary["history"][0]} == summary["history"][0]
    assert epoch["epoch"] == epoch["best_epoch"] == summary["best_epoch"] == 1
    assert epoch["best_validation_loss"] == epoch["validation_loss"]
    assert epoch["max_epochs"] == 1

    batches = [event for event in events if event["kind"] == "batch"]
    total_batches = math.ceil(expected_counts["train"] / config.batch_size)
    assert batches[0]["batch"] == 1
    assert batches[-1]["batch"] == total_batches
    assert [event["batch"] for event in batches] == sorted({event["batch"] for event in batches})
    assert all(event["total_batches"] == total_batches and event["epoch"] == 1
               and event["max_epochs"] == 1 and np.isfinite(event["train_loss"])
               for event in batches)
    assert batches[-1]["train_loss"] == pytest.approx(epoch["train_loss"])
    elapsed = [event["elapsed_seconds"] for event in events]
    assert elapsed == sorted(elapsed) and min(elapsed) >= 0
    assert logs[0].startswith("Loaded daily candles")


def test_event_callback_can_stop_before_loading_data(tmp_path):
    config = TrainingConfig(coins=["BTC"], data_dir=str(tmp_path / "missing"),
                            run_dir=str(tmp_path / "run"))

    def stop(event):
        assert event["kind"] == "stage" and event["stage"] == "loading"
        raise RuntimeError("Stopped by training job controller")

    with pytest.raises(RuntimeError, match="Stopped by training job controller"):
        train(config, on_event=stop)
    assert not (tmp_path / "run").exists()

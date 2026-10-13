"""Real upstream Kronos modules exercise fine-tuning without downloading weights."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_forecast import HORIZONS
from crypto_forecast import kronos_backend as backend
from crypto_forecast.features import split_samples
from crypto_forecast.pipeline import TrainingConfig, predict, train


def tiny_modules():
    Kronos, Tokenizer, _ = backend._upstream()
    model = Kronos(s1_bits=2, s2_bits=2, n_layers=1, d_model=16, n_heads=4, ff_dim=32,
                   ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
                   token_dropout_p=0.0, learn_te=True)
    tokenizer = Tokenizer(d_in=6, d_model=16, n_heads=4, ff_dim=32, n_enc_layers=2,
                          n_dec_layers=2, ffn_dropout_p=0.0, attn_dropout_p=0.0,
                          resid_dropout_p=0.0, s1_bits=2, s2_bits=2,
                          beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=2)
    return model, tokenizer


def candles(length=420, timeframe="1d"):
    index = np.arange(length)
    close = 100 + 0.03 * index + 2 * np.sin(index / 12)
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=length,
                                                  freq=backend.interval_delta(timeframe), tz="UTC"),
                         "open": close * 0.999, "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": 1000 + 100 * np.sin(index / 9)})


def write_history(directory, *, timeframe="1d", length=420, metadata=True):
    frame = candles(length, timeframe)
    path = directory / timeframe / "BTC.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if metadata:
        path.with_suffix(".metadata.json").write_text(json.dumps({
            "source": "synthetic_demo", "coin": "BTC", "quote_currency": "USDT", "interval": timeframe,
            "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }), encoding="utf-8")
    return frame


def config_for(directory):
    return TrainingConfig(coins=["BTC"], model_type="kronos-small", kronos_timeframe="1d",
                          kronos_prediction_length=None,
                          data_dir=str(directory / "data"), run_dir=str(directory / "run"),
                          start="2024-01-01", end="2025-02-24", device="cpu", lookback=8,
                          epochs=1, patience=1, batch_size=8, learning_rate=0.005, seed=7,
                          kronos_tokenizer_epochs=1, kronos_tokenizer_patience=1,
                          kronos_tokenizer_learning_rate=0.005,
                          kronos_eval_stride=1000, kronos_train_stride=32)


def test_upstream_api_teacher_forcing_has_real_finite_gradients():
    torch.manual_seed(2)
    model, tokenizer = tiny_modules()
    backend._freeze_tokenizer(model, tokenizer)
    raw = torch.randn(2, 12, 6)
    with torch.no_grad():
        one, two = tokenizer.encode(raw, half=True)
    stamps = torch.zeros(2, 12, 5)
    logits_one, logits_two = model(one[:, :-1], two[:, :-1], stamps[:, :-1],
                                   use_teacher_forcing=True, s1_targets=one[:, 1:])
    assert logits_one.shape == logits_two.shape == (2, 11, 4)
    dataset_batch = (one, two, stamps)
    loss = backend._batch_loss(model, dataset_batch, torch.device("cpu"), lookback=8).mean()
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("embedding.emb_s1.weight", "head.proj_s2.weight", "transformer.0.self_attn.q_proj.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    assert all(parameter.grad is None for parameter in tokenizer.parameters())


@pytest.mark.parametrize("timeframe, expected", [("4h", [6, 42, 180]), ("1d", [1, 7, 30])])
def test_raw_samples_have_exact_closed_candle_horizons_and_zero_amount(timeframe, expected):
    frame = candles(420, timeframe)
    samples = backend.build_samples({"BTC": frame}, 8, timeframe)
    sample = samples[0]
    assert backend._bars(timeframe).tolist() == expected
    context = frame[frame.timestamp.isin(sample.x_timestamps)]
    assert np.array_equal(sample.x[:, :5], context.loc[:, list(backend.RAW_FEATURES[:-1])])
    assert sample.x.shape == (8, 6) and sample.future.shape == (expected[-1], 6)
    assert np.count_nonzero(sample.x[:, 5]) == np.count_nonzero(sample.future[:, 5]) == 0
    assert sample.x_timestamps[-1] + backend.interval_delta(timeframe) == sample.origin
    assert sample.future_timestamps[0] == sample.origin
    for j, horizon in enumerate(HORIZONS):
        assert sample.future_timestamps[expected[j] - 1] + backend.interval_delta(timeframe) == sample.origin + pd.Timedelta(days=horizon)
        assert sample.y[j] == pytest.approx(np.log(sample.future[expected[j] - 1, 3] / sample.price))


def test_raw_sample_gaps_remove_context_and_future_crossings():
    frame = candles(160)
    missing = frame.timestamp.iloc[70]
    frame = frame.drop(index=70).reset_index(drop=True)
    samples = backend.build_samples({"BTC": frame}, 8, "1d")
    assert samples
    assert all(not (sample.x_timestamps[0] <= missing <= sample.future_timestamps[-1]) for sample in samples)
    gapped_latest = frame.drop(index=len(frame) - 3).reset_index(drop=True)
    with pytest.raises(ValueError, match="missing candles"):
        backend.inference_sample("BTC", gapped_latest, 8, "1d")


def test_window_normalization_and_tokenizer_never_use_future_values():
    torch.manual_seed(13)
    _, tokenizer = tiny_modules()
    tokenizer.eval()
    sample = backend.build_samples({"BTC": candles()}, 8, "1d")[20]
    altered = replace(sample, future=sample.future * 100)
    first = backend.normalize_sample(sample)
    second = backend.normalize_sample(altered)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[2], second[2]) and np.array_equal(first[3], second[3])
    assert np.array_equal(first[2], sample.x.mean(axis=0))
    assert np.array_equal(first[3], sample.x.std(axis=0) + 1e-5)
    with torch.no_grad():
        original = tokenizer.encode(torch.tensor(np.concatenate(first[:2])[None]), half=True)
        changed = tokenizer.encode(torch.tensor(np.concatenate(second[:2])[None]), half=True)
        context_only = tokenizer.encode(torch.tensor(first[0][None]), half=True)
    for index in (0, 1):
        assert torch.equal(original[index][:, :8], changed[index][:, :8])
        assert torch.equal(original[index][:, :8], context_only[index])


def test_candle_preparation_progress_counts_checked_windows_including_gaps(monkeypatch):
    frames = {"BTC": candles(40, "1h").drop(index=20).reset_index(drop=True),
              "ETH": candles(25, "1h")}
    expected = backend.build_samples(frames, 8, "1h", prediction_length=3)
    ticks = iter(range(1000))
    monkeypatch.setattr(backend.time, "monotonic", lambda: next(ticks))
    events = []
    actual = backend.build_samples(frames, 8, "1h", prediction_length=3,
                                   emit=lambda kind, **values: events.append({"kind": kind, **values}))
    windows = [event for event in events if event["preparation_phase"] == "windows"]
    total = sum(len(frame) - 3 - 7 for frame in frames.values())
    assert windows[0]["preparation_completed"] == 0
    assert [event["preparation_completed"] for event in windows] == sorted(
        event["preparation_completed"] for event in windows)
    assert all(event["preparation_total"] == total for event in windows)
    assert {event["coin"] for event in windows} == {"BTC", "ETH"}
    assert windows[-1]["preparation_completed"] == total
    assert windows[-1]["usable_samples"] == len(actual) < total
    assert any(0 < event["preparation_completed"] < total for event in windows)
    assert events[-1]["preparation_phase"] == "sorting"
    assert events[-1]["preparation_total"] == 0
    assert [(sample.coin, sample.origin) for sample in actual] == [
        (sample.coin, sample.origin) for sample in expected]
    for observed, original in zip(actual, expected):
        assert np.array_equal(observed.x, original.x)
        assert np.array_equal(observed.future, original.future)


def test_teacher_validation_masks_future_cross_attention_and_restores_mode():
    torch.manual_seed(3)
    model, _ = tiny_modules()
    model.eval()
    one = torch.randint(0, 4, (1, 9))
    two = torch.randint(0, 4, (1, 9))
    targets = torch.randint(0, 4, (1, 9))
    changed_one, changed_two, changed_targets = one.clone(), two.clone(), targets.clone()
    changed_one[:, 5:] = (changed_one[:, 5:] + 1) % 4
    changed_two[:, 5:] = (changed_two[:, 5:] + 2) % 4
    changed_targets[:, 5:] = (changed_targets[:, 5:] + 1) % 4
    with torch.no_grad(), backend._causal_validation(model):
        assert not model.training and model.dep_layer.cross_attn.training
        original = model(one, two, use_teacher_forcing=True, s1_targets=targets)
        changed = model(changed_one, changed_two, use_teacher_forcing=True, s1_targets=changed_targets)
        assert torch.allclose(original[0][:, :5], changed[0][:, :5], atol=1e-6)
        assert torch.allclose(original[1][:, :5], changed[1][:, :5], atol=1e-6)
    assert not model.training and not model.dep_layer.cross_attn.training


def test_shared_calendar_purge_precedes_stride_selection():
    samples = backend.build_samples({"BTC": candles()}, 8, "1d")
    partitions, info = split_samples(samples)
    assert max(s.label_end for s in partitions["train"]) < pd.Timestamp(info["val_start"])
    assert max(s.label_end for s in partitions["validation"]) < pd.Timestamp(info["test_start"])
    for values in partitions.values():
        assert backend.select_stride(values, 6) == values[::6]


def test_selected_stream_loader_needs_no_other_timeframes_and_preserves_units(tmp_path):
    write_history(tmp_path, timeframe="4h", metadata=False)
    loaded = backend.load_candles(tmp_path, ["BTC"], "4h", as_of="2025-01-01")
    assert set(loaded) == {"BTC"}
    assert loaded["BTC"].attrs["provenance"]["source"] == "local_csv_user_supplied"
    path = tmp_path / "4h" / "BTC.metadata.json"
    path.write_text(json.dumps({"source": "user", "coin": "BTC", "quote_currency": "EUR", "interval": "4h"}))
    with pytest.raises(ValueError, match="USDT"):
        backend.load_candles(tmp_path, ["BTC"], "4h")


def test_daily_loader_keeps_legacy_layout_and_source_metadata_rules(tmp_path):
    frame = candles(50)
    path = tmp_path / "BTC.csv"
    frame.to_csv(path, index=False)
    path.with_suffix(".metadata.json").write_text(json.dumps({"source": "legacy user data", "quote_currency": "USDT"}))
    loaded = backend.load_candles(tmp_path, ["BTC"], "1d", as_of="2025-01-01")["BTC"]
    assert len(loaded) == 50 and loaded.attrs["interval"] == "1d"
    assert loaded.attrs["provenance"]["source"] == "legacy user data"


@pytest.mark.parametrize("timeframe, lookback", [("4h", 334), ("1d", 484)])
def test_full_training_sequence_must_fit_official_context(timeframe, lookback):
    with pytest.raises(ValueError, match="512"):
        TrainingConfig(coins=["BTC"], model_type="kronos-small", kronos_timeframe=timeframe,
                       lookback=lookback, kronos_prediction_length=None).validate()


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("kronos-real-modules")
    write_history(directory / "data")
    config = config_for(directory)
    torch.manual_seed(config.seed)
    model, tokenizer = tiny_modules()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    original_tokenizer = {key: value.clone() for key, value in tokenizer.state_dict().items()}
    events = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend, "load_pretrained", lambda config, device: (model.to(device), tokenizer.to(device)))
        summary = train(config, progress=lambda _: None, on_event=events.append)
    checkpoint = torch.load(directory / "run" / "model.pt", weights_only=True, map_location="cpu")
    return config, summary, checkpoint, initial, original_tokenizer, events


def test_genuine_kronos_trains_tokenizer_then_predictor_and_saves_pair(tiny_run):
    config, summary, checkpoint, initial, original_tokenizer, events = tiny_run
    assert summary["parameters_changed"] is True and summary["best_epoch"] == 1
    assert summary["data_kind"] == "synthetic_demo"
    assert summary["kronos"]["tokenizer_frozen"] is True
    assert summary["kronos"]["forecast_device"] == "cpu"
    assert summary["kronos"]["prediction_bars"] == [1, 7, 30]
    assert any(not torch.equal(initial[key], value) for key, value in checkpoint["state_dict"].items())
    assert any(not torch.equal(original_tokenizer[key], value) for key, value in checkpoint["tokenizer_state_dict"].items())
    assert summary["kronos"]["tokenizer_finetuned"] is True
    assert summary["tokenizer_training"]["best_epoch"] == 1
    assert summary["tokenizer_training"]["parameters_changed"] is True
    assert len(summary["tokenizer_training"]["history"]) == 1
    assert checkpoint["metadata"]["format_version"] == 3
    assert checkpoint["metadata"]["features"] == list(backend.RAW_FEATURES)
    assert summary["splits"]["sampling"]["after_split"] is True
    assert summary["splits"]["sampled_counts"] == {"train": 8, "validation": 1, "test": 1}
    assert np.isfinite(summary["history"][0]["train_loss"])
    assert np.isfinite(summary["history"][0]["validation_loss"])
    assert {e["stage"] for e in events if e["kind"] == "stage"} >= {"loading", "preparing", "training", "evaluating", "saving"}
    assert next(i for i, e in enumerate(events) if e["kind"] == "tokenizer_epoch") < next(
        i for i, e in enumerate(events) if e["kind"] == "tokenizing")
    preparation = next(e for e in events if e.get("stage") == "preparing"
                       and e.get("training_component") == "predictor")
    assert preparation["epoch"] == preparation["best_epoch"] == preparation["total_batches"] == 0
    assert preparation["train_loss"] is preparation["validation_loss"] is preparation["best_validation_loss"] is None
    raw_progress = [e for e in events if e.get("preparation_phase") == "windows"]
    assert raw_progress[0]["preparation_completed"] == 0
    assert raw_progress[-1]["preparation_completed"] == raw_progress[-1]["preparation_total"]
    for partition in ("training", "validation"):
        token_progress = [e for e in events if e.get("preparation_partition") == partition]
        assert token_progress[0]["preparation_completed"] == 0
        assert token_progress[-1]["preparation_completed"] == summary["splits"]["sampled_counts"][
            "train" if partition == "training" else "validation"]
    assert {row["model"] for row in summary["metrics"]["test"]} == {"kronos-small", "persistence", "drift"}


def test_tokenizer_best_validation_checkpoint_is_frozen_before_tokenization(tmp_path, monkeypatch):
    write_history(tmp_path / "data")
    config = replace(config_for(tmp_path), kronos_tokenizer_epochs=3)
    model, tokenizer = tiny_modules()
    states, prepared = [], []
    initial = backend._state_hash(tokenizer)
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: (model, tokenizer))

    def validation(*args):
        states.append(backend._state_hash(tokenizer))
        return [0.2, 0.5][len(states) - 1]

    encode_dataset = backend._token_dataset

    def prepare(samples, module, *args, **kwargs):
        assert not module.training and not any(p.requires_grad for p in module.parameters())
        prepared.append(backend._state_hash(module))
        return encode_dataset(samples, module, *args, **kwargs)

    monkeypatch.setattr(backend, "_tokenizer_validation_loss", validation)
    monkeypatch.setattr(backend, "_token_dataset", prepare)
    summary = train(config, progress=lambda _: None)
    assert len(states) == 2 and states[0] != states[1] and states[0] != initial
    assert prepared == [states[0], states[0]]
    assert backend._state_hash(tokenizer) == states[0]
    assert summary["tokenizer_training"]["best_epoch"] == 1


def test_predictor_only_option_preserves_pretrained_tokenizer(tmp_path, monkeypatch):
    write_history(tmp_path / "data")
    config = replace(config_for(tmp_path), kronos_finetune_tokenizer=False)
    model, tokenizer = tiny_modules()
    initial = backend._state_hash(tokenizer)
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: (model, tokenizer))
    events = []
    summary = train(config, progress=lambda _: None, on_event=events.append)
    assert backend._state_hash(tokenizer) == initial
    assert summary["kronos"]["tokenizer_finetuned"] is False
    assert summary["tokenizer_training"] == {"enabled": False}
    assert not any(e["kind"] == "tokenizer_epoch" for e in events)


def test_tokenizer_training_has_encoder_decoder_and_quantizer_gradients():
    torch.manual_seed(7)
    _, tokenizer = tiny_modules()
    tokenizer.train()
    loss, full_reconstruction = backend._tokenizer_loss(tokenizer, torch.randn(2, 12, 6))
    assert torch.isfinite(loss) and torch.isfinite(full_reconstruction)
    loss.backward()
    for name in ("embed.weight", "quant_embed.weight", "post_quant_embed_pre.weight",
                 "post_quant_embed.weight", "head.weight", "encoder.0.self_attn.q_proj.weight",
                 "decoder.0.self_attn.q_proj.weight"):
        gradient = dict(tokenizer.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)


def test_nonfinite_tokenizer_validation_never_publishes_run(tmp_path, monkeypatch):
    write_history(tmp_path / "data")
    model, tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_pretrained", lambda *args: (model, tokenizer))
    monkeypatch.setattr(backend, "_tokenizer_validation_loss", lambda *args: float("nan"))
    with pytest.raises(ValueError, match="tokenizer validation loss"):
        train(config_for(tmp_path), progress=lambda _: None)
    assert not (tmp_path / "run" / "model.pt").exists()


def test_offline_reload_reconstructs_identical_sampled_forecasts(tiny_run, monkeypatch):
    config, summary, _, _, _, _ = tiny_run
    expected = json.loads((Path(config.run_dir) / "forecasts.json").read_text())
    Kronos, Tokenizer, _ = backend._upstream()

    def forbidden_download(*args, **kwargs):
        pytest.fail("A saved Kronos run must reconstruct both modules without contacting Hub.")

    monkeypatch.setattr(Kronos, "from_pretrained", forbidden_download)
    monkeypatch.setattr(Tokenizer, "from_pretrained", forbidden_download)
    monkeypatch.setattr(backend, "load_pretrained", forbidden_download)
    actual = predict(config.run_dir, config.data_dir, allow_stale=True, update_report=False)
    assert actual == expected
    for row in actual:
        assert pd.Timestamp(row["target_date"]) == pd.Timestamp(row["as_of"]) + pd.Timedelta(days=row["horizon_days"])
    with pytest.raises(ValueError, match="Stale"):
        predict(config.run_dir, config.data_dir, update_report=False)
    with pytest.raises(ValueError, match="trained coins"):
        predict(config.run_dir, config.data_dir, coins=["ETH"], allow_stale=True)


def test_prediction_honors_saved_device_selection(tiny_run, monkeypatch):
    config, _, checkpoint, _, _, _ = tiny_run
    checkpoint = {**checkpoint, "metadata": {**checkpoint["metadata"],
                  "config": {**checkpoint["metadata"]["config"], "device": "cuda"}}}
    selections = []

    def selected_device(requested):
        selections.append(requested)
        return torch.device("cpu")

    # Exercise the saved-device resolver without requiring GPU hardware in CI.
    monkeypatch.setattr("crypto_forecast.pipeline._device", selected_device)
    backend.predict_kronos(config.run_dir, config.data_dir, allow_stale=True,
                           update_report=False, checkpoint=checkpoint)
    assert selections == ["cuda"]


def test_early_stop_extension_keeps_same_state_and_saves_global_best(tmp_path, monkeypatch):
    write_history(tmp_path / "data")
    config = replace(config_for(tmp_path), epochs=20)
    model, tokenizer = tiny_modules()
    monkeypatch.setattr(backend, "load_pretrained", lambda config, device: (model, tokenizer))
    monkeypatch.setattr(backend, "_validation_loss", lambda *args, **kwargs: 1.0)
    choices, events = [], []

    def decide(details):
        choices.append(details)
        return len(choices) == 1

    summary = train(config, progress=lambda _: None, on_event=events.append, on_training_decision=decide)
    assert [choice["epoch"] for choice in choices] == [2, 12]
    assert [choice["reason"] for choice in choices] == ["early_stop", "extension_complete"]
    assert summary["training_control"]["epochs_completed"] == 12
    assert summary["training_control"]["epoch_limit"] == 12
    assert summary["best_epoch"] == 1
    assert [e["epoch"] for e in events if e["kind"] == "epoch"] == list(range(1, 13))


def test_cancel_during_preparation_never_publishes_checkpoint(tmp_path):
    write_history(tmp_path / "data")
    config = config_for(tmp_path)

    def cancel(event):
        if event["stage"] == "preparing":
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        train(config, progress=lambda _: None, on_event=cancel)
    assert not (Path(config.run_dir) / "model.pt").exists()

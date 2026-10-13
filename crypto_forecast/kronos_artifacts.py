"""Durable tokenizer checkpoints and a strict data contract for safe stage reuse."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile

import torch

# Optimizer, batch size and device are intentionally independent of the data contract.
DATA_FIELDS = ("coins", "lookback", "kronos_timeframe", "kronos_prediction_length",
               "train_fraction", "validation_fraction", "kronos_train_stride", "kronos_eval_stride")


def data_signature(config, manifest, splits):
    values = asdict(config) if not isinstance(config, dict) else config
    contract = {key: values.get(key, None) for key in DATA_FIELDS}
    contract["coins"] = sorted(values["coins"])
    contract["rows"] = {coin: entry["timeframes"][values["kronos_timeframe"]]["used_rows_sha256"]
                        for coin, entry in manifest.items()}
    contract["boundaries"] = {key: splits[key] for key in ("val_start", "test_start")}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def save_tokenizer(run_dir, state, constructor, summary):
    """Publish complete files by replacement, leaving the previous checkpoint intact on failure."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-stage-", dir=directory) as temporary:
        stage = Path(temporary)
        torch.save({"tokenizer_state_dict": state, "tokenizer_config": constructor,
                    "metadata": summary}, stage / "tokenizer.pt")
        (stage / "tokenizer_summary.json").write_text(json.dumps(summary, allow_nan=False, indent=2), encoding="utf-8")
        # The checkpoint contains authoritative metadata; summary is just the inventory view.
        os.replace(stage / "tokenizer.pt", directory / "tokenizer.pt")
        os.replace(stage / "tokenizer_summary.json", directory / "tokenizer_summary.json")


def tokenizer_summary(config, constructor, training, *, signature, manifest, splits,
                      coins, state_hash, complete, data_kind):
    return {"artifact_type": "kronos-tokenizer", "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
            "coins": coins, "model_type": config.model_type, "tokenizer_config": constructor,
            "features": ["open", "high", "low", "close", "volume", "amount"],
            "normalization": "historical-lookback-mean-std-v1", "data_signature": signature,
            "data_manifest": manifest, "splits": splits, "data_kind": data_kind,
            "training_complete": complete, "best_epoch": training["best_epoch"],
            "tokenizer_sha256": state_hash, "tokenizer_training": training}


def load_tokenizer_checkpoint(directory, signature, features):
    """Read only weights and primitives. Refuse a tokenizer selected using different held-out data."""
    directory = Path(directory)
    path = directory / "tokenizer.pt"
    try:
        if path.is_file():
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            metadata = checkpoint["metadata"]
            if (metadata.get("artifact_type") != "kronos-tokenizer" or metadata.get("format_version") != 1
                    or metadata.get("features") != list(features)
                    or metadata.get("normalization") != "historical-lookback-mean-std-v1"):
                raise ValueError("Saved tokenizer format or normalization is incompatible.")
            source_signature = metadata["data_signature"]
            constructor = checkpoint["tokenizer_config"]
        else:
            # Earlier completed runs already contain a matched tuned tokenizer.
            checkpoint = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
            metadata = checkpoint["metadata"]
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if (metadata.get("features") != list(features)
                    or not metadata.get("kronos", {}).get("tokenizer_finetuned")):
                raise ValueError("Select a run containing a fine-tuned Kronos tokenizer.")
            source_signature = data_signature(metadata["config"], summary["data_manifest"], summary["splits"])
            constructor = metadata["tokenizer_config"]
        if source_signature != signature:
            raise ValueError("Reusing a tuned tokenizer requires the same candle data and chronological partitions, "
                             "lookback, forecast length and strides. Load its saved data settings first.")
        return checkpoint["tokenizer_state_dict"], constructor, metadata
    except (OSError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load saved tokenizer checkpoint from {directory}: {error}") from error

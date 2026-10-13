"""Download official pretrained weights once, recording their exact revisions."""

import json
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_ID = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
PRETRAINED_MODELS = {"kronos-small": MODEL_ID, "kronos-base": "NeoQuasar/Kronos-base"}


def download_pretrained(model_dir=None, progress=print, *, model="kronos-small"):
    if model not in PRETRAINED_MODELS:
        raise ValueError("Choose kronos-small or kronos-base.")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise ValueError('Install Kronos dependencies first: python -m pip install -e ".[kronos]"') from error
    directory = Path(model_dir or Path("models") / model)
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    # Finish both downloads before updating any working local snapshot.
    with tempfile.TemporaryDirectory(prefix=".kronos-download-", dir=directory) as temporary:
        stage = Path(temporary)
        for name, repository in (("model", PRETRAINED_MODELS[model]), ("tokenizer", TOKENIZER_ID)):
            progress(f"Downloading {repository} to {(directory / name).resolve()}")
            try:
                revision = HfApi().model_info(repository).sha
                snapshot_download(repo_id=repository, revision=revision, local_dir=stage / name,
                                  allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin", "README.md", "LICENSE*"])
            except Exception as error:
                raise ValueError(f"Could not download {repository}: {error}") from error
            path = stage / name
            if not (path / "config.json").is_file() or not any((path / file).is_file() for file in ("model.safetensors", "pytorch_model.bin")):
                raise ValueError(f"Incomplete pretrained snapshot in {path}; rerun kronos-setup.")
            hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in path.iterdir() if file.is_file()}
            entries[name] = {"repository": repository, "revision": revision,
                             "path": str((directory / name).resolve()), "sha256": hashes}
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                    "source_repository": "https://github.com/shiyu-coder/Kronos",
                    "source_revision": UPSTREAM_REVISION, **entries}
        # A failed publication must not leave an old readiness manifest.
        (directory / "manifest.json").unlink(missing_ok=True)
        for name in entries:
            (directory / name).mkdir(parents=True, exist_ok=True)
            for file in (stage / name).iterdir():
                if file.is_file():
                    os.replace(file, directory / name / file.name)
        pending = directory / ".manifest.json.tmp"
        pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(pending, directory / "manifest.json")
    progress(f"{model.capitalize()} is ready. Weight revisions saved in {(directory / 'manifest.json').resolve()}")
    return manifest

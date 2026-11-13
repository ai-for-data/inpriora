"""Local and Hugging Face model artifact loading."""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors.torch import load_file

from ..core.config import ModelConfig
from ..core.model import InprioraModel

DEFAULT_REPO_ID = "ai-for-data/inpriora"

_MODEL_CACHE: "OrderedDict[tuple, Tuple[InprioraModel, ModelConfig]]" = OrderedDict()
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE_SIZE = 8


def _artifact_key(
    config_path: Path,
    weights_path: Path,
    kind: str,
    device: str,
) -> tuple:
    try:
        stat = weights_path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    except OSError:
        stamp = (0, 0, 0, 0)
    return (str(config_path), str(weights_path), stamp, kind, device)


def _model_cache_enabled() -> bool:
    return os.environ.get("INPRIORA_DISABLE_MODEL_CACHE", "").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def clear_model_cache() -> None:
    """Release models retained for reuse across estimator instances."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _build_model(
    config: ModelConfig,
    weights_path: Path,
    device: str,
) -> InprioraModel:
    # Keep the process-global RNG untouched while the network is initialized.
    with torch.random.fork_rng(devices=[]):
        model = InprioraModel(config)
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model


def _offline(flag: bool) -> bool:
    if flag:
        return True
    for name in ("HF_HUB_OFFLINE", "INPRIORA_OFFLINE"):
        if os.environ.get(name, "").lower() in ("1", "true", "yes", "on"):
            return True
    return False


def _download(
    repo_id: str,
    filename: str,
    revision: Optional[str],
    offline: bool,
) -> Path:
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        if offline:
            raise FileNotFoundError(
                f"Artifact {filename!r} is not cached and offline mode is enabled."
            ) from None
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
        )


def resolve_artifact(
    model_path: Optional[str | Path],
    repo_id: str,
    revision: Optional[str],
    kind: str,
    offline: bool = False,
) -> Tuple[Path, Path]:
    """Resolve a local directory/file or a Hugging Face artifact pair."""
    if model_path is not None:
        path = Path(model_path).expanduser()
        if path.is_dir():
            config_path = path / "config.json"
            weights_path = path / "model.safetensors"
        elif path.suffix == ".safetensors":
            weights_path = path
            config_path = path.with_name("config.json")
        else:
            raise FileNotFoundError(
                f"model_path must be an artifact directory or a .safetensors file, got {path}."
            )
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Missing config.json or model.safetensors under {path}."
            )
        return config_path, weights_path

    offline = _offline(offline)
    config_path = _download(repo_id, f"{kind}/config.json", revision, offline)
    weights_path = _download(repo_id, f"{kind}/model.safetensors", revision, offline)
    return config_path, weights_path


def load_model(
    *,
    model_path: Optional[str | Path] = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
    kind: str = "classifier",
    device: str = "cpu",
    offline: bool = False,
) -> Tuple[InprioraModel, ModelConfig, Dict[str, Any]]:
    """Load a converted model from a local path or Hugging Face."""
    config_path, weights_path = resolve_artifact(
        model_path,
        repo_id,
        revision,
        kind,
        offline,
    )
    config_dict = json.loads(config_path.read_text())
    config = ModelConfig.from_dict(config_dict)
    if config.model_type != kind:
        raise ValueError(
            f"Expected a {kind} artifact, but found model_type={config.model_type!r}."
        )

    if not _model_cache_enabled():
        model = _build_model(config, weights_path, device)
    else:
        key = _artifact_key(config_path, weights_path, kind, device)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is None:
                model = _build_model(config, weights_path, device)
                _MODEL_CACHE[key] = (model, config)
                _MODEL_CACHE.move_to_end(key)
                while len(_MODEL_CACHE) > _MODEL_CACHE_SIZE:
                    _MODEL_CACHE.popitem(last=False)
            else:
                model, config = cached
                _MODEL_CACHE.move_to_end(key)

    metadata = {
        "config_path": str(config_path),
        "weights_path": str(weights_path),
        "repo_id": repo_id,
        "revision": revision,
    }
    return model, config, metadata

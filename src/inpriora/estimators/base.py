"""Shared estimator behavior."""

from __future__ import annotations

import gc
import multiprocessing as mp
import pickle
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator

from ..core.model import InprioraModel
from ..core.runner import InferenceRunner
from ..device import resolve_device
from .loading import load_model


class BaseInprioraEstimator(BaseEstimator):
    """Device, model loading, caching, and persistence helpers."""

    def _resolve_device(self) -> None:
        self.device_ = resolve_device(self.device)

    _model_kind: str = "classifier"

    def _load_model(self, kind: str | None = None) -> None:
        kind = kind or self._model_kind
        if self.device_.type == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            self.device_ = resolve_device("cpu")
        self.model_, self.model_config_, self.model_paths_ = load_model(
            model_path=self.model_path,
            repo_id=self.repo_id,
            revision=self.revision,
            kind=kind,
            device=str(self.device_),
            offline=self.offline,
        )

    def _build_runner(self) -> None:
        self.runner_ = InferenceRunner(self.device_, amp=self.amp)

    def _wants_ranks(self) -> bool:
        return bool(getattr(self.model_config_, "rank_channel", False))

    def _yeojohnson_device(self) -> Optional[str]:
        return str(self.device_) if self.device_.type == "cuda" else None

    def _reclaim_memory(self) -> None:
        gc.collect()
        if self.device_.type == "cuda":
            torch.cuda.empty_cache()

    @contextmanager
    def _thread_context(self) -> Iterator[None]:
        if self.n_jobs is None:
            yield
            return
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero.")
        old_threads = torch.get_num_threads()
        logical = mp.cpu_count()
        if self.n_jobs > 0:
            if self.n_jobs > logical:
                warnings.warn(
                    f"n_jobs={self.n_jobs} exceeds the {logical} logical cores; "
                    f"using {logical} threads.",
                    UserWarning,
                    stacklevel=3,
                )
            threads = min(logical, self.n_jobs)
        else:
            threads = max(1, logical + 1 + self.n_jobs)
        torch.set_num_threads(threads)
        try:
            yield
        finally:
            torch.set_num_threads(old_threads)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.non_deterministic = True
        return tags

    def save(self, path: str | Path) -> None:
        """Serialize the fitted estimator to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=5)

    @classmethod
    def load(cls, path: str | Path, device=None):
        """Load a fitted estimator from a file."""
        with Path(path).open("rb") as handle:
            estimator = pickle.load(handle)
        if device is not None:
            estimator.device = device
            estimator._resolve_device()
            if hasattr(estimator, "model_"):
                estimator.model_.to(estimator.device_)
            estimator._build_runner()
            if hasattr(estimator, "model_cache_"):
                estimator.model_cache_ = {
                    key: memory.to(estimator.device_)
                    for key, memory in estimator.model_cache_.items()
                }
        return estimator

    def _move_model(self) -> None:
        self.model_.to(self.device_)
        self.model_.eval()

    def _prepare_views(self, X_numeric):
        if getattr(self, "cache_mode_", None) is not None:
            views = self.ensemble_.transform(
                X_numeric,
                mode="test",
                return_ranks=self._wants_ranks(),
            )
            return views, True
        views = self.ensemble_.transform(
            X_numeric,
            mode="both",
            return_ranks=self._wants_ranks(),
        )
        return views, False

    def _as_tensor(self, values) -> torch.Tensor:
        return torch.from_numpy(np.asarray(values)).float().to(self.device_)

    def _optional_tensor(self, values):
        return None if values is None else self._as_tensor(values)

    @staticmethod
    def _member_slices(n_members: int, batch_size: Optional[int]):
        size = batch_size or n_members
        return size, [
            slice(start, min(start + size, n_members))
            for start in range(0, n_members, size)
        ]

    def _model_forward(self, *args, **kwargs):
        with self.runner_.context():
            return self.model_(*args, **kwargs)

    def _model_predict_query(self, *args, **kwargs):
        with self.runner_.context():
            return self.model_.predict_query(*args, **kwargs)

    def _model_store_cache(self, *args, **kwargs):
        with self.runner_.context():
            return self.model_.build_cache(*args, **kwargs)

    @property
    def model(self) -> InprioraModel:
        return self.model_
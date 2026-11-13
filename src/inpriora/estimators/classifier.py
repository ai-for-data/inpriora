"""Scikit-learn compatible classifier."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from ..core.cache import ShardedContextCache
from ..features.encode import InputEncoder
from ..features.ensemble import EnsembleBuilder
from .base import BaseInprioraEstimator
from .loading import DEFAULT_REPO_ID


class InprioraClassifier(ClassifierMixin, BaseInprioraEstimator):
    """Tabular classifier using in-context learning.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.
    norm_methods : list of str, optional
        Feature normalization methods. Defaults to ``["none", "power"]``.
    feature_shuffle : str, default="latin"
        Feature permutation strategy.
    class_shuffle : str, default="shift"
        Class label permutation strategy.
    outlier_threshold : float, default=4.0
        Z-score clipping threshold.
    temperature : float, default=1.0
        Softmax temperature applied to averaged logits.
    average_logits : bool, default=True
        Average member logits instead of member probabilities.
    many_class : bool, default=True
        Enable hierarchical classification above the native class count.
    batch_size : int, optional
        Number of ensemble members processed per forward pass.
    cache : bool or str, default=True
        Keep Context key/value caches. ``"repr"`` uses the smaller cache.
    model_path : str or Path, optional
        Local artifact directory or ``model.safetensors`` file.
    repo_id : str
        Hugging Face repository id.
    revision : str, optional
        Hugging Face revision.
    device : str or torch.device, optional
        ``"cuda"`` or ``"cpu"``. Defaults to CUDA when available.
    amp : bool, default=True
        Use fp16 autocast on CUDA.
    random_seed : int, default=0
        Seed for ensemble permutations.
    n_jobs : int, optional
        Torch thread count override.
    verbose : bool, default=False
        Print input encoding details.
    offline : bool, default=False
        Disable Hugging Face downloads.
    """

    _model_kind = "classifier"

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[list[str]] = None,
        feature_shuffle: str = "latin",
        class_shuffle: str = "shift",
        outlier_threshold: float = 4.0,
        temperature: float = 1.0,
        average_logits: bool = True,
        many_class: bool = True,
        batch_size: Optional[int] = 8,
        cache: bool | str = True,
        model_path: Optional[str] = None,
        repo_id: str = DEFAULT_REPO_ID,
        revision: Optional[str] = None,
        device=None,
        amp: bool = True,
        random_seed: int = 0,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        offline: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feature_shuffle = feature_shuffle
        self.class_shuffle = class_shuffle
        self.outlier_threshold = outlier_threshold
        self.temperature = temperature
        self.average_logits = average_logits
        self.many_class = many_class
        self.batch_size = batch_size
        self.cache = cache
        self.model_path = model_path
        self.repo_id = repo_id
        self.revision = revision
        self.device = device
        self.amp = amp
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.offline = offline

    def _resolve_cache_mode(self) -> Optional[str]:
        if not self.cache:
            return None
        if self.cache is True or self.cache == "kv":
            mode = "kv"
        elif self.cache == "repr":
            mode = "repr"
        else:
            raise ValueError("cache must be False, True, 'kv', or 'repr'.")
        if getattr(self, "many_class_", False):
            warnings.warn(
                "Caching is disabled for datasets that require hierarchical "
                "classification.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return mode

    def fit(self, X, y):
        """Fit the feature pipeline and optional Context cache."""
        if y is None:
            raise ValueError("The classifier requires y.")
        check_classification_targets(y)

        self._resolve_device()
        self._build_runner()
        self._load_model()
        self._move_model()

        self.n_samples_in_ = int(np.asarray(y).shape[0])
        self.y_encoder_ = LabelEncoder()
        y_codes = self.y_encoder_.fit_transform(y).astype(np.int64)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.native_classes_ = int(self.model_config_.native_classes)
        self.many_class_ = self.n_classes_ > self.native_classes_
        if self.many_class_ and not self.many_class:
            raise ValueError(
                f"The dataset has {self.n_classes_} classes, exceeding the model's "
                f"native limit of {self.native_classes_}. Set many_class=True to "
                "enable hierarchical classification."
            )

        self.X_encoder_ = InputEncoder(verbose=self.verbose)
        X_numeric = self.X_encoder_.fit_transform(X)

        self.ensemble_ = EnsembleBuilder(
            classification=True,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods,
            feature_shuffle=self.feature_shuffle,
            class_shuffle=self.class_shuffle,
            outlier_threshold=self.outlier_threshold,
            random_seed=self.random_seed,
            yeojohnson_device=self._yeojohnson_device(),
        )
        self.ensemble_.fit(X_numeric, y_codes)
        self.n_estimators_actual_ = self.ensemble_.n_estimators_actual

        self.cache_mode_ = self._resolve_cache_mode()
        if self.cache_mode_ is not None:
            self._build_cache()

        self.fitted_ = True
        return self

    def _build_group_cache(
        self,
        X_members,
        y_members,
        ranks,
        mode: str,
    ):
        n_members = X_members.shape[0]
        size = self.batch_size or n_members
        while True:
            try:
                caches = []
                for start in range(0, n_members, size):
                    stop = min(start + size, n_members)
                    X_batch = self._as_tensor(X_members[start:stop])
                    y_batch = self._as_tensor(y_members[start:stop])
                    rank_batch = self._optional_tensor(
                        None if ranks is None else ranks[start:stop]
                    )
                    caches.append(
                        self._model_store_cache(
                            X_batch,
                            y_batch,
                            mode=mode,
                            ranks=rank_batch,
                        )
                    )
                if len(caches) == 1:
                    return caches[0]
                return ShardedContextCache(caches)
            except torch.OutOfMemoryError:
                self._reclaim_memory()
                if size > 1:
                    size = max(1, size // 2)
                    continue
                if mode == "kv":
                    mode = "repr"
                    size = self.batch_size or n_members
                    continue
                raise

    def _build_cache(self) -> None:
        train_views = self.ensemble_.transform(
            X=None,
            mode="train",
            return_ranks=self._wants_ranks(),
        )
        caches = OrderedDict()
        for norm_method, batch in train_views.items():
            ranks = batch[2] if len(batch) > 2 else None
            caches[norm_method] = self._build_group_cache(
                batch[0],
                batch[1],
                ranks,
                self.cache_mode_,
            )
        self.model_cache_ = caches

    def _member_logits(self, X_members, y_members, ranks):
        n_members = X_members.shape[0]
        size = self.batch_size or n_members
        while True:
            try:
                outputs = []
                for start in range(0, n_members, size):
                    stop = min(start + size, n_members)
                    X_batch = self._as_tensor(X_members[start:stop])
                    y_batch = self._as_tensor(y_members[start:stop])
                    rank_batch = self._optional_tensor(
                        None if ranks is None else ranks[start:stop]
                    )
                    output = self._model_forward(
                        X_batch,
                        y_batch,
                        ranks=rank_batch,
                        temperature=self.temperature,
                        n_classes=self.n_classes_,
                    )
                    outputs.append(output.float().cpu().numpy())
                return np.concatenate(outputs, axis=0)
            except torch.OutOfMemoryError:
                self._reclaim_memory()
                if size <= 1:
                    raise
                size = max(1, size // 2)

    def _cached_logits(self, X_members, cache, ranks):
        n_members = X_members.shape[0]
        size = self.batch_size or n_members
        while True:
            try:
                outputs = []
                for start in range(0, n_members, size):
                    stop = min(start + size, n_members)
                    X_batch = self._as_tensor(X_members[start:stop])
                    rank_batch = self._optional_tensor(
                        None if ranks is None else ranks[start:stop]
                    )
                    subset = cache.slice_batch(start, stop)

                    def forward_block(
                        row_start,
                        row_stop,
                        X_batch=X_batch,
                        rank_batch=rank_batch,
                        subset=subset,
                    ):
                        return self._model_predict_query(
                            X_batch[:, row_start:row_stop],
                            cache=subset,
                            ranks=(
                                None
                                if rank_batch is None
                                else rank_batch[:, row_start:row_stop]
                            ),
                        )

                    output = self.runner_.run_row_blocks(
                        forward_block,
                        X_batch.shape[1],
                        dim=1,
                    )
                    outputs.append(output.float().cpu().numpy())
                return np.concatenate(outputs, axis=0)
            except torch.OutOfMemoryError:
                self._reclaim_memory()
                if size <= 1:
                    raise
                size = max(1, size // 2)

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        scaled = logits / temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exponent = np.exp(scaled)
        return exponent / exponent.sum(axis=-1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and X.ndim == 1:
            raise ValueError("X must be two-dimensional.")

        with self._thread_context():
            X_numeric = self.X_encoder_.transform(X)
            views, cached = self._prepare_views(X_numeric)
            outputs = []
            for norm_method, batch in views.items():
                if cached:
                    ranks = batch[1] if len(batch) > 1 else None
                    outputs.append(
                        self._cached_logits(
                            batch[0],
                            self.model_cache_[norm_method],
                            ranks,
                        )
                    )
                else:
                    ranks = batch[2] if len(batch) > 2 else None
                    outputs.append(
                        self._member_logits(batch[0], batch[1], ranks)
                    )
            outputs = np.concatenate(outputs, axis=0)

            class_shuffles = [
                pattern
                for group in self.ensemble_.class_shuffles_.values()
                for pattern in group
            ]
            n_estimators = len(class_shuffles)
            if n_estimators == 0:
                raise RuntimeError("The ensemble contains no members.")

            if not self.average_logits:
                outputs = [
                    self._softmax(output, self.temperature)
                    for output in outputs
                ]

            average = np.zeros_like(outputs[0], dtype=np.float64)
            for index, pattern in enumerate(class_shuffles):
                average += outputs[index][..., pattern]
            average /= n_estimators

            if self.average_logits:
                average = self._softmax(average, self.temperature)
            average = np.clip(average, 1e-12, None)
            return average / average.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return self.y_encoder_.inverse_transform(np.argmax(probabilities, axis=1))
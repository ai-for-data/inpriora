"""Scikit-learn compatible regressor."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Sequence

import numpy as np
import torch
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..core.cache import ContextCache, ShardedContextCache
from ..core.distribution import GridProjector
from ..features.encode import InputEncoder
from ..features.ensemble import EnsembleBuilder
from ..features.target import inverse_transform_borders, make_target_transform
from .base import BaseInprioraEstimator
from .loading import DEFAULT_REPO_ID


class InprioraRegressor(RegressorMixin, BaseInprioraEstimator):
    """Tabular regressor using in-context learning and a bucket distribution.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.
    norm_methods : list of str, optional
        Feature normalization methods. Defaults to ``["none"]``.
    target_transforms : sequence, default=(None, None, None, None,
        "safepower", "safepower", "rankgauss", "rankgauss")
        Per-member target transforms, assigned round-robin across members.
    temperature : float, default=1.0
        Temperature applied to member bucket logits before mixing.
    feature_shuffle : str, default="latin"
        Feature permutation strategy.
    outlier_threshold : float, default=4.0
        Z-score clipping threshold.
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

    _model_kind = "regressor"

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[List[str]] = None,
        target_transforms: Sequence[Optional[str]] = (
            None,
            None,
            None,
            None,
            "safepower",
            "safepower",
            "rankgauss",
            "rankgauss",
        ),
        temperature: float = 1.0,
        feature_shuffle: str = "latin",
        outlier_threshold: float = 4.0,
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
        self.target_transforms = target_transforms
        self.temperature = temperature
        self.feature_shuffle = feature_shuffle
        self.outlier_threshold = outlier_threshold
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
            return "kv"
        if self.cache == "repr":
            return "repr"
        raise ValueError("cache must be False, True, 'kv', or 'repr'.")

    def fit(self, X, y):
        """Fit the feature pipeline, target transforms, and optional cache."""
        if y is None:
            raise ValueError("The regressor requires y.")
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values.")

        self._resolve_device()
        self._build_runner()
        self._load_model()
        self._move_model()

        self.is_constant_target_ = bool(np.unique(y).size == 1)
        self.constant_value_ = float(y[0]) if self.is_constant_target_ else None
        self.n_samples_in_ = int(y.shape[0])

        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()

        self.X_encoder_ = InputEncoder(verbose=self.verbose)
        X_numeric = self.X_encoder_.fit_transform(X)

        if self.is_constant_target_:
            self.fitted_ = True
            return self

        self.ensemble_ = EnsembleBuilder(
            classification=False,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods or ["none"],
            feature_shuffle=self.feature_shuffle,
            outlier_threshold=self.outlier_threshold,
            random_seed=self.random_seed,
            yeojohnson_device=self._yeojohnson_device(),
        )
        self.ensemble_.fit(X_numeric, y_scaled)
        self.n_estimators_actual_ = self.ensemble_.n_estimators_actual
        self._fit_targets(y_scaled)

        self.cache_mode_ = self._resolve_cache_mode()
        if self.cache_mode_ is not None:
            self._build_cache()

        self.fitted_ = True
        return self

    def _fit_targets(self, y_scaled: np.ndarray) -> None:
        self._projector_cache_: dict = {}
        base_borders = self.model_.bucket_distribution.borders.detach().cpu().numpy()
        methods = tuple(self.target_transforms)
        if not methods:
            raise ValueError("target_transforms must contain at least one entry.")

        self.target_values_ = OrderedDict()
        self.target_borders_ = OrderedDict()
        self.target_cancels_ = OrderedDict()

        for norm_method, configs in self.ensemble_.ensemble_configs_.items():
            count = len(configs)
            balanced, leftover = divmod(count, len(methods))
            member_methods = [
                method for method in methods for _ in range(balanced)
            ] + list(methods[:leftover])

            values = []
            borders = []
            cancels = []
            for name in member_methods:
                transform = make_target_transform(name)
                if transform is None:
                    values.append(y_scaled.copy())
                    borders.append(base_borders.copy())
                    cancels.append(None)
                else:
                    transformed = transform.fit_transform(
                        y_scaled.reshape(-1, 1)
                    ).ravel()
                    member_borders, cancel = inverse_transform_borders(
                        base_borders,
                        transform,
                    )
                    values.append(transformed)
                    borders.append(member_borders)
                    cancels.append(cancel)

            self.target_values_[norm_method] = np.stack(values, axis=0)
            self.target_borders_[norm_method] = np.stack(borders, axis=0)
            self.target_cancels_[norm_method] = cancels

    def _build_group_cache(
        self,
        X_members,
        y_members,
        ranks,
        mode: str,
    ) -> ContextCache | ShardedContextCache:
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
                self.target_values_[norm_method],
                ranks,
                self.cache_mode_,
            )
        self.model_cache_ = caches

    def _member_logits(self, X_members, y_members, ranks) -> torch.Tensor:
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
                    )
                    outputs.append(output.float())
                return torch.cat(outputs, dim=0)
            except torch.OutOfMemoryError:
                self._reclaim_memory()
                if size <= 1:
                    raise
                size = max(1, size // 2)

    def _project_logits(
        self,
        logits: torch.Tensor,
        source_borders,
        cancels,
        shared_borders: torch.Tensor,
        projector_cache: dict,
    ) -> torch.Tensor:
        total = None
        for index in range(logits.shape[0]):
            borders = np.ascontiguousarray(source_borders[index])
            key = borders.tobytes()
            projector = projector_cache.get(key)
            if projector is None:
                projector = GridProjector(
                    torch.as_tensor(borders, dtype=torch.float32),
                    shared_borders,
                )
                projector_cache[key] = projector

            member = logits[index] / self.temperature
            cancel = cancels[index]
            if cancel is not None:
                member = member.clone()
                member[..., torch.as_tensor(
                    cancel,
                    dtype=torch.bool,
                    device=member.device,
                )] = float("-inf")
            probabilities = projector.project(member.float())
            total = probabilities if total is None else total + probabilities
        return total

    def _accumulate_cached(
        self,
        X_members,
        cache: ContextCache,
        ranks,
        accumulator: torch.Tensor,
        source_borders,
        cancels,
        shared_borders: torch.Tensor,
    ) -> None:
        n_members = X_members.shape[0]
        size = self.batch_size or n_members
        while True:
            try:
                for start in range(0, n_members, size):
                    stop = min(start + size, n_members)
                    X_batch = self._as_tensor(X_members[start:stop])
                    rank_batch = self._optional_tensor(
                        None if ranks is None else ranks[start:stop]
                    )
                    subset = cache.slice_batch(start, stop)
                    borders_batch = source_borders[start:stop]
                    cancel_batch = cancels[start:stop]
                    batch_sum = torch.zeros_like(accumulator)
                    projector_cache = self._projector_cache_

                    def forward_block(
                        row_start,
                        row_stop,
                        X_batch=X_batch,
                        rank_batch=rank_batch,
                        subset=subset,
                        borders_batch=borders_batch,
                        cancel_batch=cancel_batch,
                        batch_sum=batch_sum,
                        projector_cache=projector_cache,
                    ):
                        logits = self._model_predict_query(
                            X_batch[:, row_start:row_stop],
                            cache=subset,
                            ranks=(
                                None
                                if rank_batch is None
                                else rank_batch[:, row_start:row_stop]
                            ),
                        )
                        probabilities = self._project_logits(
                            logits,
                            borders_batch,
                            cancel_batch,
                            shared_borders,
                            projector_cache,
                        )
                        batch_sum[row_start:row_stop] += probabilities
                        return logits.new_empty((logits.shape[0], 0))

                    self.runner_.run_row_blocks(
                        forward_block,
                        X_batch.shape[1],
                        dim=1,
                    )
                    accumulator += batch_sum
                return
            except torch.OutOfMemoryError:
                self._reclaim_memory()
                if size <= 1:
                    raise
                size = max(1, size // 2)

    def _inverse_statistic(self, values: np.ndarray, statistic: str) -> np.ndarray:
        values = np.asarray(values)
        if statistic == "variance":
            return values * float(self.y_scaler_.scale_[0]) ** 2
        return self.y_scaler_.inverse_transform(
            values.reshape(-1, 1)
        ).reshape(values.shape)

    @staticmethod
    def _constant_predictions(
        n_samples: int,
        value: float,
        requested: List[str],
        alphas: Optional[List[float]],
    ):
        probability_levels = (
            alphas if alphas is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        results = {}
        for statistic in requested:
            if statistic == "variance":
                results[statistic] = np.zeros(n_samples, dtype=np.float64)
            elif statistic == "quantiles":
                results[statistic] = np.full(
                    (n_samples, len(probability_levels)),
                    value,
                    dtype=np.float64,
                )
            elif statistic in ("mean", "median", "mode"):
                results[statistic] = np.full(n_samples, value, dtype=np.float64)
            else:
                raise ValueError(f"Unsupported regression output type {statistic!r}.")
        return results[requested[0]] if len(requested) == 1 else results

    def predict(
        self,
        X,
        output_type: str | List[str] = "mean",
        alphas: Optional[List[float]] = None,
    ):
        """Predict target values or distribution summaries.

        Parameters
        ----------
        X : array-like
            Test features.
        output_type : str or list of str, default="mean"
            One or more of ``"mean"``, ``"median"``, ``"mode"``,
            ``"variance"``, and ``"quantiles"``.
        alphas : list of float, optional
            Probability levels for ``"quantiles"``.

        Returns
        -------
        ndarray or dict
            Point predictions or a mapping of requested statistics.
        """
        check_is_fitted(self)
        if isinstance(X, np.ndarray) and X.ndim == 1:
            raise ValueError("X must be two-dimensional.")
        if not hasattr(self, "_projector_cache_"):
            self._projector_cache_ = {}

        requested = [output_type] if isinstance(output_type, str) else list(output_type)
        allowed = {"mean", "median", "mode", "variance", "quantiles"}
        unknown = set(requested) - allowed
        if unknown:
            raise ValueError(f"Unsupported regression output types: {sorted(unknown)}")

        if getattr(self, "is_constant_target_", False):
            return self._constant_predictions(
                int(np.asarray(X).shape[0]),
                self.constant_value_,
                requested,
                alphas,
            )

        with self._thread_context():
            X_numeric = self.X_encoder_.transform(X)
            shared_borders = self.model_.bucket_distribution.borders.detach()
            n_buckets = int(shared_borders.numel() - 1)
            n_test = int(np.asarray(X_numeric).shape[0])
            views, cached = self._prepare_views(X_numeric)
            probability_sum = None
            total_members = 0

            if cached:
                probability_sum = torch.zeros(
                    (n_test, n_buckets),
                    dtype=torch.float32,
                    device=self.device_,
                )
                for norm_method, batch in views.items():
                    ranks = batch[1] if len(batch) > 1 else None
                    self._accumulate_cached(
                        batch[0],
                        self.model_cache_[norm_method],
                        ranks,
                        probability_sum,
                        self.target_borders_[norm_method],
                        self.target_cancels_[norm_method],
                        shared_borders,
                    )
                    total_members += batch[0].shape[0]
            else:
                for norm_method, batch in views.items():
                    ranks = batch[2] if len(batch) > 2 else None
                    logits = self._member_logits(
                        batch[0],
                        self.target_values_[norm_method],
                        ranks,
                    )
                    probabilities = self._project_logits(
                        logits,
                        self.target_borders_[norm_method],
                        self.target_cancels_[norm_method],
                        shared_borders,
                        self._projector_cache_,
                    )
                    probability_sum = (
                        probabilities
                        if probability_sum is None
                        else probability_sum + probabilities
                    )
                    total_members += logits.shape[0]

            mixture = probability_sum / total_members
            mixture_logits = torch.log(mixture.clamp_min(1e-30))
            summary = self.model_.bucket_distribution.summarize(
                mixture_logits,
                requested,
                levels=alphas,
            )
            if not isinstance(summary, dict):
                summary = {requested[0]: summary}
            results = {
                statistic: self._inverse_statistic(
                    summary[statistic].cpu().numpy(),
                    statistic,
                )
                for statistic in requested
            }
            return results[requested[0]] if len(requested) == 1 else results
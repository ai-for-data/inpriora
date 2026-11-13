"""Ensemble view construction."""

from __future__ import annotations

import itertools
import os
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .pipeline import FEATURE_NORMALIZERS, FeaturePipeline, UniqueFeatureFilter
from .rank import RankReference
from .validation import check_features


class Shuffler:
    """Generate index permutations for ensemble diversity."""

    def __init__(
        self,
        n_elements: int,
        method: str = "latin",
        max_elements_for_latin: int = 4000,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_elements = n_elements
        self.method = method
        self.max_elements_for_latin = max_elements_for_latin
        self.random_state = random_state
        self.rng_ = random.Random(random_state)

    def shuffle(self, n_estimators: int) -> List[List[int]]:
        """Return permutation patterns."""
        indices = list(range(self.n_elements))
        if self.n_elements > self.max_elements_for_latin and self.method == "latin":
            method = "random"
        else:
            method = self.method

        if method == "none" or n_estimators == 1:
            return [indices]

        if method == "shift":
            return [indices[-index:] + indices[:-index] for index in range(self.n_elements)]
        if method == "random":
            if self.n_elements <= 5:
                all_permutations = [list(permutation) for permutation in itertools.permutations(indices)]
                return self.rng_.sample(all_permutations, min(n_estimators, len(all_permutations)))
            return [self.rng_.sample(indices, self.n_elements) for _ in range(n_estimators)]
        if method == "latin":
            return self._latin_squares()
        raise ValueError(
            f"Unknown shuffle method {method!r}; expected 'shift', 'random', 'latin', or 'none'."
        )

    def _latin_squares(self) -> List[List[int]]:
        n = self.n_elements
        base = []
        remaining = list(range(n))
        for level in range(n, 0, -1):
            if level == 1:
                base.append(remaining[0])
                break
            symbol = self.rng_.choice(remaining)
            remaining.remove(symbol)
            base.append(symbol)

        row_order = np.arange(n)
        self.rng_.shuffle(row_order)
        column_order = np.arange(n)
        self.rng_.shuffle(column_order)

        lookup = (column_order[:, None] - row_order[None, :]) % n
        columns = np.asarray(base, dtype=np.int64)[lookup]
        return [columns[index].tolist() for index in range(n)]


class EnsembleBuilder(BaseEstimator, TransformerMixin):
    """Build normalized and shuffled views of a training table."""

    def __init__(
        self,
        classification: bool,
        n_estimators: int = 8,
        norm_methods: Optional[List[str]] = None,
        feature_shuffle: str = "latin",
        class_shuffle: str = "shift",
        outlier_threshold: float = 4.0,
        random_seed: Optional[int] = 0,
        yeojohnson_device: Optional[str] = None,
    ) -> None:
        self.classification = classification
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feature_shuffle = feature_shuffle
        self.class_shuffle = class_shuffle
        self.outlier_threshold = outlier_threshold
        self.random_seed = random_seed
        self.yeojohnson_device = yeojohnson_device

    def _resolve_norm_methods(self) -> List[str]:
        if self.norm_methods is None:
            methods = ["none", "power"] if self.classification else ["none"]
        elif isinstance(self.norm_methods, str):
            methods = [self.norm_methods]
        else:
            methods = list(self.norm_methods)
        for method in methods:
            if method not in FEATURE_NORMALIZERS:
                raise ValueError(f"Unsupported normalization method {method!r}.")
        return methods

    def fit(self, X, y):
        """Fit preprocessing pipelines and generate ensemble configurations."""
        X = check_features(X)
        y = np.asarray(y)
        self.norm_methods_ = self._resolve_norm_methods()

        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.rank_reference_ = RankReference.fit(X)
        self.train_rank_ = self._rank(X, True)
        if self.classification:
            self.n_classes_ = len(np.unique(y))

        self.rng_ = random.Random(self.random_seed)
        (
            self.ensemble_configs_,
            self.feature_shuffles_,
            self.class_shuffles_,
        ) = self._generate()

        self.preprocessors_ = OrderedDict()
        for norm_method in self.ensemble_configs_:
            pipeline = FeaturePipeline(
                normalization_method=norm_method,
                outlier_threshold=self.outlier_threshold,
                random_state=self.random_seed,
                yeojohnson_device=self.yeojohnson_device,
            )
            pipeline.fit(X)
            self.preprocessors_[norm_method] = pipeline
        return self

    def _generate(self):
        feature_shuffler = Shuffler(
            n_elements=self.n_features_in_,
            method=self.feature_shuffle,
            random_state=self.random_seed,
        )
        feature_shuffles = feature_shuffler.shuffle(self.n_estimators)

        if self.classification:
            class_shuffler = Shuffler(
                n_elements=self.n_classes_,
                method=self.class_shuffle,
                random_state=self.random_seed,
            )
            class_patterns = class_shuffler.shuffle(self.n_estimators)
        else:
            class_patterns = [None]

        shuffle_configs = list(itertools.product(feature_shuffles, class_patterns))
        self.rng_.shuffle(shuffle_configs)
        shuffle_norm_configs = list(
            itertools.product(shuffle_configs, self.norm_methods_)
        )[: self.n_estimators]

        used_methods = list(dict.fromkeys(config[1] for config in shuffle_norm_configs))
        ensemble_configs = OrderedDict()
        feature_shuffle_dict = OrderedDict()
        class_shuffle_dict = OrderedDict()
        for method in used_methods:
            configs = [
                config[0] for config in shuffle_norm_configs if config[1] == method
            ]
            feature_shuffle_dict[method] = [config[0] for config in configs]
            class_shuffle_dict[method] = [config[1] for config in configs]
            ensemble_configs[method] = configs
        return ensemble_configs, feature_shuffle_dict, class_shuffle_dict

    @property
    def n_estimators_actual(self) -> int:
        return sum(len(configs) for configs in self.ensemble_configs_.values())

    def transform(
        self,
        X=None,
        mode: str = "both",
        return_ranks: bool = True,
    ) -> Dict[str, Tuple[np.ndarray, ...]]:
        """Return per-normalization ensemble views.

        Parameters
        ----------
        X : array-like, optional
            Test features. Required for ``mode`` in ``("both", "test")``.
        mode : {"both", "train", "test"}, default="both"
            Which views to return.
        return_ranks : bool, default=True
            Include the per-cell rank channel.
        """
        check_is_fitted(self, "ensemble_configs_")
        if mode not in ("both", "train", "test"):
            raise ValueError(f"Unknown transform mode {mode!r}.")

        if mode == "train":
            return self._transform_train(return_ranks)
        if mode == "test":
            if X is None:
                raise ValueError("X is required when mode='test'.")
            return self._transform_test(X, return_ranks)
        if X is None:
            raise ValueError("X is required when mode='both'.")
        return self._transform_both(X, return_ranks)

    def _rank(self, X, return_ranks: bool):
        if not return_ranks:
            return None
        use_torch = (
            self.yeojohnson_device is not None
            and torch.cuda.is_available()
            and os.environ.get("INPRIORA_TORCH_PREPROCESS", "1").lower()
            not in ("0", "false", "no", "off")
        )
        if use_torch:
            ranks = self.rank_reference_.transform_torch(
                X,
                self.yeojohnson_device,
            )
        else:
            ranks = self.rank_reference_.transform(X)
        return np.asarray(ranks, dtype=np.float32)

    def _transform_train(self, return_ranks: bool):
        y = self.y_
        rank_base = self.train_rank_ if return_ranks else None
        data = OrderedDict()
        for norm_method, configs in self.ensemble_configs_.items():
            transformed = self.preprocessors_[norm_method].X_transformed_
            features = []
            targets = []
            ranks = []
            for feature_shuffle, class_pattern in configs:
                features.append(transformed[:, feature_shuffle])
                if rank_base is not None:
                    ranks.append(rank_base[:, feature_shuffle])
                if self.classification:
                    targets.append(np.asarray(class_pattern)[y.astype(int)])
                else:
                    targets.append(y)
            payload = [np.stack(features, axis=0), np.stack(targets, axis=0)]
            if rank_base is not None:
                payload.append(np.stack(ranks, axis=0))
            data[norm_method] = tuple(payload)
        return data

    def _transform_test(self, X, return_ranks: bool):
        X = self.unique_filter_.transform(X)
        rank_base = self._rank(X, return_ranks)
        data = OrderedDict()
        for norm_method, configs in self.ensemble_configs_.items():
            transformed = self.preprocessors_[norm_method].transform(X)
            features = []
            ranks = []
            for feature_shuffle, _ in configs:
                features.append(transformed[:, feature_shuffle])
                if rank_base is not None:
                    ranks.append(rank_base[:, feature_shuffle])
            payload = [np.stack(features, axis=0)]
            if rank_base is not None:
                payload.append(np.stack(ranks, axis=0))
            data[norm_method] = tuple(payload)
        return data

    def _transform_both(self, X, return_ranks: bool):
        X = self.unique_filter_.transform(X)
        rank_train = self.train_rank_ if return_ranks else None
        rank_test = self._rank(X, return_ranks)
        y = self.y_
        data = OrderedDict()
        for norm_method, configs in self.ensemble_configs_.items():
            pipeline = self.preprocessors_[norm_method]
            train = pipeline.X_transformed_
            test = pipeline.transform(X)
            variant = np.concatenate([train, test], axis=0)
            rank_variant = None
            if rank_train is not None:
                rank_variant = np.concatenate([rank_train, rank_test], axis=0)

            features = []
            targets = []
            ranks = []
            for feature_shuffle, class_pattern in configs:
                features.append(variant[:, feature_shuffle])
                if rank_variant is not None:
                    ranks.append(rank_variant[:, feature_shuffle])
                if self.classification:
                    targets.append(np.asarray(class_pattern)[y.astype(int)])
                else:
                    targets.append(y)
            payload = [np.stack(features, axis=0), np.stack(targets, axis=0)]
            if rank_variant is not None:
                payload.append(np.stack(ranks, axis=0))
            data[norm_method] = tuple(payload)
        return data
"""Top-level model assembly."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .cache import ContextCache, StackCache
from .config import ModelConfig
from .distribution import BucketDistribution, spline_borders
from .hierarchical import HierarchicalReadout
from .scaling import BoundedQueryScaling
from .stages import ColumnStage, ICLStage, RefinementStage, RowStage

_HIERARCHICAL_NODE_TEMPERATURE = 0.9


class InprioraModel(nn.Module):
    """Tabular foundation model for in-context classification and regression."""

    def __init__(self, config: Union[ModelConfig, Dict]) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        config.validate()
        self.config = config
        self.model_type = config.model_type

        self.column_stage = ColumnStage(
            embed_dim=config.embed_dim,
            num_layers=config.column_blocks,
            heads=config.column_heads,
            hidden_width=config.hidden_width,
            num_inducing=config.column_inducing_points,
            max_classes=int(config.native_classes or 0),
            regression_method=None if config.is_classifier else "bar_dist",
            rank_encoding=config.rank_channel,
        )
        self.row_mixer = RowStage(
            embed_dim=config.embed_dim,
            num_layers=config.row_blocks,
            heads=config.row_heads,
            hidden_width=config.hidden_width,
            num_cls=config.row_cls_tokens,
            readout=False,
        )
        self.refinement_stage = RefinementStage(
            embed_dim=config.embed_dim,
            num_layers=config.column_blocks,
            heads=config.column_heads,
            hidden_width=config.hidden_width,
            num_inducing=config.column_inducing_points,
        )
        self.refinement_readout = RowStage(
            embed_dim=config.embed_dim,
            num_layers=config.row_blocks,
            heads=config.row_heads,
            hidden_width=config.hidden_width,
            num_cls=config.row_cls_tokens,
            readout=True,
        )

        out_dim = (
            int(config.native_classes)
            if config.is_classifier
            else int(config.regression_buckets)
        )
        self.icl_stage = ICLStage(
            model_type=config.model_type,
            out_dim=out_dim,
            width=config.icl_dim,
            num_layers=config.icl_blocks,
            heads=config.icl_heads,
            hidden_width=config.icl_feedforward,
            max_classes=int(config.native_classes or 0),
        )
        self.fusion_gate = nn.Linear(2 * config.embed_dim, config.embed_dim)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)

        if config.is_classifier:
            self.bucket_distribution = None
        else:
            self.bucket_distribution = BucketDistribution(
                spline_borders(int(config.regression_buckets))
            )

        for module in list(self.column_stage.modules()) + list(self.refinement_stage.modules()):
            if isinstance(module, BoundedQueryScaling):
                module.configure_gain(
                    config.column_gain_limit,
                    config.gain_reference_log_n,
                )
        for module in self.icl_stage.modules():
            if isinstance(module, BoundedQueryScaling):
                module.configure_gain(
                    config.icl_gain_limit,
                    config.gain_reference_log_n,
                )

        self._cache: Optional[ContextCache] = None

    @property
    def has_cache(self) -> bool:
        return self._cache is not None and not self._cache.is_empty()

    def clear_cache(self) -> None:
        """Drop the stored Context cache."""
        self._cache = None

    def _fuse(self, base: Tensor, refined: Tensor) -> Tensor:
        refined = refined.unflatten(
            -1,
            (self.config.row_cls_tokens, self.config.embed_dim),
        )
        delta = refined - base
        gate = torch.tanh(
            F.linear(
                torch.cat((base, delta), dim=-1),
                self.fusion_gate.weight.to(base),
                self.fusion_gate.bias.to(base),
            )
        )
        return (refined - gate * delta).flatten(-2)

    def _encode(
        self,
        X: Tensor,
        y: Tensor,
        ranks: Optional[Tensor],
        context_size: int,
        context_only: bool,
    ) -> Tuple[Tensor, StackCache, Optional[StackCache], Tensor]:
        features, column_cache, frequencies = self.column_stage.encode_context(
            X,
            y,
            ranks,
        )
        base, row_features = self.row_mixer.mix(features)
        if context_only:
            refined_features, refiner_cache = self.refinement_stage.encode_context(
                row_features
            )
        else:
            refined_features = self.refinement_stage(row_features, context_size)
            refiner_cache = None
        refined = self.refinement_readout.readout(refined_features)
        representations = self._fuse(base, refined)
        return representations, column_cache, refiner_cache, frequencies

    def build_cache(
        self,
        X: Tensor,
        y: Tensor,
        mode: str = "kv",
        ranks: Optional[Tensor] = None,
    ) -> ContextCache:
        """Encode a labeled Context table into reusable cache."""
        if mode not in ("kv", "repr"):
            raise ValueError(f"Unsupported cache mode {mode!r}; expected 'kv' or 'repr'.")
        context_size = y.shape[1]
        representations, column_cache, refiner_cache, frequencies = self._encode(
            X,
            y,
            ranks,
            context_size,
            context_only=True,
        )
        if column_cache is None:
            raise ValueError(
                "Caching is unavailable for datasets that require mixed-radix labels."
            )

        n_classes = (
            int(torch.unique(y[0]).numel())
            if self.model_type == "classifier"
            else 0
        )
        if mode == "kv":
            _, icl_cache = self.icl_stage.encode_context(representations, y)
            context_rows = None
        else:
            icl_cache = None
            context_rows = self.icl_stage.prepare_context_rows(representations, y)

        return ContextCache(
            column=column_cache,
            refiner=refiner_cache,
            frequencies=frequencies,
            context_size=context_size,
            n_classes=n_classes,
            mode=mode,
            icl=icl_cache,
            context_rows=context_rows,
            context_shape=tuple(X.shape),
        )

    def _query_representations(
        self,
        X: Tensor,
        cache: ContextCache,
        ranks: Optional[Tensor],
    ) -> Tensor:
        features = self.column_stage.decode_query(
            X,
            cache.frequencies,
            cache.column,
            ranks,
        )
        base, row_features = self.row_mixer.mix(features)
        refined_features = self.refinement_stage.decode_query(
            row_features,
            cache.refiner,
        )
        refined = self.refinement_readout.readout(refined_features)
        return self._fuse(base, refined)

    def predict_query(
        self,
        X: Tensor,
        cache: Optional[ContextCache] = None,
        ranks: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict query rows from stored Context cache."""
        cache = cache if cache is not None else self._cache
        if cache is None or cache.is_empty():
            raise RuntimeError("No valid Context cache is available.")
        representations = self._query_representations(X, cache, ranks)

        if cache.mode == "kv":
            return self.icl_stage.decode_query(
                representations,
                cache.icl,
                cache.n_classes,
            )

        full = torch.cat(
            (cache.context_rows.to(representations.device), representations),
            dim=1,
        )
        return self.icl_stage.predict_from_context_rows(
            full,
            cache.context_size,
            cache.n_classes,
        )

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        ranks: Optional[Tensor] = None,
        temperature: float = 1.0,
        n_classes: Optional[int] = None,
    ) -> Tensor:
        """Return logits for a batch of tables."""
        context_size = y_train.shape[1]
        features = self.column_stage(X, y_train, ranks)
        base, row_features = self.row_mixer.mix(features)
        refined_features = self.refinement_stage(row_features, context_size)
        refined = self.refinement_readout.readout(refined_features)
        representations = self._fuse(base, refined)

        if self.model_type == "regressor":
            return self.icl_stage.predict_standard(representations, y_train)

        n_classes = int(
            n_classes
            if n_classes is not None
            else torch.unique(y_train[0]).numel()
        )
        if n_classes <= self.config.native_classes:
            return self.icl_stage.predict_standard(
                representations,
                y_train,
                n_classes=n_classes,
            )

        outputs = []
        for table, labels in zip(representations, y_train):
            tree = HierarchicalReadout(
                self.config.native_classes,
                num_classes=n_classes,
            )
            tree.fit(table[:context_size], labels)
            probabilities = tree.predict(
                self.icl_stage.predict_standard,
                table[context_size:],
                _HIERARCHICAL_NODE_TEMPERATURE,
            )
            outputs.append(probabilities)
        probabilities = torch.stack(outputs, dim=0)
        return temperature * torch.log(probabilities + 1e-6)

    def predict_stats(
        self,
        X: Tensor,
        y_train: Tensor,
        output_type: Union[str, List[str]] = "mean",
        levels: Optional[List[float]] = None,
        ranks: Optional[Tensor] = None,
    ) -> Union[Tensor, dict]:
        """Summarize regression predictions."""
        if self.bucket_distribution is None:
            raise RuntimeError("predict_stats is only available for regression models.")
        raw = self.forward(X, y_train, ranks=ranks)
        return self.bucket_distribution.summarize(raw, output_type, levels)

    def store_cache(
        self,
        X_context: Tensor,
        y_context: Tensor,
        mode: str = "kv",
        ranks: Optional[Tensor] = None,
    ) -> ContextCache:
        """Build and retain Context cache."""
        self._cache = self.build_cache(X_context, y_context, mode, ranks)
        return self._cache

    def predict_stats_with_cache(
        self,
        X_query: Tensor,
        cache: Optional[ContextCache] = None,
        output_type: Union[str, List[str]] = "mean",
        levels: Optional[List[float]] = None,
        ranks: Optional[Tensor] = None,
    ) -> Union[Tensor, dict]:
        """Summarize regression predictions from stored Context cache."""
        if self.bucket_distribution is None:
            raise RuntimeError(
                "predict_stats_with_cache is only available for regression models."
            )
        raw = self.predict_query(X_query, cache=cache, ranks=ranks)
        return self.bucket_distribution.summarize(raw, output_type, levels)
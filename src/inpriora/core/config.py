"""Model configuration stored alongside converted weights."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and task configuration for :class:`InprioraModel`."""

    format_version: int = 1
    architecture: str = "v1"
    model_type: str = "classifier"

    embed_dim: int = 128
    column_blocks: int = 2
    column_heads: int = 8
    column_inducing_points: int = 128
    row_blocks: int = 3
    row_heads: int = 8
    row_cls_tokens: int = 4
    icl_blocks: int = 24
    icl_heads: int = 8
    ff_factor: float = 4.0

    native_classes: Optional[int] = None
    regression_buckets: Optional[int] = None

    column_gain_limit: float = 4.0
    icl_gain_limit: float = 6.7415
    gain_reference_log_n: float = 7.761488250497735
    rank_channel: bool = True

    @property
    def is_classifier(self) -> bool:
        return self.model_type == "classifier"

    @property
    def hidden_width(self) -> int:
        width = self.embed_dim * self.ff_factor
        return int(round(width))

    @property
    def icl_dim(self) -> int:
        return self.embed_dim * self.row_cls_tokens

    @property
    def icl_feedforward(self) -> int:
        return self.hidden_width * self.row_cls_tokens

    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is inconsistent."""
        if self.format_version != 1:
            raise ValueError(f"Unsupported format_version {self.format_version!r}.")
        if self.architecture != "v1":
            raise ValueError(f"Unsupported architecture {self.architecture!r}; expected 'v1'.")
        if self.model_type not in ("classifier", "regressor"):
            raise ValueError(f"Unsupported model_type {self.model_type!r}.")

        positive_ints = (
            self.embed_dim,
            self.column_blocks,
            self.column_heads,
            self.column_inducing_points,
            self.row_blocks,
            self.row_heads,
            self.row_cls_tokens,
            self.icl_blocks,
            self.icl_heads,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("Model dimensions and block counts must be positive.")

        width = self.embed_dim * self.ff_factor
        if not math.isfinite(width) or width <= 0 or not float(width).is_integer():
            raise ValueError("ff_factor must produce a positive integral hidden width.")
        if self.embed_dim % self.column_heads:
            raise ValueError("embed_dim must be divisible by column_heads.")
        if self.icl_dim % self.icl_heads:
            raise ValueError("icl_dim must be divisible by icl_heads.")

        if self.is_classifier:
            if not self.native_classes or self.native_classes <= 0:
                raise ValueError("Classifier checkpoints require native_classes > 0.")
            if self.regression_buckets not in (None, 0):
                raise ValueError("Classifier checkpoints must not define regression_buckets.")
        else:
            if not self.regression_buckets or self.regression_buckets < 2:
                raise ValueError("Regression checkpoints require regression_buckets >= 2.")
            if self.native_classes not in (None, 0):
                raise ValueError("Regression checkpoints must not define native_classes.")

        for limit in (self.column_gain_limit, self.icl_gain_limit):
            if not math.isfinite(limit) or limit <= 2:
                raise ValueError("Attention gain limits must be finite and greater than 2.")
        if not math.isfinite(self.gain_reference_log_n) or self.gain_reference_log_n <= 0:
            raise ValueError("gain_reference_log_n must be positive and finite.")

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        """Build a configuration from a mapping, ignoring unknown keys."""
        known = {field.name for field in fields(cls)}
        config = cls(**{key: value for key, value in values.items() if key in known})
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)
"""RMS normalization module."""

from __future__ import annotations

from torch import Tensor, nn

from .ops import rms_norm

__all__ = ["RMSNorm", "rms_norm"]


class RMSNorm(nn.RMSNorm):
    """RMSNorm with optional affine scaling."""

    def __init__(self, width: int, affine: bool = True) -> None:
        super().__init__(width, eps=1e-6, elementwise_affine=affine)

    def forward(self, value: Tensor) -> Tensor:
        return rms_norm(value, self.weight, self.eps)
"""Prediction heads."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .normalize import RMSNorm


class OneHotAndLinear(nn.Linear):
    """One-hot encode integer labels and project them in one operation."""

    def __init__(self, num_classes: int, embed_dim: int) -> None:
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, src: Tensor) -> Tensor:
        one_hot = F.one_hot(src.long(), self.num_classes).to(src.dtype)
        return F.linear(one_hot.float(), self.weight, self.bias)


class ClassificationHead(nn.Sequential):
    """Two-layer MLP classification head."""

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, out_dim),
        )


class RegressionHead(nn.Module):
    """Residual pre-head followed by a numeric bucket projection."""

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.hidden = nn.Linear(d_model, d_model * 2, bias=False)
        self.output = nn.Linear(d_model * 2, d_model, bias=False)
        nn.init.zeros_(self.output.weight)
        self.output_projection = nn.Linear(d_model, out_dim)

    def forward(self, src: Tensor) -> Tensor:
        residual = self.output(F.gelu(self.hidden(self.norm(src))))
        return self.output_projection(src + residual)
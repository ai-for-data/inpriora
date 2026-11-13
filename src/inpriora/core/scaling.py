"""Bounded query-aware attention scaling."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _log_length(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(math.log(max(n, 1)), device=device, dtype=dtype)


class BoundedQueryScaling(nn.Module):
    """Bounded per-head length gain with elementwise query modulation."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_hidden: int = 64,
        *,
        limit: float = 8.0,
        reference_log_n: float = 1.0,
    ) -> None:
        super().__init__()
        self.raw_scale = nn.Parameter(torch.empty(num_heads))
        self.query_mlp = nn.Sequential(
            nn.Linear(head_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, head_dim),
        )
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)
        self.configure_gain(limit, reference_log_n)

    def configure_gain(self, limit: float, reference_log_n: float) -> None:
        """Set the gain ceiling and the reference log-context size."""
        if not math.isfinite(limit) or limit <= 2:
            raise ValueError("The attention gain limit must be finite and greater than 2.")
        if not math.isfinite(reference_log_n) or reference_log_n <= 0:
            raise ValueError("reference_log_n must be positive and finite.")
        self.limit = float(limit)
        self.reference_log_n = float(reference_log_n)
        nn.init.constant_(self.raw_scale, 0.5 * math.log(2 / (limit - 2)))

    def forward(self, query: Tensor, n: int) -> Tensor:
        """Scale ``(..., heads, seq, head_dim)`` queries."""
        raw = self.raw_scale
        if raw.dtype in (torch.float16, torch.bfloat16):
            raw = raw.float()

        coefficient = (self.limit / (4.0 * self.reference_log_n)) * (1.0 + raw.tanh())
        length_gain = coefficient.to(query.dtype).view(1, -1, 1, 1) * _log_length(
            n,
            query.device,
            query.dtype,
        )
        content_gain = 1.0 + self.query_mlp(query).tanh()
        return query * (length_gain * content_gain)
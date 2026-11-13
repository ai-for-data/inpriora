"""Grouped Fourier value encoding."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ops import ieee_matmul
from .precision import no_autocast

GROUP_SIZE = 3
GROUP_STRIDE = 2
RANK_FREQUENCIES = 4


def _waveforms(values: Tensor, frequencies: Tensor) -> Tensor:
    cycles = (frequencies / (2.0 * math.pi)).ceil().clamp_min(1.0)
    period = (2.0 * math.pi) * cycles / frequencies
    phase = torch.fmod(values.unsqueeze(-1), period) * frequencies
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def _rank_waveforms(ranks: Tensor) -> Tensor:
    levels = torch.arange(
        1,
        RANK_FREQUENCIES + 1,
        device=ranks.device,
        dtype=ranks.dtype,
    )
    phase = ranks.unsqueeze(-1) * (math.pi * levels)
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class GroupWindows:
    """Fixed width-3, stride-2 windows over the feature axis."""

    size = GROUP_SIZE
    stride = GROUP_STRIDE

    @staticmethod
    def split(values: Tensor, padding: float = 0.0) -> Tensor:
        width = values.shape[-1]
        groups = max(1, width // 2)
        pad = 2 * (groups - 1) + GROUP_SIZE - width
        return F.pad(values, (0, pad), value=padding).unfold(
            -1,
            GROUP_SIZE,
            GROUP_STRIDE,
        )

    @staticmethod
    def mask(values: Tensor) -> Tensor:
        physical = torch.ones(
            values.shape[0],
            values.shape[-1],
            device=values.device,
            dtype=torch.bool,
        )
        return GroupWindows.split(physical)

    @staticmethod
    def group_mask(values: Tensor) -> Tensor:
        return GroupWindows.mask(values).any(dim=-1)


class FrequencyBank(nn.Module):
    """Learned per-column ordered Fourier frequencies."""

    def __init__(
        self,
        num_frequencies: int = 32,
        min_frequency: float = 0.01,
        max_frequency: float = 10.0,
        warp_modes: int = 8,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if num_frequencies < 4:
            raise ValueError("num_frequencies must be at least 4.")
        if not 1 <= warp_modes <= min(8, num_frequencies - 1):
            raise ValueError("warp_modes must be in [1, min(8, num_frequencies - 1)].")
        if not 0 < min_frequency < max_frequency:
            raise ValueError("Frequency bounds must satisfy 0 < min < max.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        self.num_frequencies = num_frequencies
        self.chunk_size = chunk_size

        anchors = torch.logspace(
            math.log10(min_frequency),
            math.log10(max_frequency),
            num_frequencies,
            dtype=torch.float32,
        )
        self.register_buffer("descriptor_frequencies", anchors)
        self.register_buffer("anchor_log_frequencies", anchors.log())
        positions = torch.arange(num_frequencies - 1, dtype=torch.float32) + 0.5
        modes = torch.arange(1, warp_modes + 1, dtype=torch.float32).unsqueeze(-1)
        self.register_buffer(
            "warp_basis",
            F.normalize(
                torch.cos(math.pi * modes * positions / (num_frequencies - 1)),
                dim=-1,
            ),
        )

        self.coefficient_weight = nn.Parameter(
            torch.zeros(warp_modes, 2 * num_frequencies)
        )
        self.coefficient_bias = nn.Parameter(torch.zeros(warp_modes))

    @torch.no_grad()
    def _summarize(self, context: Tensor) -> Tuple[Tensor, Tensor]:
        batch, context_size, width = context.shape
        feature_sum = torch.zeros(
            batch,
            width,
            2 * self.num_frequencies,
            device=context.device,
            dtype=torch.float32,
        )
        valid_count = torch.zeros(
            batch,
            width,
            device=context.device,
            dtype=torch.long,
        )
        for start in range(0, context_size, self.chunk_size):
            values = context[:, start : start + self.chunk_size].transpose(1, 2)
            valid = ~values.isnan()
            with no_autocast(context.device):
                safe = values.float().masked_fill(~valid, 0.0)
                features = _waveforms(safe, self.descriptor_frequencies.float())
                features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
            feature_sum.add_(features.sum(dim=-2))
            valid_count.add_(valid.sum(dim=-1))
        return feature_sum, valid_count

    def compute(self, context: Tensor) -> Tensor:
        """Return per-column frequencies of shape ``(batch, width, F)``."""
        feature_sum, valid_count = self._summarize(context)
        with no_autocast(context.device), ieee_matmul():
            coefficients = F.linear(
                feature_sum.float(),
                self.coefficient_weight.float(),
            )
            coefficients = (
                coefficients
                / valid_count.clamp_min(1).unsqueeze(-1).float()
                + self.coefficient_bias.float()
            )
            coefficients = coefficients.masked_fill(
                (valid_count == 0).unsqueeze(-1),
                0.0,
            )
            logits = 4.0 * torch.tanh(
                F.linear(coefficients, self.warp_basis.float().T) / 4.0
            )
            gaps = 0.1 / (self.num_frequencies - 1) + 0.9 * torch.softmax(
                logits,
                dim=-1,
            )
            locations = gaps[..., :-1].cumsum(dim=-1)
            log_min = self.anchor_log_frequencies[0]
            log_span = self.anchor_log_frequencies[-1] - log_min
            interior = torch.exp(log_min + locations * log_span)
            first = self.descriptor_frequencies[:1].float().expand_as(
                interior[..., :1]
            )
            last = self.descriptor_frequencies[-1:].float().expand_as(
                interior[..., :1]
            )
            frequencies = torch.cat((first, interior, last), dim=-1)
            return torch.where(
                (valid_count == 0).unsqueeze(-1),
                self.descriptor_frequencies.float(),
                frequencies,
            )


class ValueEmbedding(nn.Module):
    """Encode grouped values with frequencies derived from the Context."""

    def __init__(
        self,
        group_size: int = GROUP_SIZE,
        embed_dim: int = 128,
        rank_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.frequency_bank = FrequencyBank()
        self.num_frequencies = self.frequency_bank.num_frequencies
        self.output = nn.Linear(
            group_size * 2 * self.num_frequencies,
            embed_dim,
            bias=False,
        )
        self.missing = nn.Parameter(torch.empty(group_size, embed_dim))
        nn.init.trunc_normal_(self.missing, std=0.02)
        if rank_encoding:
            self.rank_linear = nn.Linear(
                group_size * 2 * RANK_FREQUENCIES,
                embed_dim,
                bias=False,
            )
        else:
            self.rank_linear = None

    def context_frequencies(self, context: Tensor) -> Tensor:
        """Compute frequencies from Context rows only."""
        return self.frequency_bank.compute(context)

    def encode(
        self,
        values: Tensor,
        frequencies: Tensor,
        ranks: Optional[Tensor] = None,
    ) -> Tensor:
        """Embed ``(batch, rows, features)`` values into grouped tokens."""
        if frequencies.shape != (
            values.shape[0],
            values.shape[-1],
            self.num_frequencies,
        ):
            raise ValueError("frequencies must match the input batch and width.")

        slots = GroupWindows.mask(values)
        grouped_values = GroupWindows.split(values)
        grouped_frequencies = GroupWindows.split(
            frequencies.transpose(-1, -2),
            padding=1.0,
        ).movedim(1, -1)

        missing = grouped_values.isnan() & slots.unsqueeze(1)
        valid = ~grouped_values.isnan() & slots.unsqueeze(1)

        with no_autocast(values.device), ieee_matmul():
            safe = grouped_values.float().masked_fill(~valid, 0.0)
            encoded = _waveforms(
                safe,
                grouped_frequencies.float().unsqueeze(1),
            )
            encoded = encoded.masked_fill(
                ~slots.unsqueeze(1).unsqueeze(-1),
                0.0,
            )

            rank_projection = None
            if self.rank_linear is not None:
                if ranks is None:
                    rank_features = torch.zeros(
                        (*grouped_values.shape, 2 * RANK_FREQUENCIES),
                        device=values.device,
                        dtype=torch.float32,
                    )
                else:
                    grouped_ranks = GroupWindows.split(ranks, padding=0)
                    rank_valid = slots.unsqueeze(1) & ~grouped_ranks.isnan()
                    safe_ranks = grouped_ranks.float().masked_fill(
                        ~rank_valid,
                        0.0,
                    )
                    rank_features = _rank_waveforms(safe_ranks).masked_fill(
                        ~rank_valid.unsqueeze(-1),
                        0.0,
                    )
                rank_projection = F.linear(
                    rank_features.flatten(start_dim=-2),
                    self.rank_linear.weight.float(),
                ).to(values.dtype)

        observed = self.output(
            encoded.to(values.dtype).flatten(start_dim=-2)
        )
        indicators = missing.to(observed.dtype) @ self.missing.to(observed.dtype)
        if rank_projection is None:
            return observed + indicators
        return observed + indicators + rank_projection
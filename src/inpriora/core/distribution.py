# SPDX-License-Identifier: Apache-2.0
# Adapted from TabPFN 3.5 (https://github.com/PriorLabs/TabPFN),
# Copyright (c) 2026 Prior Labs GmbH. Modified for inpriora.

"""Discrete bucket distribution for regression outputs."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import HalfNormal

_BORDER_XP = np.array(
    [0.0, 0.001, 0.004, 0.02, 0.04, 0.1, 0.2, 0.5, 0.8, 0.9, 0.96, 0.98, 0.996, 0.999, 1.0]
)
_BORDER_FP = np.array(
    [-128.0, -16.9, -13.0, -9.9, -8.47, -6.48, -4.4, 0.0, 4.4, 6.48, 8.47, 9.9, 13.0, 16.9, 128.0]
)


def spline_borders(num_buckets: int) -> Tensor:
    """Return monotonic borders in standard-normal space."""
    if num_buckets < 2:
        raise ValueError(f"num_buckets must be at least 2, got {num_buckets}.")
    positions = np.linspace(0.0, 1.0, num_buckets + 1)
    values = np.interp(positions, _BORDER_XP, _BORDER_FP)
    return torch.tensor(values, dtype=torch.float32)


class GridProjector:
    """Project bucket logits from one monotonic grid onto another."""

    def __init__(self, source_borders: Tensor, target_borders: Tensor) -> None:
        source = source_borders.detach().float()
        target = target_borders.detach().to(source.device).float()
        self.identity = torch.equal(source, target)
        if self.identity:
            return

        index = torch.searchsorted(source, target.contiguous()) - 1
        index = torch.where(target == source[0], torch.zeros_like(index), index)
        index = torch.where(
            target == source[-1],
            torch.full_like(index, source.numel() - 2),
            index,
        ).clamp(0, source.numel() - 2)
        widths = source[1:] - source[:-1]
        self.index = index
        self.fraction = (
            (target - source[index]) / widths[index]
        ).clamp(0.0, 1.0)
        self.zero_prior = target <= source[0]
        self.one_prior = target >= source[-1]
        self.zero_prior[0] = True
        self.one_prior[-1] = True
        self._cache: dict[str, tuple] = {}

    def _tensors(self, device: torch.device):
        key = str(device)
        cached = self._cache.get(key)
        if cached is None:
            cached = (
                self.index.to(device),
                self.fraction.to(device),
                self.zero_prior.to(device),
                self.one_prior.to(device),
            )
            self._cache[key] = cached
        return cached

    def project(
        self,
        logits: Tensor,
        chunk_elements: int = 20_000_000,
    ) -> Tensor:
        """Return bucket probabilities on the target grid."""
        if self.identity:
            return torch.softmax(logits, dim=-1)
        if chunk_elements <= 0:
            raise ValueError("chunk_elements must be positive.")

        index, fraction, zero_prior, one_prior = self._tensors(logits.device)
        batch_shape = logits.shape[:-1]
        flat = logits.reshape(-1, logits.shape[-1])
        out_size = index.numel()
        result = torch.empty(
            flat.shape[0],
            out_size - 1,
            dtype=logits.dtype,
            device=logits.device,
        )
        chunk = max(1, chunk_elements // out_size)
        for start in range(0, flat.shape[0], chunk):
            stop = start + chunk
            current = flat[start:stop]
            expanded = index.expand(current.shape[0], -1)
            probabilities = torch.softmax(current, dim=-1)
            cumulative = torch.cumsum(probabilities, dim=-1) - probabilities
            left = cumulative.gather(-1, expanded)
            left = left + probabilities.gather(-1, expanded) * fraction
            left = torch.where(zero_prior, torch.zeros_like(left), left)
            left = torch.where(one_prior, torch.ones_like(left), left)
            result[start:stop] = (left[..., 1:] - left[..., :-1]).clamp_min(0.0)
        return result.reshape(*batch_shape, out_size - 1)


def _half_normal(range_max: Tensor, mass_before: float = 0.5) -> HalfNormal:
    scale = range_max / HalfNormal(torch.tensor(1.0)).icdf(
        torch.tensor(mass_before)
    )
    return HalfNormal(scale)


class BucketDistribution(nn.Module):
    """Bucket distribution with half-normal outer tails."""

    def __init__(self, borders: Tensor) -> None:
        super().__init__()
        borders = borders.detach().float().contiguous()
        if borders.ndim != 1 or borders.numel() < 3:
            raise ValueError("borders must be one-dimensional with at least three values.")
        widths = borders[1:] - borders[:-1]
        if not bool(torch.all(widths >= 0.0)):
            raise ValueError("borders must be sorted.")
        if not bool(widths[0] > 0) or not bool(widths[-1] > 0):
            raise ValueError("Outer bucket widths must be strictly positive.")

        centers = borders[:-1] + widths / 2
        squares = (
            borders[:-1].square()
            + borders[1:].square()
            + borders[:-1] * borders[1:]
        ) / 3.0
        left_tail = _half_normal(widths[0])
        right_tail = _half_normal(widths[-1])
        centers = centers.clone()
        centers[0] = -left_tail.mean + borders[1]
        centers[-1] = right_tail.mean + borders[-2]
        squares = squares.clone()
        squares[0] = left_tail.variance + centers[0].square()
        squares[-1] = right_tail.variance + centers[-1].square()

        self.register_buffer("borders", borders)
        self.register_buffer("_widths", widths, persistent=False)
        self.register_buffer("_centers", centers, persistent=False)
        self.register_buffer("_squares", squares, persistent=False)

    @property
    def num_bars(self) -> int:
        return self.borders.numel() - 1

    @property
    def num_buckets(self) -> int:
        return self.num_bars

    def _probabilities(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits.float(), dim=-1)

    def mean(self, logits: Tensor) -> Tensor:
        """Expected value of the distribution."""
        return self._probabilities(logits) @ self._centers.to(logits.device)

    def variance(self, logits: Tensor) -> Tensor:
        """Variance of the distribution."""
        mean = self.mean(logits)
        second = self._probabilities(logits) @ self._squares.to(logits.device)
        return second - mean.square()

    def mode(self, logits: Tensor) -> Tensor:
        """Center of the highest-density bucket."""
        density = self._probabilities(logits) / self._widths.to(logits.device)
        return self._centers.to(logits.device)[density.argmax(-1)]

    def quantile(
        self,
        logits: Tensor,
        levels: Union[float, Tensor],
    ) -> Tensor:
        """Inverse CDF at the requested probability levels."""
        probabilities = self._probabilities(logits)
        cumulative = torch.cumsum(probabilities, dim=-1)
        level_tensor = torch.as_tensor(
            levels,
            device=logits.device,
            dtype=torch.float32,
        )
        scalar = level_tensor.ndim == 0
        query = level_tensor.reshape(-1).expand(
            cumulative.shape[:-1] + (level_tensor.reshape(-1).shape[-1],)
        ).contiguous()

        index = torch.searchsorted(cumulative.contiguous(), query, right=False)
        index = index.clamp(0, self.num_bars - 1)
        cumulative_left = F.pad(cumulative[..., :-1], (1, 0))
        remainder = query - cumulative_left.gather(-1, index)
        probability_at_index = probabilities.gather(-1, index)
        borders = self.borders.to(logits.device)
        left = borders[..., index]
        right = borders[..., (index + 1).clamp_max(self.num_bars)]
        values = left + (right - left) * remainder / probability_at_index
        return values.squeeze(-1) if scalar else values

    def median(self, logits: Tensor) -> Tensor:
        """Median of the distribution."""
        return self.quantile(logits, 0.5)

    def summarize(
        self,
        logits: Tensor,
        output_type: Union[str, List[str]] = "mean",
        levels: Optional[List[float]] = None,
    ) -> Union[Tensor, dict]:
        """Summarize bucket logits into the requested statistics."""
        requested = [output_type] if isinstance(output_type, str) else list(output_type)
        results = {}
        for statistic in requested:
            if statistic == "mean":
                results[statistic] = self.mean(logits)
            elif statistic == "variance":
                results[statistic] = self.variance(logits)
            elif statistic == "median":
                results[statistic] = self.median(logits)
            elif statistic == "mode":
                results[statistic] = self.mode(logits)
            elif statistic == "quantiles":
                probability_levels = (
                    levels
                    if levels is not None
                    else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
                results[statistic] = self.quantile(logits, probability_levels)
            else:
                raise ValueError(f"Unsupported regression output type {statistic!r}.")
        return results[requested[0]] if len(requested) == 1 else results
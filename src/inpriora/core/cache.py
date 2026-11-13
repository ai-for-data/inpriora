"""Context cache objects for cached inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class LayerCache:
    """Projected key and value tensors for one attention layer."""

    keys: Tensor
    values: Tensor

    def slice_batch(self, start: int, end: int) -> "LayerCache":
        return LayerCache(self.keys[start:end], self.values[start:end])

    def to(self, device, dtype=None) -> "LayerCache":
        return LayerCache(
            self.keys.to(device=device, dtype=dtype),
            self.values.to(device=device, dtype=dtype),
        )

    @staticmethod
    def concat(caches: List["LayerCache"], dim: int = 0) -> "LayerCache":
        return LayerCache(
            torch.cat([cache.keys for cache in caches], dim=dim),
            torch.cat([cache.values for cache in caches], dim=dim),
        )


@dataclass
class StackCache:
    """Per-layer caches stacked along a leading layer axis."""

    keys: Tensor
    values: Tensor

    @classmethod
    def from_layers(cls, layers: List[LayerCache]) -> "StackCache":
        if not layers:
            raise ValueError("At least one layer cache is required.")
        return cls(
            keys=torch.stack([layer.keys for layer in layers], dim=0),
            values=torch.stack([layer.values for layer in layers], dim=0),
        )

    def __len__(self) -> int:
        return self.keys.shape[0]

    def __getitem__(self, index: int) -> LayerCache:
        return LayerCache(self.keys[index], self.values[index])

    def is_empty(self) -> bool:
        return self.keys.numel() == 0

    def slice_batch(self, start: int, end: int) -> "StackCache":
        return StackCache(self.keys[:, start:end], self.values[:, start:end])

    def to(self, device, dtype=None) -> "StackCache":
        return StackCache(
            self.keys.to(device=device, dtype=dtype),
            self.values.to(device=device, dtype=dtype),
        )

    @staticmethod
    def concat(caches: List["StackCache"], dim: int = 1) -> "StackCache":
        return StackCache(
            torch.cat([cache.keys for cache in caches], dim=dim),
            torch.cat([cache.values for cache in caches], dim=dim),
        )


@dataclass
class ContextCache:
    """Reusable state built from a labeled Context table."""

    column: StackCache
    refiner: StackCache
    frequencies: Tensor
    context_size: int
    n_classes: int
    mode: str
    icl: Optional[StackCache] = None
    context_rows: Optional[Tensor] = None
    context_shape: Tuple[int, int, int] = (0, 0, 0)

    def is_empty(self) -> bool:
        return self.column.is_empty()

    def slice_batch(self, start: int, end: int) -> "ContextCache":
        return ContextCache(
            column=self.column.slice_batch(start, end),
            refiner=self.refiner.slice_batch(start, end),
            frequencies=self.frequencies[start:end],
            context_size=self.context_size,
            n_classes=self.n_classes,
            mode=self.mode,
            icl=self.icl.slice_batch(start, end) if self.icl is not None else None,
            context_rows=(
                self.context_rows[start:end]
                if self.context_rows is not None
                else None
            ),
            context_shape=(end - start, self.context_shape[1], self.context_shape[2]),
        )

    def to(self, device, dtype=None) -> "ContextCache":
        return ContextCache(
            column=self.column.to(device, dtype),
            refiner=self.refiner.to(device, dtype),
            frequencies=self.frequencies.to(device=device),
            context_size=self.context_size,
            n_classes=self.n_classes,
            mode=self.mode,
            icl=self.icl.to(device, dtype) if self.icl is not None else None,
            context_rows=(
                self.context_rows.to(device=device, dtype=dtype)
                if self.context_rows is not None
                else None
            ),
            context_shape=self.context_shape,
        )

    @staticmethod
    def concat(caches: List["ContextCache"], dim: int = 0) -> "ContextCache":
        if not caches:
            raise ValueError("At least one context cache is required.")
        if any(cache.mode != caches[0].mode for cache in caches):
            raise ValueError("Cannot concatenate context caches with different modes.")
        if any(cache.context_size != caches[0].context_size for cache in caches):
            raise ValueError("Cannot concatenate context caches with different context sizes.")
        if any(cache.n_classes != caches[0].n_classes for cache in caches):
            raise ValueError("Cannot concatenate context caches with different class counts.")
        if any(cache.context_shape[1:] != caches[0].context_shape[1:] for cache in caches):
            raise ValueError("Cannot concatenate context caches with different shapes.")

        has_icl = [cache.icl for cache in caches if cache.icl is not None]
        if has_icl and len(has_icl) != len(caches):
            raise ValueError("Cannot concatenate partially populated ICL caches.")
        has_rows = [cache.context_rows for cache in caches if cache.context_rows is not None]
        if has_rows and len(has_rows) != len(caches):
            raise ValueError("Cannot concatenate partially populated context rows.")

        return ContextCache(
            column=StackCache.concat([cache.column for cache in caches], dim=dim + 1),
            refiner=StackCache.concat([cache.refiner for cache in caches], dim=dim + 1),
            frequencies=torch.cat([cache.frequencies for cache in caches], dim=dim),
            context_size=caches[0].context_size,
            n_classes=caches[0].n_classes,
            mode=caches[0].mode,
            icl=(
                StackCache.concat(has_icl, dim=dim + 1) if has_icl else None
            ),
            context_rows=(
                torch.cat(has_rows, dim=dim) if has_rows else None
            ),
            context_shape=(
                sum(cache.context_shape[0] for cache in caches),
                caches[0].context_shape[1],
                caches[0].context_shape[2],
            ),
        )

    def size_mb(self) -> int:
        """Return the approximate cache footprint in mebibytes."""
        total = 0
        for cache in (self.column, self.refiner, self.icl):
            if cache is not None:
                total += cache.keys.numel() * cache.keys.element_size()
                total += cache.values.numel() * cache.values.element_size()
        for tensor in (self.frequencies, self.context_rows):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total // (1024 * 1024)


@dataclass
class ShardedContextCache:
    """Member-batched Context caches kept as shards.

    Ensemble members are encoded in batches during ``fit``. Keeping one cache
    per batch avoids copying every cache tensor into a single contiguous cache.
    Prediction batches normally align with the encoding batches, so slicing a
    shard returns a view instead of copying.
    """

    shards: List[ContextCache]

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("At least one cache shard is required.")
        first = self.shards[0]
        if any(cache.mode != first.mode for cache in self.shards):
            raise ValueError("Cannot mix cache shards with different modes.")
        if any(cache.context_size != first.context_size for cache in self.shards):
            raise ValueError("Cannot mix cache shards with different context sizes.")
        if any(cache.n_classes != first.n_classes for cache in self.shards):
            raise ValueError("Cannot mix cache shards with different class counts.")
        if any(cache.context_shape[1:] != first.context_shape[1:] for cache in self.shards):
            raise ValueError("Cannot mix cache shards with different shapes.")
        if any((cache.icl is None) != (first.icl is None) for cache in self.shards):
            raise ValueError("Cannot mix shards with and without ICL caches.")
        if any(
            (cache.context_rows is None) != (first.context_rows is None)
            for cache in self.shards
        ):
            raise ValueError("Cannot mix shards with and without context rows.")

    @property
    def mode(self) -> str:
        return self.shards[0].mode

    @property
    def context_size(self) -> int:
        return self.shards[0].context_size

    @property
    def n_classes(self) -> int:
        return self.shards[0].n_classes

    @property
    def context_shape(self) -> Tuple[int, int, int]:
        return (
            sum(cache.context_shape[0] for cache in self.shards),
            self.shards[0].context_shape[1],
            self.shards[0].context_shape[2],
        )

    def __len__(self) -> int:
        return self.context_shape[0]

    def is_empty(self) -> bool:
        return all(cache.is_empty() for cache in self.shards)

    def slice_batch(self, start: int, end: int) -> ContextCache:
        if start < 0 or end < start or end > len(self):
            raise IndexError("Cache batch slice is out of range.")
        pieces: List[ContextCache] = []
        offset = 0
        for shard in self.shards:
            size = shard.context_shape[0]
            lower = max(start, offset)
            upper = min(end, offset + size)
            if lower < upper:
                pieces.append(shard.slice_batch(lower - offset, upper - offset))
            offset += size
        if len(pieces) == 1:
            return pieces[0]
        return ContextCache.concat(pieces)

    def to(self, device, dtype=None) -> "ShardedContextCache":
        return ShardedContextCache(
            [cache.to(device, dtype) for cache in self.shards]
        )

    def size_mb(self) -> int:
        return sum(cache.size_mb() for cache in self.shards)
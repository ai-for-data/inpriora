"""Low-level numerical operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..kernels import triton_rmsnorm, triton_swiglu

#: Kernel limits: short per-row attention runs directly, and large batches are chunked.
_MAX_ATTENTION_BATCH = 32_768
_SMALL_ATTENTION_SEQ = 16


def _sdpa(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    batch = query.shape[0]
    if query.shape[-2] <= _SMALL_ATTENTION_SEQ and key.shape[-2] <= _SMALL_ATTENTION_SEQ:
        scale = query.shape[-1] ** -0.5
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1).to(value.dtype)
        return torch.matmul(weights, value)
    if batch <= _MAX_ATTENTION_BATCH:
        return F.scaled_dot_product_attention(query, key, value)
    outs = []
    for start in range(0, batch, _MAX_ATTENTION_BATCH):
        stop = min(start + _MAX_ATTENTION_BATCH, batch)
        key_chunk = key[start:stop] if key.shape[0] == batch else key
        value_chunk = value[start:stop] if value.shape[0] == batch else value
        outs.append(F.scaled_dot_product_attention(query[start:stop], key_chunk, value_chunk))
    return torch.cat(outs, dim=0)


def rms_norm(value: Tensor, weight: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
    """Normalize over the last axis with float32 statistics."""
    if not torch.is_grad_enabled():
        fused = triton_rmsnorm(value, weight, eps)
        if fused is not None:
            return fused
    if value.dtype in (torch.float16, torch.bfloat16):
        return F.rms_norm(
            value.float(),
            (value.size(-1),),
            None if weight is None else weight.float(),
            eps,
        ).to(value.dtype)
    return F.rms_norm(value, (value.size(-1),), weight, eps)


def swiglu(packed: Tensor) -> Tensor:
    """Apply ``value * silu(gate)`` to a packed ``(..., 2H)`` tensor."""
    if not torch.is_grad_enabled():
        fused = triton_swiglu(packed)
        if fused is not None:
            return fused
    value, gate = packed.chunk(2, dim=-1)
    return value * F.silu(gate)


def attend(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scaling=None,
) -> Tensor:
    """Apply head-wise RMS normalization and scaled dot-product attention."""
    output_shape = query.shape
    query = query.reshape(-1, *output_shape[-3:])
    key = key.reshape(-1, *key.shape[-3:])
    value = value.reshape(-1, *value.shape[-3:])

    query = rms_norm(query)
    key = rms_norm(key)
    if scaling is not None:
        query = scaling(query, key.shape[-2])
    return _sdpa(query, key, value).view(output_shape)


@contextmanager
def ieee_matmul() -> Iterator[None]:
    """Temporarily disable TF32 matmuls for exact fp32 statistics."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
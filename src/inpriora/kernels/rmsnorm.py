"""Optional fused RMSNorm kernel for half-precision CUDA inference."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Optional

import torch
from torch import Tensor

_Kernel = Callable[[Tensor, Optional[Tensor], float], Optional[Tensor]]


def _disabled() -> bool:
    return os.environ.get("INPRIORA_DISABLE_TRITON", "").lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=1)
def _kernel() -> Optional[_Kernel]:
    if _disabled():
        return None
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    if int(triton.__version__.split(".")[0]) < 3:
        return None

    @triton.jit
    def _rmsnorm_cast_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        rows,
        channels: tl.constexpr,
        eps: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
        mask = row < rows
        col = tl.arange(0, channels)
        value = tl.load(
            x_ptr + row[:, None] * channels + col[None, :],
            mask=mask[:, None],
            other=0,
        ).to(tl.float32)
        variance = tl.sum(value * value, 1) / channels
        value = value * tl.rsqrt(variance + eps)[:, None]
        if w_ptr is not None:
            value = value * tl.load(w_ptr + col).to(tl.float32)
        tl.store(
            out_ptr + row[:, None] * channels + col[None, :],
            value,
            mask=mask[:, None],
        )

    def rmsnorm_cast(value: Tensor, weight: Optional[Tensor], eps: float) -> Optional[Tensor]:
        if value.numel() == 0:
            return None
        if weight is not None and (
            weight.device != value.device
            or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or weight.dim() != 1
        ):
            return None
        contiguous = value.contiguous()
        channels = contiguous.size(-1)
        if channels < 16 or channels & (channels - 1):
            return None
        flat = contiguous.view(-1, channels)
        rows = flat.size(0)
        grid = ((rows + 7) // 8,)
        out = torch.empty_like(flat)
        with torch.cuda.device(value.device):
            _rmsnorm_cast_kernel[grid](
                flat,
                weight,
                out,
                rows,
                channels,
                eps,
                8,
                num_warps=4,
            )
        return out.view(*value.shape)

    return rmsnorm_cast


def triton_rmsnorm(value: Tensor, weight: Optional[Tensor], eps: float) -> Optional[Tensor]:
    """Return a fused RMSNorm result, or ``None`` when the kernel cannot run."""
    builder = _kernel()
    if builder is None:
        return None
    if (
        not value.is_cuda
        or value.dtype not in (torch.float16, torch.bfloat16)
        or value.size(-1) < 16
        or value.size(-1) & (value.size(-1) - 1)
    ):
        return None
    return builder(value, weight, eps)
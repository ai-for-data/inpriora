"""Optional fused SwiGLU kernel for half-precision CUDA inference."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Optional

import torch
from torch import Tensor

_Kernel = Callable[[Tensor], Optional[Tensor]]


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

    @triton.jit
    def _swiglu_kernel(
        h_ptr,
        out_ptr,
        rows,
        hidden: tl.constexpr,
        block_h: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
        col = tl.arange(0, block_h)
        mask = (row < rows)[:, None] & (col < hidden)[None, :]
        base = row[:, None] * (2 * hidden)
        value = tl.load(h_ptr + base + col[None, :], mask=mask, other=0).to(tl.float32)
        gate = tl.load(h_ptr + base + hidden + col[None, :], mask=mask, other=0).to(tl.float32)
        product = value * gate * tl.sigmoid(gate)
        tl.store(out_ptr + row[:, None] * hidden + col[None, :], product, mask=mask)

    def swiglu_fused(packed: Tensor) -> Optional[Tensor]:
        if packed.size(-1) % 2 or packed.size(-1) < 2:
            return None
        hidden = packed.size(-1) // 2
        if hidden > 4096:
            return None
        contiguous = packed.contiguous()
        flat = contiguous.view(-1, 2 * hidden)
        rows = flat.size(0)
        block_h = max(2, 1 << (hidden - 1).bit_length())
        grid = ((rows + 7) // 8,)
        out = torch.empty((rows, hidden), device=flat.device, dtype=flat.dtype)
        with torch.cuda.device(flat.device):
            _swiglu_kernel[grid](flat, out, rows, hidden, block_h, 8, num_warps=4)
        return out.view(*packed.shape[:-1], hidden)

    return swiglu_fused


def triton_swiglu(packed: Tensor) -> Optional[Tensor]:
    """Return a fused ``value * silu(gate)`` result, or ``None``."""
    if torch.is_grad_enabled():
        return None
    builder = _kernel()
    if builder is None:
        return None
    if not packed.is_cuda or packed.dtype not in (torch.float16, torch.bfloat16):
        return None
    return builder(packed)
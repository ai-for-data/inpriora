"""Residual-stream precision control for mixed-precision inference."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

import torch
from torch import Tensor

_stream_depth: ContextVar[int] = ContextVar("inpriora_fp16_stream", default=0)


@contextmanager
def fp16_stream(enabled: bool = True) -> Iterator[None]:
    """Mark a CUDA mixed-precision forward pass.

    While active, the column embedding emits its residual stream at float16 so
    subsequent blocks keep the stream in half precision.
    """
    if not enabled:
        yield
        return
    token = _stream_depth.set(_stream_depth.get() + 1)
    try:
        yield
    finally:
        _stream_depth.reset(token)


def stream_active() -> bool:
    """Return whether a mixed-precision stream scope is active."""
    return _stream_depth.get() > 0


def maybe_cast_stream(value: Tensor) -> Tensor:
    """Cast the residual stream to float16 inside an active CUDA scope."""
    if stream_active() and value.is_cuda and not torch.is_grad_enabled():
        return value.to(torch.float16)
    return value


@contextmanager
def no_autocast(device: torch.device) -> Iterator[None]:
    """Disable autocast for exact-statistics computations."""
    with torch.autocast(device_type=device.type, enabled=False):
        yield


def autocast_dtype(device: torch.device, amp: bool) -> Optional[torch.dtype]:
    """Return the autocast dtype for a forward pass, or ``None``."""
    if amp and device.type == "cuda":
        return torch.float16
    return None
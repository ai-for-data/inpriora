"""Optional fused CUDA kernels with eager fallbacks."""

from .rmsnorm import triton_rmsnorm
from .swiglu import triton_swiglu

__all__ = ["triton_rmsnorm", "triton_swiglu"]
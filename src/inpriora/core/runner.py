"""Inference execution helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Union

import torch
from torch import Tensor

from .precision import fp16_stream

_ROW_BLOCK_START = 4096


class InferenceRunner:
    """Wrap forward passes with no-grad, mixed precision, and OOM recovery."""

    def __init__(self, device: torch.device, amp: bool = True) -> None:
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")

    @contextmanager
    def context(self) -> Iterator[None]:
        """Enter the inference execution context."""
        with torch.inference_mode():
            if self.amp:
                with torch.autocast("cuda", dtype=torch.float16), fp16_stream(True):
                    yield
            else:
                yield

    def run_row_blocks(
        self,
        forward: Callable[[int, int], Union[Tensor, Dict[str, Tensor]]],
        n_rows: int,
        dim: int = 1,
        start_block: int = _ROW_BLOCK_START,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Run a forward over row blocks, halving the block on device OOM.

        Parameters
        ----------
        forward : callable
            Called as ``forward(start, stop)``.
        n_rows : int
            Number of rows to process.
        dim : int, default=1
            Axis along which block outputs are concatenated.
        start_block : int, default=4096
            Initial block size on CUDA.

        Returns
        -------
        Tensor or dict
            Concatenated block outputs.
        """
        if n_rows <= 0:
            return forward(0, 0)

        block = n_rows if self.device.type != "cuda" else min(start_block, n_rows)
        pieces: List[Union[Tensor, Dict[str, Tensor]]] = []
        position = 0
        while position < n_rows:
            stop = min(position + block, n_rows)
            try:
                pieces.append(forward(position, stop))
            except torch.OutOfMemoryError:
                if block <= 1:
                    raise
                block = max(1, block // 2)
                continue
            position = stop

        first = pieces[0]
        if isinstance(first, dict):
            return {
                key: torch.cat([piece[key] for piece in pieces], dim=dim)
                for key in first
            }
        return torch.cat(pieces, dim=dim)
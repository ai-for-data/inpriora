"""Device resolution for inference."""

from __future__ import annotations

from typing import Optional, Union

import torch


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a device specification to CUDA or CPU.

    Parameters
    ----------
    device : str, torch.device, or None, default=None
        Device to use. ``None`` selects CUDA when available, otherwise CPU.

    Returns
    -------
    torch.device
        Concrete device.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device) if isinstance(device, str) else device
    if resolved.type not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported device type {resolved.type!r}; expected 'cuda' or 'cpu'.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return resolved
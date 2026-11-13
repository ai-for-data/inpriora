"""Version-tolerant input validation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_array


def check_features(X, *, dtype="numeric", copy=False) -> np.ndarray:
    """Validate a feature matrix while allowing NaN values."""
    kwargs = {"dtype": dtype, "copy": copy}
    try:
        return check_array(X, ensure_all_finite="allow-nan", **kwargs)
    except TypeError:
        return check_array(X, force_all_finite="allow-nan", **kwargs)
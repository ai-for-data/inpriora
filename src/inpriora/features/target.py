"""Target transforms for regression ensembles."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import optimize
from scipy.special import ndtri
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


def _yeojohnson_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    transformed = np.zeros_like(x, dtype=dtype)
    positive = x >= 0
    if abs(lmbda) < np.spacing(1.0):
        transformed[positive] = np.log1p(x[positive])
    else:
        transformed[positive] = np.expm1(
            lmbda * np.log1p(x[positive])
        ) / lmbda

    if abs(lmbda - 2) > np.spacing(1.0):
        transformed[~positive] = -np.expm1(
            (2 - lmbda) * np.log1p(-x[~positive])
        ) / (2 - lmbda)
    else:
        transformed[~positive] = -np.log1p(-x[~positive])
    return transformed


def _yeojohnson_inverse_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    inverse = np.zeros_like(x, dtype=dtype)
    positive = x >= 0
    log_max = np.log(np.finfo(dtype).max)
    max_argument = log_max - 4 * np.spacing(log_max)

    if abs(lmbda) < np.spacing(1.0):
        inverse[positive] = np.expm1(
            np.clip(x[positive], -max_argument, max_argument)
        )
    else:
        inverse[positive] = np.expm1(
            np.clip(
                np.log1p(x[positive] * lmbda) / lmbda,
                -max_argument,
                max_argument,
            )
        )

    if abs(lmbda - 2) > np.spacing(1.0):
        inverse[~positive] = -np.expm1(
            np.clip(
                np.log1p(-(2 - lmbda) * x[~positive]) / (2 - lmbda),
                -max_argument,
                max_argument,
            )
        )
    else:
        inverse[~positive] = -np.expm1(
            np.clip(-x[~positive], -max_argument, max_argument)
        )
    return inverse


def _yeojohnson_log_likelihood(lmbda: float, data: np.ndarray) -> np.ndarray:
    transformed = _yeojohnson_transform(data, lmbda)
    variance = transformed.var(axis=0)
    likelihood = np.empty_like(variance)
    tiny = variance < np.finfo(variance.dtype).tiny
    likelihood[tiny] = np.inf
    likelihood[~tiny] = -data.shape[0] / 2 * np.log(variance[~tiny])
    likelihood[~tiny] += (lmbda - 1) * (
        np.sign(data) * np.log1p(np.abs(data))
    ).sum(axis=0)
    return likelihood


def _yeojohnson_normmax(x: np.ndarray) -> float:
    if not np.all(np.isfinite(x)):
        raise ValueError("Yeo-Johnson target values must be finite.")
    if np.all(x == 0):
        return 1.0

    def objective(lmbda: float, data: np.ndarray) -> np.ndarray:
        likelihood = _yeojohnson_log_likelihood(lmbda, data)
        likelihood[np.isinf(likelihood)] = -np.inf
        return -likelihood

    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    log1p_max = np.log1p(20 * np.max(np.abs(x)))
    log_epsilon = np.log(np.finfo(dtype).eps)
    lower = (np.log(np.finfo(dtype).tiny) - log_epsilon) / 2 / log1p_max
    upper = (np.log(np.finfo(dtype).max) + log_epsilon) / 2 / log1p_max
    if np.all(x < 0):
        lower, upper = 2 - upper, 2 - lower
    elif np.any(x < 0):
        lower, upper = max(2 - upper, lower), min(2 - lower, upper)
    return float(
        optimize.fminbound(
            objective,
            lower,
            upper,
            args=(x,),
            xtol=1.48e-8,
        )
    )


class SafePowerTransformer(PowerTransformer):
    """Yeo-Johnson transformer with overflow-safe forward and inverse math."""

    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        return _yeojohnson_normmax(x[~np.isnan(x)])

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        return _yeojohnson_transform(x, lmbda)

    def _yeo_johnson_inverse_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        return _yeojohnson_inverse_transform(x, lmbda)


class RankGaussianTransformer(BaseEstimator, TransformerMixin):
    """Map the empirical target CDF to standard-normal scores.

    The inverse interpolates the sorted training targets as a function of the
    normal score and linearly extrapolates beyond the observed range.  Tail
    extrapolation keeps bucket borders strictly monotone instead of collapsing
    every extreme border onto the training minimum/maximum.
    """

    def __init__(self) -> None:
        self.y_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None
        self.sorted_: Optional[np.ndarray] = None
        self.n_ = 0
        self.lower_slope_ = 1.0
        self.upper_slope_ = 1.0

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        self.n_ = int(values.size)
        if values.size == 0:
            self.y_ = np.zeros(1, dtype=np.float64)
            self.scores_ = np.zeros(1, dtype=np.float64)
            self.sorted_ = np.zeros(1, dtype=np.float64)
            return self

        order = np.argsort(values, kind="mergesort")
        ordered = values[order]
        self.sorted_ = np.asarray(ordered, dtype=np.float64)
        # One score per distinct value, using the average 1-based rank of the
        # tie group so repeated targets share one lookup point.
        unique, starts, counts = np.unique(
            ordered, return_index=True, return_counts=True
        )
        average_rank = starts.astype(np.float64) + (counts + 1.0) / 2.0
        probabilities = (average_rank - 0.5) / values.size
        eps = np.finfo(np.float64).eps
        np.clip(probabilities, eps, 1.0 - eps, out=probabilities)
        scores = ndtri(probabilities)

        self.y_ = np.asarray(unique, dtype=np.float64)
        self.scores_ = np.asarray(scores, dtype=np.float64)
        self.lower_slope_ = self._tail_slope(lower=True)
        self.upper_slope_ = self._tail_slope(lower=False)
        return self

    def _tail_slope(self, lower: bool) -> float:
        y = self.y_
        scores = self.scores_
        if y.size < 2:
            return 1.0
        span = max(2, int(np.ceil(0.05 * y.size)))
        if lower:
            ys = y[:span]
            ss = scores[:span]
        else:
            ys = y[-span:]
            ss = scores[-span:]
        delta_score = float(ss[-1] - ss[0])
        delta_y = float(ys[-1] - ys[0])
        if not np.isfinite(delta_score) or abs(delta_score) < 1e-12:
            return 1.0
        if not np.isfinite(delta_y) or delta_y <= 0:
            return 1.0
        return delta_y / delta_score

    def transform(self, X):
        values = np.asarray(X, dtype=np.float64)
        output = np.full(values.shape, np.nan, dtype=np.float64)
        observed = np.isfinite(values)
        if self.sorted_ is None or not observed.any():
            return output
        query = values[observed]
        left = np.searchsorted(self.sorted_, query, side="left")
        right = np.searchsorted(self.sorted_, query, side="right")
        probabilities = (left + right).astype(np.float64) / (2.0 * self.n_)
        eps = np.finfo(np.float64).eps
        np.clip(probabilities, eps, 1.0 - eps, out=probabilities)
        output[observed] = ndtri(probabilities)
        return output

    def inverse_transform(self, X):
        values = np.asarray(X, dtype=np.float64)
        if self.y_ is None or self.y_.size == 0:
            return np.zeros_like(values)
        if self.y_.size == 1:
            return np.full_like(values, self.y_[0])
        output = np.interp(values, self.scores_, self.y_)
        lower = values < self.scores_[0]
        upper = values > self.scores_[-1]
        if lower.any():
            output[lower] = self.y_[0] + (
                values[lower] - self.scores_[0]
            ) * self.lower_slope_
        if upper.any():
            output[upper] = self.y_[-1] + (
                values[upper] - self.scores_[-1]
            ) * self.upper_slope_
        return output


TARGET_TRANSFORMS = {
    None: None,
    "none": None,
    "safepower": lambda: SafePowerTransformer(standardize=False),
    "rankgauss": RankGaussianTransformer,
    "rank_gauss": RankGaussianTransformer,
}


def make_target_transform(name: Optional[str]):
    """Build one independently fitted target transform, or ``None``."""
    if name not in TARGET_TRANSFORMS:
        raise ValueError(
            f"Unsupported target transform {name!r}; expected one of "
            f"{sorted(key for key in TARGET_TRANSFORMS if key is not None)}."
        )
    factory = TARGET_TRANSFORMS[name]
    if factory is None:
        return None
    return Pipeline(
        [
            ("power", factory()),
            ("standard", StandardScaler()),
        ]
    )


def inverse_transform_borders(
    borders: np.ndarray,
    transform: Pipeline,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Map model-space borders back to the shared target grid."""
    transformed = np.asarray(
        transform.inverse_transform(borders.reshape(-1, 1)).squeeze(),
        dtype=np.float64,
    )
    broken = ~np.isfinite(transformed) | (transformed > 1e3) | (transformed < -1e3)
    cancel_mask = None
    if broken.any():
        if broken.all():
            raise ValueError("Target transform mapped every border out of range.")
        transformed = transformed.copy()
        transitions_from_broken = np.where(broken[:-1] & ~broken[1:])[0]
        transitions_to_broken = np.where(~broken[:-1] & broken[1:])[0]
        if transitions_from_broken.size:
            first_valid = transitions_from_broken[0] + 1
            transformed[:first_valid] = transformed[first_valid]
            transformed[0] = transformed[1] - 1.0
        if transitions_to_broken.size:
            last_valid = transitions_to_broken[-1]
            transformed[last_valid + 1 :] = transformed[last_valid]
            transformed[-1] = transformed[-2] + 1.0
        cancel_mask = broken[:-1] | broken[1:]

    if transformed[-1] - transformed[-2] < 1e-6:
        transformed[-1] = transformed[-2] + max(abs(transformed[-2]) * 0.1, 1.0)
    if transformed[1] - transformed[0] < 1e-6:
        transformed[0] = transformed[1] - max(abs(transformed[1]) * 0.1, 1.0)

    differences = np.diff(transformed)
    floor = np.maximum(np.abs(transformed[:-1]) * 1e-12, 1e-9)
    correction = np.where(differences < floor, floor - differences, 0.0)
    transformed[1:] += np.cumsum(correction)
    if np.any(np.diff(transformed) < 0):
        raise ValueError("Target transform produced non-monotonic borders.")
    return transformed, cancel_mask
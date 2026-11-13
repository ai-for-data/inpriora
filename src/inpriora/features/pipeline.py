"""Feature normalization pipelines."""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted

from .validation import check_features

_YEOJOHNSON_XTOL = 1.48e-8
_YEOJOHNSON_INV_GOLD = 0.3819660112501089


def _observed_moments(X, *, ddof: int = 0):
    X = np.asarray(X, dtype=np.float64)
    observed = ~np.isnan(X)
    counts = observed.sum(axis=0)
    values = np.where(observed, X, 0.0)
    magnitude = np.max(np.abs(values), axis=0)
    magnitude = np.where(magnitude > 0, magnitude, 1.0)
    values = values / magnitude
    means = values.sum(axis=0) / np.maximum(counts, 1)
    residual = np.where(observed, values - means, 0.0)
    scales = np.sqrt((residual * residual).sum(axis=0) / np.maximum(counts - ddof, 1)) * magnitude
    scales = np.where((counts > ddof) & (scales > 0), scales, 1.0)
    return means * magnitude, scales


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
    """Remove fully observed constant features."""

    def __init__(self, threshold: int = 1) -> None:
        self.threshold = threshold

    def fit(self, X, y=None):
        X = check_features(X)
        self.n_features_in_ = X.shape[1]
        if X.shape[0] <= self.threshold:
            self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
        else:
            self.features_to_keep_ = np.array(
                [
                    np.isnan(X[:, index]).any()
                    or len(np.unique(X[:, index])) > self.threshold
                    for index in range(self.n_features_in_)
                ]
            )
            if not self.features_to_keep_.any():
                self.features_to_keep_[0] = True
        self.n_features_out_ = int(np.sum(self.features_to_keep_))
        return self

    def transform(self, X):
        check_is_fitted(self, "features_to_keep_")
        X = check_features(X)
        return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
    """Clip extreme values with a two-stage z-score bound."""

    def __init__(self, threshold: float = 4.0) -> None:
        self.threshold = threshold

    def fit(self, X, y=None):
        X = check_features(X)
        self.n_features_in_ = X.shape[1]
        self.means_, self.stds_ = _observed_moments(X, ddof=1)
        X_clean = X.copy()
        lower = self.means_ - self.threshold * self.stds_
        upper = self.means_ + self.threshold * self.stds_
        outlier_mask = (X < lower[np.newaxis, :]) | (X > upper[np.newaxis, :])
        X_clean[outlier_mask] = np.nan
        self.means_, self.stds_ = _observed_moments(X_clean, ddof=1)
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        X = check_features(X)
        X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
        X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)
        return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
    """Standard scaling with observed-value statistics and clipping."""

    def __init__(
        self,
        clip_min: float = -100,
        clip_max: float = 100,
        epsilon: float = 1e-6,
    ) -> None:
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_features(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        self.mean_, self.scale_ = _observed_moments(X)
        self.scale_ = np.maximum(self.scale_, self.epsilon)
        return self

    def transform(self, X):
        check_is_fitted(self, "mean_")
        X = check_features(X)
        vector = X.ndim == 1
        if vector:
            X = X.reshape(-1, 1)
        scaled = (X - self.mean_) / self.scale_
        clipped = np.clip(scaled, self.clip_min, self.clip_max)
        return clipped.reshape(-1) if vector else clipped


class _PrecomputedYeoJohnsonPowerTransformer(PowerTransformer):
    """PowerTransformer with a precomputed Yeo-Johnson lambda."""

    def __init__(self, lmbda: float) -> None:
        super().__init__(method="yeo-johnson", standardize=True)
        self.lmbda = float(lmbda)

    def _yeo_johnson_optimize(self, x):
        return self.lmbda


class PowerNormalizer:
    """Per-column Yeo-Johnson normalizer."""

    def __init__(self, lmbda: Optional[float] = None) -> None:
        self.lmbda = lmbda
        self.tfm_: Optional[PowerTransformer] = None

    def fit(self, values):
        if self.lmbda is None:
            self.tfm_ = PowerTransformer(method="yeo-johnson", standardize=True)
        else:
            self.tfm_ = _PrecomputedYeoJohnsonPowerTransformer(self.lmbda)
        self.tfm_.fit(values)
        return self

    def transform(self, values):
        return self.tfm_.transform(values)

    @property
    def power(self) -> float:
        return float(self.tfm_.lambdas_[0])


FEATURE_NORMALIZERS: dict[str, Optional[Callable[..., PowerNormalizer]]] = {
    "none": None,
    "power": PowerNormalizer,
}


def _yeojohnson_llf_columns_torch(X, observed, S, n, lmbda):
    pos = observed & (X >= 0)
    logpos = torch.where(pos, torch.log1p(X), torch.zeros((), dtype=X.dtype, device=X.device))
    logneg = torch.where(
        observed & ~pos,
        torch.log1p(-X),
        torch.zeros((), dtype=X.dtype, device=X.device),
    )

    eps = torch.finfo(X.dtype).eps
    l0 = lmbda.abs() < eps
    l2 = (lmbda - 2.0).abs() < eps
    safe_lmbda = torch.where(l0, torch.ones_like(lmbda), lmbda)
    safe_two_minus = torch.where(l2, torch.ones_like(lmbda), 2.0 - lmbda)

    y_pos = torch.expm1(lmbda * logpos) / safe_lmbda
    y_pos = torch.where(l0, logpos, y_pos)
    y_neg = -torch.expm1((2.0 - lmbda) * logneg) / safe_two_minus
    y_neg = torch.where(l2, -logneg, y_neg)
    y = torch.where(pos, y_pos, y_neg)

    sum_y = y.sum(dim=0)
    sum_y2 = (y * y).sum(dim=0)
    mean = sum_y / n
    variance = sum_y2 / n - mean * mean
    log_sigma = torch.log(torch.clamp(variance, min=torch.finfo(X.dtype).tiny))
    llf = -(n / 2) * log_sigma + (lmbda - 1.0) * S
    return torch.where(
        torch.isfinite(llf),
        llf,
        torch.full_like(llf, float("-inf")),
    )


def _yeojohnson_lambdas_block_torch(X, observed, device):
    n = observed.sum(dim=0).to(X.dtype)
    pos = observed & (X >= 0)
    logpos = torch.where(pos, torch.log1p(X), torch.zeros((), dtype=X.dtype, device=X.device))
    logneg = torch.where(
        observed & ~pos,
        torch.log1p(-X),
        torch.zeros((), dtype=X.dtype, device=X.device),
    )
    S = logpos.sum(dim=0) - logneg.sum(dim=0)

    abs_max = torch.where(
        observed,
        X.abs(),
        torch.zeros((), dtype=X.dtype, device=X.device),
    ).amax(dim=0)
    finfo = torch.finfo(X.dtype)
    log_eps = -torch.log(torch.tensor(finfo.eps, dtype=X.dtype, device=device))
    log_tiny = (torch.log(torch.tensor(finfo.tiny, dtype=X.dtype, device=device)) - log_eps) / 2
    log_max = (torch.log(torch.tensor(finfo.max, dtype=X.dtype, device=device)) + log_eps) / 2
    log1p_max_x = torch.log1p(abs_max)
    lower = log_tiny / log1p_max_x
    upper = log_max / log1p_max_x
    all_neg = torch.all(observed & (X < 0), dim=0)
    any_neg = torch.any(observed & (X < 0), dim=0)
    lower = torch.where(all_neg, 2 - upper, torch.where(any_neg, torch.maximum(2 - upper, lower), lower))
    upper = torch.where(all_neg, 2 - lower, torch.where(any_neg, torch.minimum(2 - lower, upper), upper))

    a = lower.clone()
    b = upper.clone()
    xlo = a + _YEOJOHNSON_INV_GOLD * (b - a)
    xhi = b - _YEOJOHNSON_INV_GOLD * (b - a)
    flo = _yeojohnson_llf_columns_torch(X, observed, S, n, xlo)
    fhi = _yeojohnson_llf_columns_torch(X, observed, S, n, xhi)
    for _ in range(200):
        if bool(torch.all(b - a <= _YEOJOHNSON_XTOL * (1.0 + a.abs() + b.abs()) / 2)):
            break
        take_right = flo <= fhi
        new_a = torch.where(take_right, xlo, a)
        new_b = torch.where(take_right, b, xhi)
        kept_x = torch.where(take_right, xhi, xlo)
        kept_f = torch.where(take_right, fhi, flo)
        span = new_b - new_a
        candidate = torch.where(
            take_right,
            new_b - _YEOJOHNSON_INV_GOLD * span,
            new_a + _YEOJOHNSON_INV_GOLD * span,
        )
        f_candidate = _yeojohnson_llf_columns_torch(X, observed, S, n, candidate)
        xlo = torch.where(take_right, kept_x, candidate)
        flo = torch.where(take_right, kept_f, f_candidate)
        xhi = torch.where(take_right, candidate, kept_x)
        fhi = torch.where(take_right, f_candidate, kept_f)
        a, b = new_a, new_b
    return torch.where(flo > fhi, xlo, xhi)


def _yeojohnson_torch(x: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """Vectorized Yeo-Johnson transform over columns."""
    eps = torch.finfo(x.dtype).eps
    positive = x >= 0
    x_pos = torch.clamp_min(x, 0)
    x_neg = torch.clamp_min(-x, 0)
    at_zero = lambdas.abs() < eps
    at_two = (lambdas - 2).abs() < eps
    safe_lambda = torch.where(at_zero, torch.ones_like(lambdas), lambdas)
    safe_two = torch.where(at_two, torch.ones_like(lambdas), 2 - lambdas)

    positive_values = torch.expm1(safe_lambda * torch.log1p(x_pos)) / safe_lambda
    positive_values = torch.where(at_zero, torch.log1p(x_pos), positive_values)
    negative_values = -torch.expm1(safe_two * torch.log1p(x_neg)) / safe_two
    negative_values = torch.where(at_two, -torch.log1p(x_neg), negative_values)
    return torch.where(positive, positive_values, negative_values)


def _fit_yeojohnson_lambdas_torch(
    X_scaled,
    observed,
    device,
    max_block_elements: int = 24_000_000,
):
    device = torch.device(device)
    n_rows, n_cols = X_scaled.shape
    lambdas = np.full(n_cols, np.nan)
    if n_rows == 0 or n_cols == 0:
        return lambdas
    block_cols = max(1, max_block_elements // max(n_rows, 1))
    with torch.no_grad():
        for start in range(0, n_cols, block_cols):
            stop = min(start + block_cols, n_cols)
            X_block = torch.from_numpy(
                np.ascontiguousarray(X_scaled[:, start:stop])
            ).to(device, torch.float64)
            observed_block = torch.from_numpy(observed[:, start:stop]).to(device)
            X_block = torch.where(
                observed_block,
                X_block,
                torch.zeros((), device=device, dtype=X_block.dtype),
            )
            block_lambdas = _yeojohnson_lambdas_block_torch(
                X_block,
                observed_block,
                device,
            ).cpu().numpy()
            observed_slice = observed[:, start:stop]
            scaled_slice = np.where(observed_slice, X_scaled[:, start:stop], np.nan)
            n_obs = observed_slice.sum(axis=0)
            col_min = np.nanmin(np.where(observed_slice, scaled_slice, np.inf), axis=0)
            col_max = np.nanmax(np.where(observed_slice, scaled_slice, -np.inf), axis=0)
            block_lambdas[(n_obs < 2) | ~(col_max > col_min)] = np.nan
            lambdas[start:stop] = block_lambdas
    return lambdas


class FeaturePipeline(TransformerMixin, BaseEstimator):
    """Scale, normalize, and clip a feature matrix."""

    def __init__(
        self,
        normalization_method: str = "none",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = 0,
        yeojohnson_device: Optional[str] = None,
    ) -> None:
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.yeojohnson_device = yeojohnson_device

    def fit(self, X, y=None):
        X = check_features(X)
        if X.dtype == np.float16:
            X = X.astype(np.float32)
        self.output_dtype_ = (
            X.dtype if np.issubdtype(X.dtype, np.floating) else np.dtype(np.float64)
        )
        if self.normalization_method not in FEATURE_NORMALIZERS:
            raise ValueError(
                f"Unsupported normalization method {self.normalization_method!r}."
            )
        self.n_features_in_ = X.shape[1]

        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)

        self.normalizers_: List[Optional[PowerNormalizer]] = []
        self.normalization_lower_ = np.full(
            self.n_features_in_,
            self.standard_scaler_.clip_min,
            dtype=float,
        )
        self.normalization_upper_ = np.full(
            self.n_features_in_,
            self.standard_scaler_.clip_max,
            dtype=float,
        )

        lambdas = None
        if (
            self.normalization_method == "power"
            and self.yeojohnson_device
            and self.n_features_in_
        ):
            try:
                lambdas = _fit_yeojohnson_lambdas_torch(
                    X_scaled,
                    ~np.isnan(X_scaled),
                    self.yeojohnson_device,
                )
            except (RuntimeError, TypeError, ValueError):
                lambdas = None

        for column in range(self.n_features_in_):
            observed = ~np.isnan(X_scaled[:, column])
            values = X_scaled[observed, column, None]
            normalizer = None
            if self.normalization_method != "none" and np.unique(values).size > 1:
                lmbda = None if lambdas is None else lambdas[column]
                normalizer = self._make_normalizer(lmbda)
                normalizer.fit(values)
                if isinstance(normalizer, PowerNormalizer):
                    exponent_limit = np.log(np.finfo(np.float64).max) / 4
                    power = normalizer.power
                    self.normalization_upper_[column] = np.expm1(
                        min(
                            np.log1p(self.standard_scaler_.clip_max),
                            exponent_limit / max(power, 1.0),
                        )
                    )
                    self.normalization_lower_[column] = -np.expm1(
                        min(
                            np.log1p(-self.standard_scaler_.clip_min),
                            exponent_limit / max(2 - power, 1.0),
                        )
                    )
            self.normalizers_.append(normalizer)

        self._torch_lambdas = np.full(self.n_features_in_, np.nan, dtype=np.float64)
        self._torch_power_mean = np.zeros(self.n_features_in_, dtype=np.float64)
        self._torch_power_scale = np.ones(self.n_features_in_, dtype=np.float64)
        self._torch_power_mask = np.zeros(self.n_features_in_, dtype=bool)
        for column, normalizer in enumerate(self.normalizers_):
            if isinstance(normalizer, PowerNormalizer):
                fitted = normalizer.tfm_
                self._torch_lambdas[column] = float(fitted.lambdas_[0])
                self._torch_power_mean[column] = float(fitted._scaler.mean_[0])
                self._torch_power_scale[column] = float(fitted._scaler.scale_[0])
                self._torch_power_mask[column] = True

        X_normalized = self._normalize(X_scaled)
        self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
        self.X_transformed_ = self.outlier_remover_.fit_transform(
            X_normalized
        ).astype(self.output_dtype_)
        return self

    def _make_normalizer(self, lmbda: Optional[float]):
        factory = FEATURE_NORMALIZERS[self.normalization_method]
        if factory is None:
            return None
        return factory(lmbda)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(X, self.normalization_lower_, self.normalization_upper_)
        for column, normalizer in enumerate(self.normalizers_):
            if normalizer is not None:
                observed = ~np.isnan(X[:, column])
                if observed.any():
                    X[observed, column] = normalizer.transform(
                        X[observed, column, None]
                    )[:, 0]
        return X

    def _use_torch_transform(self) -> bool:
        if self.yeojohnson_device is None or not torch.cuda.is_available():
            return False
        if os.environ.get("INPRIORA_TORCH_PREPROCESS", "1").lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return False
        return torch.device(self.yeojohnson_device).type == "cuda"

    def _torch_transform(self, X: np.ndarray) -> np.ndarray:
        device = torch.device(self.yeojohnson_device)
        value = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=device)

        mean = torch.as_tensor(
            self.standard_scaler_.mean_,
            dtype=torch.float64,
            device=device,
        )
        scale = torch.as_tensor(
            self.standard_scaler_.scale_,
            dtype=torch.float64,
            device=device,
        )
        value = (value - mean) / scale
        value = torch.clamp(
            value,
            self.standard_scaler_.clip_min,
            self.standard_scaler_.clip_max,
        )

        lower = torch.as_tensor(
            self.normalization_lower_,
            dtype=torch.float64,
            device=device,
        )
        upper = torch.as_tensor(
            self.normalization_upper_,
            dtype=torch.float64,
            device=device,
        )
        value = torch.clamp(value, lower, upper)

        lambdas = torch.as_tensor(
            self._torch_lambdas,
            dtype=torch.float64,
            device=device,
        )
        transformed = _yeojohnson_torch(value, lambdas)
        power_mean = torch.as_tensor(
            self._torch_power_mean,
            dtype=torch.float64,
            device=device,
        )
        power_scale = torch.as_tensor(
            self._torch_power_scale,
            dtype=torch.float64,
            device=device,
        )
        transformed = (transformed - power_mean) / power_scale
        mask = torch.as_tensor(
            self._torch_power_mask,
            device=device,
        ).unsqueeze(0)
        transformed = torch.where(mask, transformed, value)

        out_lower = torch.as_tensor(
            self.outlier_remover_.lower_bounds_,
            dtype=torch.float64,
            device=device,
        )
        out_upper = torch.as_tensor(
            self.outlier_remover_.upper_bounds_,
            dtype=torch.float64,
            device=device,
        )
        transformed = torch.maximum(
            -torch.log1p(transformed.abs()) + out_lower,
            transformed,
        )
        transformed = torch.minimum(
            torch.log1p(transformed.abs()) + out_upper,
            transformed,
        )
        return transformed.cpu().numpy().astype(self.output_dtype_)

    def transform(self, X):
        check_is_fitted(self, "X_transformed_")
        X = check_features(X, copy=True)
        if self._use_torch_transform():
            return self._torch_transform(X)
        X = self.standard_scaler_.transform(X)
        X = self._normalize(X)
        X = self.outlier_remover_.transform(X)
        return X.astype(self.output_dtype_)
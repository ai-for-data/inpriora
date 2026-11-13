"""Input feature encoding."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.utils.validation import check_array, check_is_fitted


def _numeric_array(X):
    if hasattr(X, "to_numpy"):
        X = X.to_numpy(dtype=np.float64, na_value=np.nan)
    return check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")


def _categorical_missing_values(X):
    return X.astype(object).where(X.notna(), np.nan)


class InputEncoder(TransformerMixin, BaseEstimator):
    """Convert tabular inputs to a numeric matrix.

    DataFrame categorical, string, object, and boolean columns are ordinal
    encoded with missing values preserved. Numeric columns are converted to
    float64. NumPy inputs must already be numeric.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the column-wise encoders."""
        categorical = Pipeline(
            [
                ("missing", FunctionTransformer(_categorical_missing_values)),
                (
                    "ordinal",
                    OrdinalEncoder(
                        dtype=np.float64,
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                        encoded_missing_value=np.nan,
                    ),
                ),
            ]
        )
        numeric = FunctionTransformer(_numeric_array)

        if not hasattr(X, "columns"):
            if issparse(X):
                raise TypeError(
                    "Sparse input is not supported. Convert X to a dense array."
                )
            array = np.asarray(X)
            try:
                array.astype(np.float64)
            except (ValueError, TypeError) as error:
                raise TypeError(
                    "NumPy input must be numeric. Pass a pandas DataFrame for "
                    f"categorical or string columns. Original error: {error}"
                ) from None
            self.tfm_ = numeric
            self.categorical_positions_: list[int] = []
        else:
            categorical_columns = make_column_selector(
                dtype_include=["string", "object", "category", "boolean"]
            )(X)
            categorical_positions = [X.columns.get_loc(column) for column in categorical_columns]
            high_cardinality = [
                column for column in categorical_columns if X[column].nunique() > 40
            ]
            if high_cardinality:
                warnings.warn(
                    "Categorical columns with cardinality above 40 use ordinal "
                    f"encoding: {high_cardinality}",
                    UserWarning,
                    stacklevel=2,
                )
            numeric_columns = make_column_selector(dtype_include="number")(X)
            numeric_positions = [X.columns.get_loc(column) for column in numeric_columns]
            self.tfm_ = ColumnTransformer(
                transformers=[
                    ("categorical", categorical, categorical_positions),
                    ("continuous", numeric, numeric_positions),
                ]
            )
            self.categorical_positions_ = list(range(len(categorical_positions)))

        self.tfm_.fit(X)
        transformed = self.tfm_.transform(X)
        check_array(transformed, ensure_all_finite="allow-nan")
        self.n_features_in_ = transformed.shape[1]
        return self

    def transform(self, X):
        """Transform inputs to a numeric matrix."""
        check_is_fitted(self, "tfm_")
        transformed = self.tfm_.transform(X)
        return check_array(transformed, ensure_all_finite="allow-nan")
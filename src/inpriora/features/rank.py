"""Per-cell rank channel."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

EMPTY_RANK = 0.5


class RankReference:
    """Per-column Support distribution used to rank table cells."""

    def __init__(
        self,
        sorted_columns: List[np.ndarray],
        imputation: np.ndarray,
        n_rows: int,
    ) -> None:
        self.sorted_columns = sorted_columns
        self.imputation = imputation
        self.n_rows = n_rows

    @classmethod
    def fit(cls, support: np.ndarray) -> "RankReference":
        """Fit one sorted Support column per feature."""
        support = np.asarray(support)
        if support.ndim != 2:
            raise ValueError(f"support must be two-dimensional, got shape {support.shape}.")
        n_rows, num_columns = support.shape
        observed = np.isfinite(support)
        sums = np.where(observed, support, 0.0).sum(axis=0)
        counts = observed.sum(axis=0)
        imputation = np.divide(
            sums,
            counts,
            out=np.zeros(num_columns, dtype=sums.dtype),
            where=counts > 0,
        )
        filled = np.where(observed, support, imputation)
        sorted_columns = [np.sort(filled[:, index]) for index in range(num_columns)]
        return cls(sorted_columns, imputation, n_rows)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Return midranks in ``[0, 1]`` for every cell."""
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"values must be two-dimensional, got shape {values.shape}.")
        if values.shape[1] != len(self.sorted_columns):
            raise ValueError(
                f"values has {values.shape[1]} columns but the reference has "
                f"{len(self.sorted_columns)}."
            )

        output = np.full(values.shape, EMPTY_RANK, dtype=np.float64)
        if self.n_rows == 0:
            return output

        for index, sorted_column in enumerate(self.sorted_columns):
            column = values[:, index]
            filled = np.where(
                np.isfinite(column),
                column,
                self.imputation[index],
            )
            left = np.searchsorted(sorted_column, filled, side="left")
            right = np.searchsorted(sorted_column, filled, side="right")
            output[:, index] = (left + right) / (2.0 * self.n_rows)
        return output

    def transform_torch(self, values: np.ndarray, device) -> np.ndarray:
        """Return midranks using batched torch search on a CUDA device."""
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"values must be two-dimensional, got shape {values.shape}.")
        if values.shape[1] != len(self.sorted_columns):
            raise ValueError(
                f"values has {values.shape[1]} columns but the reference has "
                f"{len(self.sorted_columns)}."
            )

        output = np.full(values.shape, EMPTY_RANK, dtype=np.float64)
        if self.n_rows == 0:
            return output

        torch_device = torch.device(device)
        sorted_tensor = torch.as_tensor(
            np.stack(self.sorted_columns, axis=0),
            dtype=torch.float64,
            device=torch_device,
        )
        imputation = torch.as_tensor(
            self.imputation,
            dtype=torch.float64,
            device=torch_device,
        )
        columns = torch.as_tensor(
            np.ascontiguousarray(values.T),
            dtype=torch.float64,
            device=torch_device,
        )
        filled = torch.where(
            torch.isfinite(columns),
            columns,
            imputation[:, None],
        )
        left = torch.searchsorted(sorted_tensor, filled, side="left")
        right = torch.searchsorted(sorted_tensor, filled, side="right")
        ranks = (left + right).to(torch.float64) / (2.0 * self.n_rows)
        return ranks.T.cpu().numpy()
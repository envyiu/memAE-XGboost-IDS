from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class IDSPreprocessor:
    def __init__(
        self,
        feature_columns: list[str],
        invalid_negative_columns: list[str] | None = None,
        clip_quantiles: tuple[float, float] = (0.001, 0.999),
    ):
        self.feature_columns = feature_columns
        self.invalid_negative_columns = invalid_negative_columns or []
        self.invalid_negative_indices = [
            idx for idx, col in enumerate(self.feature_columns) if col in set(self.invalid_negative_columns)
        ]
        self.clip_quantiles = clip_quantiles
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.fitted = False

    def _sanitize(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            X = data[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        else:
            X = np.asarray(data, dtype=np.float32).copy()
        for idx in self.invalid_negative_indices:
            mask = X[:, idx] < 0
            if mask.any():
                X[mask, idx] = np.nan
        return X

    def fit(self, data: pd.DataFrame | np.ndarray) -> "IDSPreprocessor":
        X = self._sanitize(data)
        low_q, high_q = self.clip_quantiles
        self.lower_bounds_ = np.nanquantile(X, low_q, axis=0).astype(np.float32)
        self.upper_bounds_ = np.nanquantile(X, high_q, axis=0).astype(np.float32)
        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        self.pipeline.fit(X.astype(np.float32, copy=False))
        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor chưa được fit")
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Preprocessor thiếu clipping bounds")
        X = self._sanitize(data)
        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        arr = self.pipeline.transform(X.astype("float32", copy=False))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        return arr

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "IDSPreprocessor":
        return joblib.load(path)

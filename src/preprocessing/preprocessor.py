from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PreprocessDevice = Literal["cpu", "cuda", "auto"]


def _is_no_clip_feature(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("ctx_")
        or lowered.startswith("is_")
        or "_is_" in lowered
        or lowered.endswith("_indicator")
        or "_indicator_" in lowered
    )


class IDSPreprocessor:
    def __init__(
        self,
        feature_columns: list[str],
        invalid_negative_columns: list[str] | None = None,
        clip_quantiles: tuple[float, float] = (0.001, 0.999),
    ):
        low_q, high_q = clip_quantiles
        if not 0.0 <= low_q <= high_q <= 1.0:
            raise ValueError("clip_quantiles must satisfy 0 <= low <= high <= 1")
        self.feature_columns = feature_columns
        self.invalid_negative_columns = invalid_negative_columns or []
        invalid_negative_set = set(self.invalid_negative_columns)
        self.invalid_negative_indices = [
            idx for idx, col in enumerate(self.feature_columns) if col in invalid_negative_set
        ]
        self.no_clip_indices = [
            idx for idx, col in enumerate(self.feature_columns) if _is_no_clip_feature(col)
        ]
        self.clip_quantiles = clip_quantiles
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        self.fitted = False

    @staticmethod
    def resolve_device(device: PreprocessDevice = "cpu") -> str:
        if device == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError:
            if device == "cuda":
                raise RuntimeError("preprocess-device=cuda cần cài torch") from None
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if device == "cuda":
            raise RuntimeError("preprocess-device=cuda nhưng CUDA không khả dụng")
        return "cpu"

    def _validate_array_shape(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {X.shape}")
        if X.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} feature columns, got {X.shape[1]}"
            )

    def _sanitize(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            missing = [col for col in self.feature_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            X = data[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        else:
            X = np.asarray(data, dtype=np.float32).copy()
        self._validate_array_shape(X)
        X[~np.isfinite(X)] = np.nan
        for idx in self.invalid_negative_indices:
            mask = X[:, idx] < 0
            if mask.any():
                X[mask, idx] = np.nan
        return X

    def fit(self, data: pd.DataFrame | np.ndarray) -> "IDSPreprocessor":
        X = self._sanitize(data)
        if X.shape[0] == 0:
            raise ValueError("Cannot fit preprocessor on an empty matrix")
        low_q, high_q = self.clip_quantiles
        all_nan = np.isnan(X).all(axis=0)
        X_for_quantiles = X.copy() if all_nan.any() else X
        if all_nan.any():
            X_for_quantiles[:, all_nan] = 0.0
        self.lower_bounds_ = np.nanquantile(X_for_quantiles, low_q, axis=0).astype(np.float32)
        self.upper_bounds_ = np.nanquantile(X_for_quantiles, high_q, axis=0).astype(np.float32)
        self.lower_bounds_[~np.isfinite(self.lower_bounds_)] = 0.0
        self.upper_bounds_[~np.isfinite(self.upper_bounds_)] = 0.0
        if self.no_clip_indices:
            no_clip = np.asarray(self.no_clip_indices, dtype=np.int64)
            self.lower_bounds_[no_clip] = -np.inf
            self.upper_bounds_[no_clip] = np.inf
        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        self.pipeline.fit(X.astype(np.float32, copy=False))
        self.fitted = True
        return self

    def _as_float32_array(self, data: pd.DataFrame | np.ndarray, copy: bool) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            missing = [col for col in self.feature_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            return data[self.feature_columns].to_numpy(dtype=np.float32, copy=copy)
        X = np.asarray(data, dtype=np.float32)
        self._validate_array_shape(X)
        return X.copy() if copy else X

    def _transform_cpu(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = self._sanitize(data)
        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        arr = self.pipeline.transform(X.astype("float32", copy=False))
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def _transform_cuda(self, data: pd.DataFrame | np.ndarray, batch_rows: int) -> np.ndarray:
        import torch

        if batch_rows <= 0:
            raise ValueError("batch_rows phải > 0")
        X = self._as_float32_array(data, copy=False)
        if not X.flags.c_contiguous:
            X = np.ascontiguousarray(X)
        out = np.empty(X.shape, dtype=np.float32)
        device = torch.device("cuda")
        lower = torch.as_tensor(self.lower_bounds_, dtype=torch.float32, device=device)
        upper = torch.as_tensor(self.upper_bounds_, dtype=torch.float32, device=device)
        imputer = self.pipeline.named_steps["imputer"]
        scaler = self.pipeline.named_steps["scaler"]
        medians = torch.as_tensor(imputer.statistics_.astype(np.float32, copy=False), dtype=torch.float32, device=device)
        mean = torch.as_tensor(scaler.mean_.astype(np.float32, copy=False), dtype=torch.float32, device=device)
        scale = torch.as_tensor(scaler.scale_.astype(np.float32, copy=False), dtype=torch.float32, device=device)
        invalid_idx = torch.as_tensor(self.invalid_negative_indices, dtype=torch.long, device=device)
        nan = torch.tensor(float("nan"), dtype=torch.float32, device=device)

        with torch.no_grad():
            for start in range(0, X.shape[0], batch_rows):
                end = min(start + batch_rows, X.shape[0])
                batch = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
                batch = torch.where(torch.isfinite(batch), batch, nan)
                if invalid_idx.numel():
                    selected = batch[:, invalid_idx]
                    batch[:, invalid_idx] = torch.where(selected < 0, nan, selected)
                batch = torch.minimum(torch.maximum(batch, lower), upper)
                batch = torch.where(torch.isnan(batch), medians, batch)
                batch = (batch - mean) / scale
                batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
                out[start:end] = batch.cpu().numpy()
        return out

    def transform(
        self,
        data: pd.DataFrame | np.ndarray,
        device: PreprocessDevice = "cpu",
        batch_rows: int = 262_144,
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor chưa được fit")
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Preprocessor thiếu clipping bounds")
        resolved_device = self.resolve_device(device)
        if resolved_device == "cuda":
            return self._transform_cuda(data, batch_rows=batch_rows)
        return self._transform_cpu(data)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "IDSPreprocessor":
        return joblib.load(path)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from src.features.window_features import (
    add_window_features,
    resolve_window_config,
    window_feature_names,
    window_required_columns,
)
from src.preprocessing.preprocessor import IDSPreprocessor
from src.utils.io import ensure_dir, read_json, read_yaml, write_json


def _load_split_ids(split_dir: Path, name: str) -> np.ndarray:
    return pd.read_csv(split_dir / f"{name}.csv")["row_id"].to_numpy()


def _source_files(clean_path: Path) -> list[str]:
    df = pd.read_parquet(clean_path, columns=["source_file"])
    return sorted(df["source_file"].dropna().astype(str).unique().tolist())


def preprocess_experiment(
    experiment: str = "zero_day_dos",
    clean_path: str | Path = "data/interim/cicids2017_clean.parquet",
    split_dir: str | Path = "data/splits/zero_day_dos",
    schema_path: str | Path = "data/interim/column_schema.json",
    window_config_path: str | Path | None = None,
    window_config: dict | None = None,
    benchmark_mode: str | None = None,
    fit_sample_rows: int = 400_000,
) -> Path:
    split_dir = Path(split_dir)
    clean_path = Path(clean_path)
    processed_dir = ensure_dir(Path("data/processed") / experiment)
    preprocessor_dir = ensure_dir(Path("artifacts/preprocessors") / experiment)

    schema = read_json(schema_path)
    available_columns = set(schema.get("all_columns", []))
    feature_columns = schema["numerical_features"]
    invalid_negative_columns = [
        col
        for col in [
            "fwd_header_length",
            "fwd_header_length_1",
            "bwd_header_length",
            "min_seg_size_forward",
        ]
        if col in feature_columns
    ]
    window_cfg = window_config
    if window_cfg is None and window_config_path is not None:
        window_cfg = read_yaml(window_config_path)
    window_enabled = bool(window_cfg and window_cfg.get("enabled", True))
    window_scope = window_cfg.get("window_scope", "full_source_file") if window_enabled else None
    if benchmark_mode is None:
        benchmark_mode = "contextual_window" if window_enabled else "strict_nowindow"
    extra_columns: list[str] = []
    if window_enabled:
        window_cfg = resolve_window_config(window_cfg, available_columns)
        extra_columns = [col for col in window_required_columns(window_cfg) if col in available_columns]
    columns = ["row_id", "attack_family", *feature_columns]
    for col in extra_columns:
        if col not in columns:
            columns.append(col)

    split_ids = {name: _load_split_ids(split_dir, name) for name in ("train", "val", "test_seen", "test_zero_day")}
    split_indices = {name: pd.Index(ids) for name, ids in split_ids.items()}
    files = _source_files(clean_path)

    train_chunks: list[np.ndarray] = []
    window_columns: list[str] = window_feature_names(window_cfg) if window_enabled else []
    final_feature_columns = feature_columns + window_columns if window_enabled else list(feature_columns)
    rng = np.random.default_rng(42)
    train_total = int(len(split_ids["train"]))
    sample_ratio = min(1.0, float(fit_sample_rows) / max(train_total, 1))
    for source_file in files:
        chunk = pd.read_parquet(clean_path, columns=columns, filters=[("source_file", "==", source_file)])
        if chunk.empty:
            continue
        chunk = chunk.set_index("row_id", drop=False)
        if window_enabled and window_scope == "full_source_file":
            chunk, _ = add_window_features(chunk, window_cfg)
        train_chunk_ids = chunk.index.intersection(split_indices["train"], sort=False)
        if train_chunk_ids.empty:
            continue
        train_part = chunk.loc[train_chunk_ids].copy()
        if window_enabled and window_scope == "split_only":
            train_part, _ = add_window_features(train_part, window_cfg)
        if window_enabled and window_scope == "causal_past_only":
            # causal past only for train means using only train rows
            train_part, _ = add_window_features(train_part, window_cfg)
        if sample_ratio < 1.0:
            sample_size = max(1, int(round(len(train_part) * sample_ratio)))
            sample_size = min(sample_size, len(train_part))
            picked = rng.choice(len(train_part), size=sample_size, replace=False)
            train_part = train_part.iloc[np.sort(picked)]
        train_chunks.append(train_part[final_feature_columns].to_numpy(dtype=np.float32, copy=True))

    if not train_chunks:
        raise RuntimeError(f"Không tạo được train split cho {experiment}")
    train_matrix = np.concatenate(train_chunks, axis=0)

    preprocessor = IDSPreprocessor(
        final_feature_columns,
        invalid_negative_columns=invalid_negative_columns,
        clip_quantiles=(0.001, 0.999),
    )
    preprocessor.fit(train_matrix)
    preprocessor.save(preprocessor_dir / "preprocessor.joblib")
    del train_matrix
    del train_chunks

    for name in ("train", "val", "test_seen", "test_zero_day"):
        total_rows = int(len(split_ids[name]))
        X_memmap = open_memmap(
            processed_dir / f"X_{name}.npy",
            mode="w+",
            dtype="float32",
            shape=(total_rows, len(final_feature_columns)),
        )
        row_id_arr = np.empty(total_rows, dtype="int64")
        y_arr = np.empty(total_rows, dtype="int64")
        family_arr = np.empty(total_rows, dtype=object)
        offset = 0
        for source_file in files:
            chunk = pd.read_parquet(clean_path, columns=columns, filters=[("source_file", "==", source_file)])
            if chunk.empty:
                continue
            chunk = chunk.set_index("row_id", drop=False)
            if window_enabled and window_scope == "full_source_file":
                chunk, _ = add_window_features(chunk, window_cfg)
            
            if window_enabled and window_scope == "causal_past_only":
                # For causal past, val can see train+val, test can see train+val+test
                allowed_splits = ["train"]
                if name == "val": allowed_splits.extend(["val"])
                if name == "test_seen": allowed_splits.extend(["val", "test_seen"])
                if name == "test_zero_day": allowed_splits.extend(["val", "test_zero_day"])
                # We need all row_ids from these allowed splits
                causal_ids = pd.Index([])
                for s_name in allowed_splits:
                    causal_ids = causal_ids.union(split_indices[s_name])
                causal_chunk_ids = chunk.index.intersection(causal_ids, sort=False)
                causal_part = chunk.loc[causal_chunk_ids].copy()
                causal_part, _ = add_window_features(causal_part, window_cfg)
                
                chunk_ids = chunk.index.intersection(split_indices[name], sort=False)
                if chunk_ids.empty:
                    continue
                part = causal_part.loc[chunk_ids].copy()
            else:
                chunk_ids = chunk.index.intersection(split_indices[name], sort=False)
                if chunk_ids.empty:
                    continue
                part = chunk.loc[chunk_ids].copy()
                if window_enabled and window_scope == "split_only":
                    part, _ = add_window_features(part, window_cfg)
            raw = part[final_feature_columns].to_numpy(dtype=np.float32, copy=True)
            family = part["attack_family"].to_numpy(copy=True)
            row_ids = part["row_id"].to_numpy(dtype="int64", copy=True)
            transformed = preprocessor.transform(raw)
            next_offset = offset + len(part)
            X_memmap[offset:next_offset] = transformed
            row_id_arr[offset:next_offset] = row_ids
            family_arr[offset:next_offset] = family
            y_arr[offset:next_offset] = (family != "benign").astype("int64")
            offset = next_offset
        if offset == 0:
            raise RuntimeError(f"Split {name} rỗng cho {experiment}")
        if offset != total_rows:
            raise RuntimeError(f"Split {name} ghi {offset} rows nhưng mong đợi {total_rows}")
        del X_memmap
        np.save(processed_dir / f"row_id_{name}.npy", row_id_arr)
        np.save(processed_dir / f"y_{name}.npy", y_arr)
        np.save(processed_dir / f"family_{name}.npy", family_arr)

    schema_payload = {
        "experiment": experiment,
        "benchmark_mode": benchmark_mode,
        "clean_path": str(clean_path),
        "split_dir": str(split_dir),
        "schema_path": str(schema_path),
        "window_config_path": str(window_config_path) if window_config_path is not None else None,
        "feature_order": final_feature_columns,
        "total_features_after_preprocessing": len(final_feature_columns),
        "scaler_type": "StandardScaler",
        "imputation": "median",
        "invalid_negative_columns": invalid_negative_columns,
        "clip_quantiles": [0.001, 0.999],
        "clip_fit_scope": "train split only",
        "fit_scope": "train split only",
        "fit_sample_rows": int(min(fit_sample_rows, train_total)),
        "window_features": {
            "enabled": window_enabled,
            "columns": window_columns,
            "config": window_cfg if window_enabled else None,
            "window_scope": window_scope,
            "computation_scope": "full source_file chunk before split filtering" if window_enabled else None,
        },
    }
    write_json(processed_dir / "feature_schema.json", schema_payload)
    return processed_dir

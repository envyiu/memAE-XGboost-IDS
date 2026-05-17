from __future__ import annotations

import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from src.features.window import (
    add_window_features,
    resolve_window_config,
    window_feature_names,
    window_required_columns,
)
from src.preprocessing.preprocessor import IDSPreprocessor
from src.utils.io import ensure_dir, read_json, read_yaml, write_json


REQUIRED_SPLITS = ("train", "val", "test_seen", "test_zero_day")
OPTIONAL_SPLITS = ("model_selection_val",)


def _log_preprocess_device(requested_device: str, resolved_device: str, batch_rows: int) -> None:
    if resolved_device == "cuda":
        print(f"[preprocess] transform backend: cuda, batch_rows={batch_rows}")
        return
    if requested_device != "cpu":
        print(f"[preprocess] transform backend: cpu (requested={requested_device}), batch_rows={batch_rows}")
        return
    print(f"[preprocess] transform backend: cpu, batch_rows={batch_rows}")


def _load_split_ids(split_dir: Path, name: str) -> np.ndarray:
    return pd.read_csv(split_dir / f"{name}.csv")["row_id"].to_numpy()


def _resolve_split_names(split_dir: Path) -> tuple[str, ...]:
    split_names = list(REQUIRED_SPLITS)
    for name in OPTIONAL_SPLITS:
        path = split_dir / f"{name}.csv"
        if not path.exists():
            continue
        if len(pd.read_csv(path, usecols=["row_id"])) == 0:
            print(f"[preprocess] optional split {name} exists but is empty; skipping")
            continue
        split_names.append(name)
    return tuple(split_names)


def _source_files(clean_path: Path) -> list[str]:
    df = pd.read_parquet(clean_path, columns=["source_file"])
    return sorted(df["source_file"].dropna().astype(str).unique().tolist())


def _tmp_output_path(final_path: Path, tmp_dir: Path | None) -> Path:
    if tmp_dir is None:
        return final_path
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / final_path.name


def _move_tmp_output(tmp_path: Path, final_path: Path) -> None:
    if tmp_path == final_path:
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.unlink()
    shutil.move(str(tmp_path), str(final_path))


def _save_npy(path: Path, arr: np.ndarray, tmp_dir: Path | None) -> None:
    tmp_path = _tmp_output_path(path, tmp_dir)
    if tmp_path.exists():
        tmp_path.unlink()
    np.save(tmp_path, arr)
    _move_tmp_output(tmp_path, path)


def _effective_preprocess_workers(
    requested_workers: int,
    resolved_device: str,
    window_enabled: bool,
    window_scope: str | None,
) -> int:
    if resolved_device != "cpu" or not window_enabled or window_scope != "full_source_file":
        return 1
    if requested_workers > 0:
        return requested_workers
    return max(1, min(4, os.cpu_count() or 1))


def _load_prepared_source_chunk(
    clean_path: Path,
    source_file: str,
    columns: list[str],
    window_enabled: bool,
    window_scope: str | None,
    window_cfg: dict | None,
) -> pd.DataFrame:
    chunk = pd.read_parquet(clean_path, columns=columns, filters=[("source_file", "==", source_file)])
    if chunk.empty:
        return chunk
    chunk = chunk.set_index("row_id", drop=False)
    if window_enabled and window_scope == "full_source_file":
        chunk, _ = add_window_features(chunk, window_cfg)
    return chunk


def _fit_source_chunk_worker(args: tuple) -> np.ndarray | None:
    (
        clean_path,
        source_file,
        columns,
        window_enabled,
        window_scope,
        window_cfg,
        train_ids,
        final_feature_columns,
        sample_ratio,
        seed,
    ) = args
    chunk = _load_prepared_source_chunk(
        Path(clean_path),
        source_file,
        columns,
        window_enabled,
        window_scope,
        window_cfg,
    )
    if chunk.empty:
        return None
    train_chunk_ids = chunk.index.intersection(pd.Index(train_ids), sort=False)
    if train_chunk_ids.empty:
        return None
    train_part = chunk.loc[train_chunk_ids].copy()
    if sample_ratio < 1.0:
        rng = np.random.default_rng(seed)
        sample_size = max(1, int(round(len(train_part) * sample_ratio)))
        sample_size = min(sample_size, len(train_part))
        picked = rng.choice(len(train_part), size=sample_size, replace=False)
        train_part = train_part.iloc[np.sort(picked)]
    return train_part[final_feature_columns].to_numpy(dtype=np.float32, copy=True)


_TRANSFORM_WORKER_CTX: dict = {}


def _init_transform_worker(ctx: dict) -> None:
    global _TRANSFORM_WORKER_CTX
    _TRANSFORM_WORKER_CTX = ctx


def _transform_source_chunk_worker(source_file: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    ctx = _TRANSFORM_WORKER_CTX
    chunk = _load_prepared_source_chunk(
        Path(ctx["clean_path"]),
        source_file,
        ctx["columns"],
        ctx["window_enabled"],
        ctx["window_scope"],
        ctx["window_cfg"],
    )
    if chunk.empty:
        return {}

    result = {}
    for name, split_ids in ctx["split_ids"].items():
        chunk_ids = chunk.index.intersection(pd.Index(split_ids), sort=False)
        if chunk_ids.empty:
            continue
        part = chunk.loc[chunk_ids].copy()
        raw = part[ctx["final_feature_columns"]].to_numpy(dtype=np.float32, copy=True)
        family = part["attack_family"].to_numpy(copy=True)
        row_ids = part["row_id"].to_numpy(dtype="int64", copy=True)
        transformed = ctx["preprocessor"].transform(
            raw,
            device=ctx["transform_device"],
            batch_rows=ctx["preprocess_batch_rows"],
        )
        y = (family != "benign").astype("int64")
        result[name] = (transformed, row_ids, y, family)
    return result


def preprocess_experiment(
    experiment: str = "zero_day_dos",
    clean_path: str | Path = "data/interim/cicids2017_clean.parquet",
    split_dir: str | Path = "data/splits/zero_day_dos",
    schema_path: str | Path = "data/interim/column_schema.json",
    window_config_path: str | Path | None = None,
    window_config: dict | None = None,
    benchmark_mode: str | None = None,
    fit_sample_rows: int = 400_000,
    preprocess_device: str = "cpu",
    preprocess_batch_rows: int = 262_144,
    preprocess_tmp_dir: str | Path | None = None,
    preprocess_num_workers: int = 0,
) -> Path:
    if preprocess_batch_rows <= 0:
        raise ValueError("preprocess_batch_rows phải > 0")
    resolved_preprocess_device = IDSPreprocessor.resolve_device(preprocess_device)
    split_dir = Path(split_dir)
    clean_path = Path(clean_path)
    processed_dir = ensure_dir(Path("data/processed") / experiment)
    preprocessor_dir = ensure_dir(Path("artifacts/preprocessors") / experiment)
    tmp_output_dir = Path(preprocess_tmp_dir) / experiment if preprocess_tmp_dir is not None else None

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

    split_names = _resolve_split_names(split_dir)
    split_ids = {name: _load_split_ids(split_dir, name) for name in split_names}
    split_indices = {name: pd.Index(ids) for name, ids in split_ids.items()}
    files = _source_files(clean_path)
    effective_workers = _effective_preprocess_workers(
        int(preprocess_num_workers),
        resolved_preprocess_device,
        window_enabled,
        window_scope,
    )
    if preprocess_num_workers and effective_workers == 1 and int(preprocess_num_workers) > 1:
        print(
            "[preprocess] file parallelism disabled because it is only supported for "
            "CPU full_source_file window preprocessing"
        )

    train_chunks: list[np.ndarray] = []
    window_columns: list[str] = window_feature_names(window_cfg) if window_enabled else []
    final_feature_columns = feature_columns + window_columns if window_enabled else list(feature_columns)
    train_total = int(len(split_ids["train"]))
    sample_ratio = min(1.0, float(fit_sample_rows) / max(train_total, 1))
    if window_scope == "full_source_file":
        fit_tasks = [
            (
                str(clean_path),
                source_file,
                columns,
                window_enabled,
                window_scope,
                window_cfg,
                split_ids["train"],
                final_feature_columns,
                sample_ratio,
                42 + file_idx,
            )
            for file_idx, source_file in enumerate(files)
        ]
        if effective_workers > 1:
            print(f"[preprocess] fitting source files with {effective_workers} workers")
            with mp.Pool(processes=effective_workers) as pool:
                for train_chunk in pool.imap(_fit_source_chunk_worker, fit_tasks):
                    if train_chunk is not None:
                        train_chunks.append(train_chunk)
        else:
            for task in fit_tasks:
                train_chunk = _fit_source_chunk_worker(task)
                if train_chunk is not None:
                    train_chunks.append(train_chunk)
    else:
        rng = np.random.default_rng(42)
        for source_file in files:
            chunk = pd.read_parquet(clean_path, columns=columns, filters=[("source_file", "==", source_file)])
            if chunk.empty:
                continue
            chunk = chunk.set_index("row_id", drop=False)
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
    _log_preprocess_device(preprocess_device, resolved_preprocess_device, preprocess_batch_rows)
    print(f"[preprocess] source-file workers: {effective_workers}")
    del train_matrix
    del train_chunks

    outputs = {}
    for name in split_names:
        total_rows = int(len(split_ids[name]))
        final_x_path = processed_dir / f"X_{name}.npy"
        tmp_x_path = _tmp_output_path(final_x_path, tmp_output_dir)
        if tmp_x_path.exists():
            tmp_x_path.unlink()
        print(
            f"[preprocess] writing {name}: rows={total_rows}, features={len(final_feature_columns)}, "
            f"tmp={tmp_x_path}"
        )
        outputs[name] = {
            "final_x_path": final_x_path,
            "tmp_x_path": tmp_x_path,
            "X": open_memmap(tmp_x_path, mode="w+", dtype="float32", shape=(total_rows, len(final_feature_columns))),
            "row_id": np.empty(total_rows, dtype="int64"),
            "y": np.empty(total_rows, dtype="int64"),
            "family": np.empty(total_rows, dtype=object),
            "offset": 0,
            "total_rows": total_rows,
        }

    def write_result(result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
        for name, (transformed, row_ids, y_arr, family_arr) in result.items():
            output = outputs[name]
            offset = output["offset"]
            next_offset = offset + len(transformed)
            output["X"][offset:next_offset] = transformed
            output["row_id"][offset:next_offset] = row_ids
            output["y"][offset:next_offset] = y_arr
            output["family"][offset:next_offset] = family_arr
            output["offset"] = next_offset

    if window_scope == "full_source_file":
        worker_ctx = {
            "clean_path": str(clean_path),
            "columns": columns,
            "window_enabled": window_enabled,
            "window_scope": window_scope,
            "window_cfg": window_cfg,
            "split_ids": split_ids,
            "final_feature_columns": final_feature_columns,
            "preprocessor": preprocessor,
            "preprocess_batch_rows": preprocess_batch_rows,
            "transform_device": resolved_preprocess_device if effective_workers == 1 else "cpu",
        }
        if effective_workers > 1:
            print(f"[preprocess] transforming source files with {effective_workers} workers")
            with mp.Pool(
                processes=effective_workers,
                initializer=_init_transform_worker,
                initargs=(worker_ctx,),
            ) as pool:
                for result in pool.imap(_transform_source_chunk_worker, files):
                    write_result(result)
        else:
            _init_transform_worker(worker_ctx)
            for source_file in files:
                write_result(_transform_source_chunk_worker(source_file))
    else:
        for name in split_names:
            output = outputs[name]
            for source_file in files:
                chunk = pd.read_parquet(clean_path, columns=columns, filters=[("source_file", "==", source_file)])
                if chunk.empty:
                    continue
                chunk = chunk.set_index("row_id", drop=False)

                if window_enabled and window_scope == "causal_past_only":
                    # For causal past, val can see train+val, test can see train+val+test
                    allowed_splits = ["train"]
                    if name == "model_selection_val":
                        allowed_splits.extend(["model_selection_val"])
                    if name == "val":
                        allowed_splits.extend([split for split in ("model_selection_val", "val") if split in split_indices])
                    if name == "test_seen":
                        allowed_splits.extend(
                            [split for split in ("model_selection_val", "val", "test_seen") if split in split_indices]
                        )
                    if name == "test_zero_day":
                        allowed_splits.extend(
                            [split for split in ("model_selection_val", "val", "test_zero_day") if split in split_indices]
                        )
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
                transformed = preprocessor.transform(
                    raw,
                    device=resolved_preprocess_device,
                    batch_rows=preprocess_batch_rows,
                )
                write_result({name: (transformed, row_ids, (family != "benign").astype("int64"), family)})

    for name in split_names:
        output = outputs[name]
        offset = int(output["offset"])
        total_rows = int(output["total_rows"])
        if offset == 0:
            raise RuntimeError(f"Split {name} rỗng cho {experiment}")
        if offset != total_rows:
            raise RuntimeError(f"Split {name} ghi {offset} rows nhưng mong đợi {total_rows}")
        output["X"].flush()
        del output["X"]
        _move_tmp_output(output["tmp_x_path"], output["final_x_path"])
        _save_npy(processed_dir / f"row_id_{name}.npy", output["row_id"], tmp_output_dir)
        _save_npy(processed_dir / f"y_{name}.npy", output["y"], tmp_output_dir)
        _save_npy(processed_dir / f"family_{name}.npy", output["family"], tmp_output_dir)
        print(f"[preprocess] finished {name}: rows={offset}")

    schema_payload = {
        "experiment": experiment,
        "benchmark_mode": benchmark_mode,
        "clean_path": str(clean_path),
        "split_dir": str(split_dir),
        "schema_path": str(schema_path),
        "window_config_path": str(window_config_path) if window_config_path is not None else None,
        "feature_order": final_feature_columns,
        "split_names": list(split_names),
        "model_selection_split": "model_selection_val" if "model_selection_val" in split_names else None,
        "total_features_after_preprocessing": len(final_feature_columns),
        "scaler_type": "StandardScaler",
        "imputation": "median",
        "invalid_negative_columns": invalid_negative_columns,
        "clip_quantiles": [0.001, 0.999],
        "no_clip_feature_patterns": ["ctx_*", "is_*", "*_is_*", "*_indicator", "*_indicator_*"],
        "no_clip_features": [
            final_feature_columns[idx]
            for idx in getattr(preprocessor, "no_clip_indices", [])
        ],
        "clip_fit_scope": "train split only",
        "fit_scope": "train split only",
        "fit_sample_rows": int(min(fit_sample_rows, train_total)),
        "preprocess_transform_backend": resolved_preprocess_device,
        "preprocess_transform_batch_rows": int(preprocess_batch_rows),
        "preprocess_num_workers": int(effective_workers),
        "preprocess_tmp_dir": str(tmp_output_dir) if tmp_output_dir is not None else None,
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

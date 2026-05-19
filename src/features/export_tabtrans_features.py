from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from src.models.tabtrans.model import NumericTabTransformer
from src.utils.io import ensure_dir, read_json, write_json


REQUIRED_SPLITS = ("train", "val", "test_seen", "test_zero_day")


def _cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _processed_split_names(processed_dir: Path, processed_schema: dict | None = None) -> tuple[str, ...]:
    schema_names = list((processed_schema or {}).get("split_names", []))
    if schema_names:
        return tuple(name for name in schema_names if (processed_dir / f"X_{name}.npy").exists())
    names = list(REQUIRED_SPLITS)
    known = set(names)
    for path in sorted(processed_dir.glob("X_*.npy")):
        name = path.stem.removeprefix("X_")
        if name not in known:
            names.append(name)
            known.add(name)
    return tuple(names)


def _raw_feature_indices(feature_order: list[str], patterns: list[str] | None) -> list[int]:
    if not patterns:
        return list(range(len(feature_order)))
    selected = []
    for idx, name in enumerate(feature_order):
        if any(pattern in name for pattern in patterns):
            selected.append(idx)
    if not selected:
        raise ValueError(f"No raw processed input features matched patterns: {patterns}")
    return selected


def _extract_to_memmap(
    model: torch.nn.Module,
    X: np.ndarray,
    output_path: Path,
    device: torch.device,
    feature_dim: int,
    batch_size: int = 4096,
    include_raw_input: bool = False,
    raw_input_indices: list[int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_amp: bool = False,
) -> list[int]:
    if num_workers:
        raise ValueError("export_tabtrans_features currently requires num_workers=0 for memmap-safe batch loading")
    out = open_memmap(output_path, mode="w+", dtype="float32", shape=(len(X), feature_dim))
    offset = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch_np = np.asarray(X[start : start + batch_size], dtype=np.float32).copy()
            batch = torch.from_numpy(batch_np).to(device, non_blocking=pin_memory)
            with _cuda_autocast(use_amp and device.type == "cuda"):
                _, features_t = model(batch, return_features=True)
            features = features_t.detach().cpu().numpy().astype("float32")
            if include_raw_input:
                raw = batch.detach().cpu().numpy().astype("float32")
                if raw_input_indices is not None:
                    raw = raw[:, raw_input_indices]
                features = np.concatenate([features, raw], axis=1).astype("float32", copy=False)
            if not np.isfinite(features).all():
                raise AssertionError(f"Feature batch for {output_path.name} has NaN/inf")
            if features.shape[1] != feature_dim:
                raise AssertionError(
                    f"Feature batch for {output_path.name} has {features.shape[1]} columns, expected {feature_dim}"
                )
            next_offset = offset + len(features)
            out[offset:next_offset] = features
            offset = next_offset
    del out
    return [int(len(X)), int(feature_dim)]


def export_tabtrans_features(
    experiment: str,
    batch_size: int = 4096,
    artifact_name: str | None = None,
    feature_set: str | None = None,
    include_raw_input: bool = False,
    raw_input_feature_patterns: list[str] | None = None,
    data_parallel: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    amp: bool = False,
) -> Path:
    processed_dir = Path("data/processed") / experiment
    feature_dir = ensure_dir(Path("data/features") / (feature_set or experiment))
    checkpoint = torch.load(
        Path("artifacts/tabtrans") / (artifact_name or experiment) / "tabtrans_best.pt",
        map_location="cpu",
    )
    if checkpoint.get("architecture") != "tabtrans":
        raise ValueError("TabTransformer checkpoint must have architecture='tabtrans'")
    processed_shape = np.load(processed_dir / "X_train.npy", mmap_mode="r").shape
    if int(checkpoint["input_dim"]) != int(processed_shape[1]):
        raise ValueError(
            "TabTransformer checkpoint input_dim does not match current processed data: "
            f"checkpoint input_dim={int(checkpoint['input_dim'])}, X_train columns={int(processed_shape[1])}. "
            "Rerun TabTransformer training for this experiment/artifact before exporting features."
        )
    model = NumericTabTransformer(checkpoint["input_dim"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_data_parallel = bool(data_parallel) and device.type == "cuda" and torch.cuda.device_count() > 1
    model_for_export: torch.nn.Module = torch.nn.DataParallel(model) if use_data_parallel else model
    use_pin_memory = bool(pin_memory) if pin_memory is not None else device.type == "cuda"
    use_amp = bool(amp) and device.type == "cuda"

    input_dim = int(checkpoint["input_dim"])
    representation_dim = int(checkpoint["model_config"]["latent_dim"])
    processed_schema_path = processed_dir / "feature_schema.json"
    processed_schema = read_json(processed_schema_path) if processed_schema_path.exists() else {}
    feature_order = list(processed_schema.get("feature_order", []))
    if include_raw_input and raw_input_feature_patterns and not feature_order:
        raise ValueError("raw_input_feature_patterns requires processed feature_schema.json with feature_order")
    raw_input_indices = (
        _raw_feature_indices(feature_order or [f"raw_{idx}" for idx in range(input_dim)], raw_input_feature_patterns)
        if include_raw_input
        else []
    )
    raw_input_dim = len(raw_input_indices) if include_raw_input else 0
    feature_dim = int(representation_dim + raw_input_dim)
    split_names = _processed_split_names(processed_dir, processed_schema)
    dims = {}
    for name in split_names:
        X = np.load(processed_dir / f"X_{name}.npy", mmap_mode="r")
        dims[name] = _extract_to_memmap(
            model_for_export,
            X,
            feature_dir / f"F_{name}.npy",
            device,
            feature_dim,
            batch_size=batch_size,
            include_raw_input=include_raw_input,
            raw_input_indices=raw_input_indices,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            use_amp=use_amp,
        )

    feature_blocks = ["tabtrans_latent"]
    if include_raw_input:
        feature_blocks.append("raw_processed_input")
    schema = {
        "architecture": "tabtrans",
        "experiment": experiment,
        "feature_set": feature_set or experiment,
        "artifact_name": artifact_name or experiment,
        "D_value": input_dim,
        "representation_feature_dim": representation_dim,
        "include_raw_input": bool(include_raw_input),
        "raw_input_dim": int(raw_input_dim),
        "raw_input_feature_patterns": list(raw_input_feature_patterns or []),
        "raw_input_feature_indices": list(raw_input_indices),
        "raw_input_feature_names": [
            feature_order[idx] if idx < len(feature_order) else f"raw_{idx}" for idx in raw_input_indices
        ],
        "total_dims_numeric": feature_dim,
        "processed_feature_count": len(processed_schema.get("feature_order", [])),
        "processed_benchmark_mode": processed_schema.get("benchmark_mode"),
        "processed_window_features": processed_schema.get("window_features"),
        "device": str(device),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "data_parallel": bool(use_data_parallel),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(use_pin_memory),
        "amp": bool(use_amp),
        "split_names": list(split_names),
        "model_selection_split": "model_selection_val" if "model_selection_val" in split_names else None,
        "shapes": dims,
        "feature_blocks": feature_blocks,
    }
    write_json(feature_dir / "representation_feature_schema.json", schema)
    write_json(feature_dir / "tabtrans_feature_schema.json", schema)
    return feature_dir

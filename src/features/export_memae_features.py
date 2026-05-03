from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from numpy.lib.format import open_memmap

from src.models.memae.model import MemAE
from src.utils.io import ensure_dir, read_json, write_json


def _memae_feature_dim(input_dim: int, latent_dim: int) -> int:
    return int(4 + 2 * input_dim + 3 * latent_dim)


def _feature_dim(input_dim: int, latent_dim: int, raw_input_dim: int = 0) -> int:
    raw_dim = int(raw_input_dim)
    return int(_memae_feature_dim(input_dim, latent_dim) + raw_dim)


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


def _batch_features(model: MemAE, batch: torch.Tensor) -> torch.Tensor:
    x_hat, z, z_hat, attn = model(batch)
    re_scalar = ((batch - x_hat) ** 2).sum(dim=1, keepdim=True)
    residual = batch - x_hat
    abs_residual = torch.abs(residual)
    latent_deviation = z - z_hat
    attn_entropy = (-attn * torch.log(attn + 1e-12)).sum(dim=1, keepdim=True)
    attn_sparsity = (attn > 1e-4).float().sum(dim=1, keepdim=True)
    attn_max = attn.max(dim=1, keepdim=True).values
    return torch.cat(
        [re_scalar, residual, abs_residual, z, z_hat, latent_deviation, attn_entropy, attn_sparsity, attn_max],
        dim=1,
    )


def _extract_to_memmap(
    model: MemAE,
    X: np.ndarray,
    output_path: Path,
    device: torch.device,
    feature_dim: int,
    batch_size: int = 4096,
    include_raw_input: bool = False,
    raw_input_indices: list[int] | None = None,
) -> list[int]:
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32))), batch_size=batch_size)
    out = open_memmap(output_path, mode="w+", dtype="float32", shape=(len(X), feature_dim))
    offset = 0
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            features = _batch_features(model, batch).detach().cpu().numpy().astype("float32")
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


def export_features(
    experiment: str,
    batch_size: int = 4096,
    artifact_name: str | None = None,
    feature_set: str | None = None,
    include_raw_input: bool = False,
    raw_input_feature_patterns: list[str] | None = None,
) -> Path:
    processed_dir = Path("data/processed") / experiment
    feature_dir = ensure_dir(Path("data/features") / (feature_set or experiment))
    checkpoint = torch.load(Path("artifacts/memae") / (artifact_name or experiment) / "memae_best.pt", map_location="cpu")
    model = MemAE(checkpoint["input_dim"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    D = int(checkpoint["input_dim"])
    C = int(checkpoint["model_config"]["latent_dim"])
    memae_feature_dim = _memae_feature_dim(D, C)
    processed_schema_path = processed_dir / "feature_schema.json"
    processed_schema = read_json(processed_schema_path) if processed_schema_path.exists() else {}
    feature_order = list(processed_schema.get("feature_order", []))
    if include_raw_input and raw_input_feature_patterns and not feature_order:
        raise ValueError("raw_input_feature_patterns requires processed feature_schema.json with feature_order")
    raw_input_indices = (
        _raw_feature_indices(feature_order or [f"raw_{idx}" for idx in range(D)], raw_input_feature_patterns)
        if include_raw_input
        else []
    )
    raw_input_dim = len(raw_input_indices) if include_raw_input else 0
    feature_dim = _feature_dim(D, C, raw_input_dim=raw_input_dim)
    dims = {}
    for name in ("train", "val", "test_seen", "test_zero_day"):
        X = np.load(processed_dir / f"X_{name}.npy", mmap_mode="r")
        dims[name] = _extract_to_memmap(
            model,
            X,
            feature_dir / f"F_{name}.npy",
            device,
            feature_dim,
            batch_size=batch_size,
            include_raw_input=include_raw_input,
            raw_input_indices=raw_input_indices,
        )

    feature_blocks = [
        "re_scalar",
        "residual",
        "abs_residual",
        "latent_z",
        "latent_z_hat",
        "latent_deviation",
        "attn_entropy",
        "attn_sparsity",
        "attn_max",
    ]
    if include_raw_input:
        feature_blocks.append("raw_processed_input")

    write_json(
        feature_dir / "memae_feature_schema.json",
        {
            "experiment": experiment,
            "feature_set": feature_set or experiment,
            "artifact_name": artifact_name or experiment,
            "D_value": D,
            "C_value": C,
            "include_raw_input": bool(include_raw_input),
            "raw_input_dim": int(raw_input_dim),
            "raw_input_feature_patterns": list(raw_input_feature_patterns or []),
            "raw_input_feature_indices": list(raw_input_indices),
            "raw_input_feature_names": [
                feature_order[idx] if idx < len(feature_order) else f"raw_{idx}" for idx in raw_input_indices
            ],
            "memae_feature_dim": int(memae_feature_dim),
            "total_dims_numeric": feature_dim,
            "processed_feature_count": len(processed_schema.get("feature_order", [])),
            "processed_benchmark_mode": processed_schema.get("benchmark_mode"),
            "processed_window_features": processed_schema.get("window_features"),
            "shapes": dims,
            "feature_blocks": feature_blocks,
        },
    )
    return feature_dir

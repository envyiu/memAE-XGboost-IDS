from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.memae.model import MemAE, memae_loss
from src.utils.io import ensure_dir, write_json
from src.utils.scoring import threshold_for_fpr
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _cuda_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _sample(X: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if not max_samples or len(X) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx]


def _reconstruction_errors(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    reduction: str = "sum",
    num_workers: int = 0,
    pin_memory: bool = False,
    use_amp: bool = False,
) -> np.ndarray:
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"Unknown reconstruction reduction: {reduction}")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32))),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    errors = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=pin_memory)
            with _cuda_autocast(use_amp and device.type == "cuda"):
                x_hat, _, _, _ = model(batch)
            squared = (batch - x_hat) ** 2
            err = squared.sum(dim=1) if reduction == "sum" else squared.mean(dim=1)
            errors.append(err.detach().cpu().numpy())
    return np.concatenate(errors) if errors else np.array([], dtype=np.float32)


def _reconstruction_stats(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
) -> dict:
    values = _reconstruction_errors(
        model,
        X,
        device,
        batch_size,
        reduction="mean",
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_amp=use_amp,
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _unwrap_model(model: torch.nn.Module) -> MemAE:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _model_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    return _unwrap_model(model).state_dict()


def _memory_diversity_loss(model: torch.nn.Module) -> torch.Tensor:
    return _unwrap_model(model).memory_diversity_loss()


def _resolve_torch_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda was requested but CUDA is not available")
        return torch.device("cuda")
    raise ValueError(f"Unknown training.device: {requested}")


def _validate_xy_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"X_{name} must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y_{name} must be 1D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X_{name} and y_{name} lengths differ: {len(X)} != {len(y)}")


def _validation_split_name(processed_dir: Path) -> str:
    if (processed_dir / "X_model_selection_val.npy").exists() and (processed_dir / "y_model_selection_val.npy").exists():
        return "model_selection_val"
    return "val"


def train_memae(experiment: str, config: dict, seed: int = 42, artifact_name: str | None = None) -> Path:
    set_global_seed(seed)
    processed_dir = Path("data/processed") / experiment
    artifact_dir = ensure_dir(Path("artifacts/memae") / (artifact_name or experiment))
    validation_split = _validation_split_name(processed_dir)
    X_train = np.load(processed_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy", mmap_mode="r")
    X_val = np.load(processed_dir / f"X_{validation_split}.npy", mmap_mode="r")
    y_val = np.load(processed_dir / f"y_{validation_split}.npy", mmap_mode="r")
    _validate_xy_split("train", X_train, y_train)
    _validate_xy_split(validation_split, X_val, y_val)
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(f"Train/val feature dimensions differ: {X_train.shape[1]} != {X_val.shape[1]}")

    train_benign = np.asarray(X_train[y_train == 0], dtype=np.float32)
    val_benign = np.asarray(X_val[y_val == 0], dtype=np.float32)
    val_seen_attack = np.asarray(X_val[y_val == 1], dtype=np.float32)
    training_cfg = config["training"]
    selection_cfg = config.get("selection", {})
    selection_metric = selection_cfg.get("metric", "val_loss")
    target_fpr = float(selection_cfg.get("target_fpr", 0.01))
    train_benign = _sample(train_benign, training_cfg.get("max_train_samples"), seed)
    val_benign = _sample(val_benign, training_cfg.get("max_val_samples"), seed + 1)
    val_attack_sample = _sample(val_seen_attack, training_cfg.get("max_val_samples"), seed + 2)
    if len(train_benign) == 0:
        raise ValueError("MemAE training requires at least one benign train sample")
    if len(val_benign) == 0:
        raise ValueError("MemAE training requires at least one benign validation sample")

    device = _resolve_torch_device(training_cfg.get("device", "auto"))
    model_cfg = config["model"]
    base_model = MemAE(input_dim=train_benign.shape[1], **model_cfg).to(device)
    use_data_parallel = bool(training_cfg.get("data_parallel", False)) and device.type == "cuda" and torch.cuda.device_count() > 1
    model: torch.nn.Module = torch.nn.DataParallel(base_model) if use_data_parallel else base_model
    diversity_weight = float(training_cfg.get("memory_diversity_weight", 0.0))
    batch_size = int(training_cfg["batch_size"])
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))
    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg["patience"])
    min_epochs = int(training_cfg.get("min_epochs", 1))
    if batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("batch_size and eval_batch_size must be > 0")
    if epochs <= 0:
        raise ValueError("training.epochs must be > 0")
    if patience <= 0:
        raise ValueError("training.patience must be > 0")
    if min_epochs <= 0:
        raise ValueError("training.min_epochs must be > 0")
    min_epochs = min(min_epochs, epochs)
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda"))
    use_amp = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    effective_selection_metric = selection_metric
    min_attack_samples = int(selection_cfg.get("min_seen_attack_samples", 10))
    if selection_metric == "seen_recall_at_benign_fpr" and len(val_attack_sample) < min_attack_samples:
        logger.warning(
            "Falling back to val_loss model selection: val_seen_attack has %d samples (< %d).",
            len(val_attack_sample),
            min_attack_samples,
        )
        effective_selection_metric = "val_loss"
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_cfg.get("lr_scheduler_factor", 0.5),
        patience=training_cfg.get("lr_scheduler_patience", 3),
        min_lr=training_cfg.get("min_learning_rate", 1e-5),
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_benign)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_benign)),
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    best_val = float("inf")
    best_selection_value = float("-inf")
    best_selection_summary = None
    best_epoch = -1
    stale = 0
    history = []
    scaler = _cuda_grad_scaler(use_amp)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in tqdm(train_loader, desc=f"MemAE epoch {epoch}", leave=False):
            batch = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with _cuda_autocast(use_amp):
                x_hat, _, _, attn = model(batch)
                loss, recon, entropy, diversity = memae_loss(
                    batch,
                    x_hat,
                    attn,
                    training_cfg["entropy_weight"],
                    diversity_loss=_memory_diversity_loss(model),
                    diversity_weight=diversity_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=pin_memory)
                with _cuda_autocast(use_amp):
                    x_hat, _, _, attn = model(batch)
                    loss, _, _, _ = memae_loss(
                        batch,
                        x_hat,
                        attn,
                        training_cfg["entropy_weight"],
                        diversity_loss=_memory_diversity_loss(model),
                        diversity_weight=diversity_weight,
                    )
                val_losses.append(float(loss.detach().cpu()))

        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        if effective_selection_metric == "seen_recall_at_benign_fpr":
            benign_score = _reconstruction_errors(
                model,
                val_benign,
                device,
                eval_batch_size,
                reduction="sum",
                num_workers=num_workers,
                pin_memory=pin_memory,
                use_amp=use_amp,
            )
            attack_score = (
                _reconstruction_errors(
                    model,
                    val_attack_sample,
                    device,
                    eval_batch_size,
                    reduction="sum",
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    use_amp=use_amp,
                )
                if len(val_attack_sample)
                else np.array([], dtype=np.float32)
            )
            selected_threshold = threshold_for_fpr(benign_score, target_fpr, fallback_mode="nextafter")
            threshold = float(selected_threshold["threshold"])
            val_fpr = float(selected_threshold["calibration_fpr"])
            val_seen_recall = float((attack_score >= threshold).mean()) if len(attack_score) else 0.0
            row["selection"] = {
                "metric": effective_selection_metric,
                "target_fpr": target_fpr,
                "threshold": threshold,
                "val_benign_fpr": val_fpr,
                "val_seen_attack_recall": val_seen_recall,
            }
        history.append(row)
        is_better = False
        if epoch >= min_epochs:
            if effective_selection_metric == "seen_recall_at_benign_fpr":
                selection_value = row["selection"]["val_seen_attack_recall"]
                if selection_value > best_selection_value + 1e-12 or (
                    abs(selection_value - best_selection_value) <= 1e-12 and row["val_loss"] < best_val
                ):
                    best_selection_value = selection_value
                    best_val = row["val_loss"]
                    best_selection_summary = row["selection"]
                    is_better = True
            elif row["val_loss"] < best_val:
                best_val = row["val_loss"]
                is_better = True

        if is_better:
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": _model_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "input_dim": train_benign.shape[1],
                    "model_config": model_cfg,
                    "training_config": dict(training_cfg),
                    "selection_config": dict(selection_cfg),
                    "selection_metric": effective_selection_metric,
                    "selection_summary": row.get("selection"),
                    "train_split": "train",
                    "validation_split": validation_split,
                    "best_selection_value": (
                        best_selection_value if effective_selection_metric == "seen_recall_at_benign_fpr" else None
                    ),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "seed": seed,
                },
                artifact_dir / "memae_best.pt",
            )
        else:
            stale = 0 if epoch < min_epochs else stale + 1
            if stale >= patience:
                break
        scheduler.step(row["val_loss"])

    checkpoint = torch.load(artifact_dir / "memae_best.pt", map_location=device)
    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    reconstruction_stats = {
        "val_benign": _reconstruction_stats(
            model,
            val_benign,
            device,
            eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_amp=use_amp,
        ),
        "val_seen_attack": _reconstruction_stats(
            model,
            val_attack_sample,
            device,
            eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_amp=use_amp,
        )
        if len(val_attack_sample)
        else None,
    }
    write_json(
        artifact_dir / "training_log.json",
        {
            "experiment": experiment,
            "train_split": "train",
            "validation_split": validation_split,
            "device": str(device),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "data_parallel": bool(use_data_parallel),
            "amp": bool(use_amp),
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "min_epochs": min_epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "configured_selection_metric": selection_metric,
            "selection_metric": effective_selection_metric,
            "selection_target_fpr": target_fpr if effective_selection_metric == "seen_recall_at_benign_fpr" else None,
            "best_selection_value": best_selection_value if effective_selection_metric == "seen_recall_at_benign_fpr" else None,
            "best_selection_summary": best_selection_summary,
            "memory_diversity_weight": diversity_weight,
            "selection_fallback_reason": (
                f"val_seen_attack samples {len(val_attack_sample)} < {min_attack_samples}"
                if effective_selection_metric != selection_metric
                else None
            ),
            "train_benign_samples_used": int(len(train_benign)),
            "val_benign_samples_used": int(len(val_benign)),
            "val_seen_attack_samples_used_for_sanity": int(len(val_attack_sample)),
            "reconstruction_stats": reconstruction_stats,
            "history": history,
        },
    )
    return artifact_dir / "memae_best.pt"

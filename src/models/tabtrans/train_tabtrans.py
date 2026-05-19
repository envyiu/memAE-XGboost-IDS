from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.tabtrans.model import NumericTabTransformer
from src.utils.io import ensure_dir, write_json
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


def _validation_split_name(processed_dir: Path) -> str:
    if (processed_dir / "X_model_selection_val.npy").exists() and (processed_dir / "y_model_selection_val.npy").exists():
        return "model_selection_val"
    return "val"


def _validate_xy_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"X_{name} must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y_{name} must be 1D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X_{name} and y_{name} lengths differ: {len(X)} != {len(y)}")


def _sample_xy(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not max_samples or len(X) <= max_samples:
        return np.array(X, dtype=np.float32, copy=True), np.array(y, dtype=np.float32, copy=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=max_samples, replace=False))
    return np.array(X[idx], dtype=np.float32, copy=True), np.array(y[idx], dtype=np.float32, copy=True)


def _unwrap_model(model: torch.nn.Module) -> NumericTabTransformer:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _model_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    return _unwrap_model(model).state_dict()


def _classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(y_true)) < 2:
        return {"aucpr": None, "roc_auc": None}
    return {
        "aucpr": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }


def _validation_scores(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
    criterion: torch.nn.Module,
) -> tuple[float, np.ndarray, dict[str, float | None]]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.float32))),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    losses: list[float] = []
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch, target in loader:
            batch = batch.to(device, non_blocking=pin_memory)
            target = target.to(device, non_blocking=pin_memory)
            with _cuda_autocast(use_amp and device.type == "cuda"):
                logits = model(batch)
                loss = criterion(logits, target)
            losses.append(float(loss.detach().cpu()))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    prob = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    return float(np.mean(losses)), prob, _classification_metrics(np.asarray(y, dtype=np.int64), prob)


def train_tabtrans(experiment: str, config: dict, seed: int = 42, artifact_name: str | None = None) -> Path:
    set_global_seed(seed)
    processed_dir = Path("data/processed") / experiment
    artifact_dir = ensure_dir(Path("artifacts/tabtrans") / (artifact_name or experiment))
    validation_split = _validation_split_name(processed_dir)
    X_train = np.load(processed_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy", mmap_mode="r")
    X_val = np.load(processed_dir / f"X_{validation_split}.npy", mmap_mode="r")
    y_val = np.load(processed_dir / f"y_{validation_split}.npy", mmap_mode="r")
    _validate_xy_split("train", X_train, y_train)
    _validate_xy_split(validation_split, X_val, y_val)
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(f"Train/val feature dimensions differ: {X_train.shape[1]} != {X_val.shape[1]}")

    training_cfg = config["training"]
    selection_cfg = config.get("selection", {})
    X_train_sample, y_train_sample = _sample_xy(X_train, y_train, training_cfg.get("max_train_samples"), seed)
    X_val_sample, y_val_sample = _sample_xy(X_val, y_val, training_cfg.get("max_val_samples"), seed + 1)
    if len(np.unique(y_train_sample.astype(np.int64))) < 2:
        raise ValueError("TabTransformer training requires both benign and attack samples in train")
    if len(X_val_sample) == 0:
        raise ValueError("TabTransformer training requires at least one validation sample")

    device = _resolve_torch_device(training_cfg.get("device", "auto"))
    model_cfg = config["model"]
    base_model = NumericTabTransformer(input_dim=int(X_train.shape[1]), **model_cfg).to(device)
    use_data_parallel = bool(training_cfg.get("data_parallel", False)) and device.type == "cuda" and torch.cuda.device_count() > 1
    model: torch.nn.Module = torch.nn.DataParallel(base_model) if use_data_parallel else base_model

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
    min_epochs = min(max(1, min_epochs), epochs)
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda"))
    use_amp = bool(training_cfg.get("amp", False)) and device.type == "cuda"

    neg = int((y_train_sample == 0).sum())
    pos = int((y_train_sample == 1).sum())
    pos_weight = torch.tensor(float(neg / max(pos, 1)), dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(training_cfg.get("lr_scheduler_patience", 3)),
        min_lr=float(training_cfg.get("min_learning_rate", 1e-5)),
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_sample), torch.from_numpy(y_train_sample.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    configured_selection_metric = selection_cfg.get("metric", "val_aucpr")
    effective_selection_metric = configured_selection_metric
    if configured_selection_metric in {"val_aucpr", "val_roc_auc"} and len(np.unique(y_val_sample.astype(np.int64))) < 2:
        logger.warning("Falling back to val_loss selection because validation has fewer than two classes.")
        effective_selection_metric = "val_loss"

    best_val_loss = float("inf")
    best_selection_value = float("-inf")
    best_epoch = -1
    stale = 0
    history = []
    scaler = _cuda_grad_scaler(use_amp)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch, target in tqdm(train_loader, desc=f"TabTrans epoch {epoch}", leave=False):
            batch = batch.to(device, non_blocking=pin_memory)
            target = target.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with _cuda_autocast(use_amp):
                logits = model(batch)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        val_loss, val_prob, metrics = _validation_scores(
            model,
            X_val_sample,
            y_val_sample,
            device,
            eval_batch_size,
            num_workers,
            pin_memory,
            use_amp,
            criterion,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_aucpr": metrics["aucpr"],
            "val_roc_auc": metrics["roc_auc"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        is_better = False
        if epoch >= min_epochs:
            if effective_selection_metric == "val_loss":
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_better = True
            else:
                metric_value = row[effective_selection_metric]
                if metric_value is not None and float(metric_value) > best_selection_value + 1e-12:
                    best_selection_value = float(metric_value)
                    best_val_loss = val_loss
                    is_better = True

        if is_better:
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "architecture": "tabtrans",
                    "model_state_dict": _model_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "input_dim": int(X_train.shape[1]),
                    "model_config": dict(model_cfg),
                    "training_config": dict(training_cfg),
                    "selection_config": dict(selection_cfg),
                    "configured_selection_metric": configured_selection_metric,
                    "selection_metric": effective_selection_metric,
                    "best_selection_value": (
                        best_selection_value if effective_selection_metric != "val_loss" else None
                    ),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "train_split": "train",
                    "validation_split": validation_split,
                    "seed": seed,
                },
                artifact_dir / "tabtrans_best.pt",
            )
        else:
            stale = 0 if epoch < min_epochs else stale + 1
            if stale >= patience:
                break
        scheduler.step(val_loss)

    checkpoint_path = artifact_dir / "tabtrans_best.pt"
    if not checkpoint_path.exists():
        raise RuntimeError("TabTransformer training did not produce a checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    final_val_loss, final_val_prob, final_metrics = _validation_scores(
        model,
        X_val_sample,
        y_val_sample,
        device,
        eval_batch_size,
        num_workers,
        pin_memory,
        use_amp,
        criterion,
    )
    write_json(
        artifact_dir / "training_log.json",
        {
            "architecture": "tabtrans",
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
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_val_loss": float(checkpoint["best_val_loss"]),
            "configured_selection_metric": configured_selection_metric,
            "selection_metric": effective_selection_metric,
            "best_selection_value": checkpoint.get("best_selection_value"),
            "train_samples_used": int(len(y_train_sample)),
            "val_samples_used": int(len(y_val_sample)),
            "class_counts_train": {"benign": neg, "malicious": pos},
            "pos_weight": float(pos_weight.detach().cpu()),
            "final_val_loss": final_val_loss,
            "final_val_aucpr": final_metrics["aucpr"],
            "final_val_roc_auc": final_metrics["roc_auc"],
            "history": history,
        },
    )
    np.save(artifact_dir / "val_score.npy", final_val_prob.astype(np.float32))
    return checkpoint_path

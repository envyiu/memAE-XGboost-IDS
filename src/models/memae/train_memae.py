from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.memae.model import MemAE, memae_loss
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _sample(X: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if not max_samples or len(X) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx]


def _reconstruction_errors(
    model: MemAE,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    reduction: str = "sum",
) -> np.ndarray:
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"Unknown reconstruction reduction: {reduction}")
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32))), batch_size=batch_size)
    errors = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, _, _, _ = model(batch)
            squared = (batch - x_hat) ** 2
            err = squared.sum(dim=1) if reduction == "sum" else squared.mean(dim=1)
            errors.append(err.detach().cpu().numpy())
    return np.concatenate(errors) if errors else np.array([], dtype=np.float32)


def _reconstruction_stats(model: MemAE, X: np.ndarray, device: torch.device, batch_size: int) -> dict:
    values = _reconstruction_errors(model, X, device, batch_size, reduction="mean")
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def train_memae(experiment: str, config: dict, seed: int = 42, artifact_name: str | None = None) -> Path:
    set_global_seed(seed)
    processed_dir = Path("data/processed") / experiment
    artifact_dir = ensure_dir(Path("artifacts/memae") / (artifact_name or experiment))
    X_train = np.load(processed_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy", mmap_mode="r")
    X_val = np.load(processed_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(processed_dir / "y_val.npy", mmap_mode="r")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = MemAE(input_dim=train_benign.shape[1], **model_cfg).to(device)
    diversity_weight = float(training_cfg.get("memory_diversity_weight", 0.0))
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
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_benign)), batch_size=training_cfg["batch_size"])

    best_val = float("inf")
    best_selection_value = float("-inf")
    best_selection_summary = None
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, training_cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for (batch,) in tqdm(train_loader, desc=f"MemAE epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            x_hat, _, _, attn = model(batch)
            loss, recon, entropy, diversity = memae_loss(
                batch,
                x_hat,
                attn,
                training_cfg["entropy_weight"],
                diversity_loss=model.memory_diversity_loss(),
                diversity_weight=diversity_weight,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x_hat, _, _, attn = model(batch)
                loss, _, _, _ = memae_loss(
                    batch,
                    x_hat,
                    attn,
                    training_cfg["entropy_weight"],
                    diversity_loss=model.memory_diversity_loss(),
                    diversity_weight=diversity_weight,
                )
                val_losses.append(float(loss.detach().cpu()))

        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        if selection_metric == "seen_recall_at_benign_fpr":
            benign_score = _reconstruction_errors(model, val_benign, device, training_cfg["batch_size"], reduction="sum")
            attack_score = (
                _reconstruction_errors(model, val_attack_sample, device, training_cfg["batch_size"], reduction="sum")
                if len(val_attack_sample)
                else np.array([], dtype=np.float32)
            )
            threshold = float(np.quantile(benign_score, 1.0 - target_fpr))
            val_fpr = float((benign_score >= threshold).mean()) if len(benign_score) else 0.0
            val_seen_recall = float((attack_score >= threshold).mean()) if len(attack_score) else 0.0
            row["selection"] = {
                "metric": selection_metric,
                "target_fpr": target_fpr,
                "threshold": threshold,
                "val_benign_fpr": val_fpr,
                "val_seen_attack_recall": val_seen_recall,
            }
        history.append(row)
        is_better = False
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
                    "model_state_dict": model.state_dict(),
                    "input_dim": train_benign.shape[1],
                    "model_config": model_cfg,
                    "selection_metric": effective_selection_metric,
                },
                artifact_dir / "memae_best.pt",
            )
        else:
            stale += 1
            if stale >= training_cfg["patience"]:
                break
        scheduler.step(row["val_loss"])

    checkpoint = torch.load(artifact_dir / "memae_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    reconstruction_stats = {
        "val_benign": _reconstruction_stats(model, val_benign, device, training_cfg["batch_size"]),
        "val_seen_attack": _reconstruction_stats(model, val_attack_sample, device, training_cfg["batch_size"])
        if len(val_attack_sample)
        else None,
    }
    write_json(
        artifact_dir / "training_log.json",
        {
            "experiment": experiment,
            "device": str(device),
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

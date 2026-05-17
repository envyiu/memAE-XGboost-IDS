from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.utils.io import read_json

DEFAULT_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)


def attach_selected_feature_indices(model: Any, artifact_dir: Path) -> None:
    metadata_path = artifact_dir / "feature_selection.json"
    if not metadata_path.exists():
        return
    metadata = read_json(metadata_path)
    selected = metadata.get("selected_indices")
    if selected is not None:
        setattr(model, "_selected_feature_indices", np.asarray(selected, dtype=np.int64))


def _model_input(X: np.ndarray, model: Any) -> np.ndarray:
    selected = getattr(model, "_selected_feature_indices", None)
    if selected is None:
        return X
    return X[:, selected]


def predict_prob(model: Any, X: np.ndarray) -> np.ndarray:
    X_model = _model_input(X, model)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict_proba(X_model, iteration_range=(0, best_iteration + 1))[:, 1]
    return model.predict_proba(X_model)[:, 1]


def predict_prob_batched(model: Any, X: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        out[start:end] = predict_prob(model, X[start:end]).astype(np.float32)
    return out


def _as_1d_array(name: str, values: np.ndarray, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")
    return arr


def _validate_same_length(**arrays: np.ndarray) -> None:
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Input lengths must match, got {lengths}")


def threshold_for_fpr(
    benign_score: np.ndarray,
    target_fpr: float,
    add_jitter: bool = False,
    fallback_mode: str = "nextafter",
) -> dict[str, float]:
    target_fpr = float(target_fpr)
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    benign_score = _as_1d_array("benign_score", benign_score, dtype=np.float64)
    if benign_score.size == 0:
        raise ValueError("benign_score must not be empty")
    if not np.isfinite(benign_score).all():
        raise ValueError("benign_score must contain only finite values")
    if target_fpr == 0.0:
        return {
            "threshold": float(np.nextafter(float(np.max(benign_score)), np.inf)),
            "calibration_fpr": 0.0,
            "target_fpr": target_fpr,
        }
    score_to_use = benign_score
    if add_jitter and np.unique(score_to_use).size <= 10:
        rng = np.random.default_rng(42)
        score_to_use = score_to_use + rng.uniform(0, 1e-7, size=score_to_use.size)

    sorted_score = np.sort(score_to_use)
    thresholds = np.unique(score_to_use)
    thresholds.sort()
    ge_counts = score_to_use.size - np.searchsorted(sorted_score, thresholds, side="left")
    fprs = ge_counts / score_to_use.size
    valid = np.flatnonzero(fprs <= target_fpr)
    if valid.size == 0:
        if fallback_mode == "percentile":
            threshold = float(np.percentile(score_to_use, 100.0 * (1.0 - target_fpr)))
            actual_fpr = float((benign_score >= threshold).mean())
            if actual_fpr > target_fpr:
                threshold = float(np.nextafter(float(np.max(benign_score)), np.inf))
                actual_fpr = 0.0
        elif fallback_mode == "nextafter":
            threshold = float(np.nextafter(float(np.max(benign_score)), np.inf))
            actual_fpr = 0.0
        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")
    else:
        threshold = float(thresholds[valid[0]])
        actual_fpr = float((benign_score >= threshold).mean())
    return {"threshold": threshold, "calibration_fpr": actual_fpr, "target_fpr": target_fpr}


def metrics_from_pred(y_true: np.ndarray, family: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y_true = _as_1d_array("y_true", y_true)
    family = _as_1d_array("family", family)
    pred = _as_1d_array("pred", pred)
    _validate_same_length(y_true=y_true, family=family, pred=pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    attack_mask = family != "benign"
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "z_dr": float((pred[attack_mask] == 1).mean()) if attack_mask.any() else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def metrics_at_threshold(y_true: np.ndarray, family: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    score = _as_1d_array("score", score)
    _validate_same_length(y_true=np.asarray(y_true), family=np.asarray(family), score=score)
    pred = (score >= threshold).astype("int64")
    return {"threshold": float(threshold), **metrics_from_pred(y_true, family, pred)}


def fpr_drift_ratio(test_fpr: float, cal_fpr: float) -> float:
    return float(test_fpr / max(float(cal_fpr), 1e-12))


def calibration_benign_scores(
    mode: str,
    val: dict[str, np.ndarray],
    test_seen: dict[str, np.ndarray],
    val_score: np.ndarray,
    test_seen_score: np.ndarray,
) -> np.ndarray:
    val_score = _as_1d_array("val_score", val_score)
    test_seen_score = _as_1d_array("test_seen_score", test_seen_score)
    val_family = _as_1d_array("val_family", val["family"])
    test_seen_family = _as_1d_array("test_seen_family", test_seen["family"])
    _validate_same_length(val_family=val_family, val_score=val_score)
    _validate_same_length(test_seen_family=test_seen_family, test_seen_score=test_seen_score)
    val_benign = val_score[val_family == "benign"]
    if mode == "val_only":
        return val_benign
    if mode == "val_plus_test_seen_benign":
        seen_benign = test_seen_score[test_seen_family == "benign"]
        return np.concatenate([val_benign, seen_benign])
    raise ValueError(f"Unknown calibration_mode: {mode}")

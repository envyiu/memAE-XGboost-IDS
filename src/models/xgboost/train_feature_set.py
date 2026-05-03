from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

from src.models.xgboost.threshold_optimizer import optimize_threshold
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.seed import set_global_seed


def _sample_xy(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
    family_arr: np.ndarray | None = None,
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    if cfg and cfg.get("family_balance") and family_arr is not None:
        idx_list = []
        max_atk = cfg.get("max_samples_per_attack_family", 50000)
        ratio = cfg.get("benign_to_attack_ratio", 3.0)
        atk_total = 0
        for fam in np.unique(family_arr):
            mask = family_arr == fam
            fam_idx = np.flatnonzero(mask)
            if fam == "benign": continue
            if len(fam_idx) > max_atk:
                fam_idx = rng.choice(fam_idx, size=max_atk, replace=False)
            idx_list.extend(fam_idx)
            atk_total += len(fam_idx)
        
        ben_mask = family_arr == "benign"
        ben_idx = np.flatnonzero(ben_mask)
        target_ben = int(atk_total * ratio)
        if len(ben_idx) > target_ben:
            ben_idx = rng.choice(ben_idx, size=target_ben, replace=False)
        idx_list.extend(ben_idx)
        
        idx = np.asarray(idx_list)
        if max_samples and len(idx) > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
    else:
        if not max_samples or len(X) <= max_samples:
            return np.asarray(X, dtype=np.float32), np.asarray(y), None
        idx = rng.choice(len(X), size=max_samples, replace=False)
    
    idx = np.sort(idx)
    return X[idx], y[idx], idx


def _value_counts(values: np.ndarray | None) -> dict[str, int]:
    if values is None:
        return {}
    unique, counts = np.unique(values.astype(str), return_counts=True)
    return {str(key): int(count) for key, count in zip(unique, counts, strict=False)}


def _predict_prob(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict_proba(X, iteration_range=(0, best_iteration + 1))[:, 1]
    return model.predict_proba(X)[:, 1]


def train_xgboost_feature_set(
    experiment: str,
    feature_set: str,
    config: dict,
    seed: int = 42,
) -> dict:
    set_global_seed(seed)
    feature_dir = Path("data/features") / feature_set
    processed_dir = Path("data/processed") / experiment
    artifact_dir = ensure_dir(Path("artifacts/xgboost") / feature_set)

    F_train = np.load(feature_dir / "F_train.npy", mmap_mode="r")
    F_val = np.load(feature_dir / "F_val.npy", mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")

    train_cfg = config.get("training", {})
    family_train = np.load(processed_dir / "family_train.npy", allow_pickle=True) if (processed_dir / "family_train.npy").exists() else None
    F_train_sample, y_train_sample, train_idx = _sample_xy(
        F_train,
        y_train,
        train_cfg.get("max_train_samples"),
        seed,
        family_arr=family_train,
        cfg=train_cfg,
    )
    F_val_sample, y_val_sample, _ = _sample_xy(F_val, y_val, train_cfg.get("max_val_samples"), seed + 1)
    family_train_sample = family_train[train_idx] if family_train is not None and train_idx is not None else family_train
    neg = int((y_train_sample == 0).sum())
    pos = int((y_train_sample == 1).sum())
    scale_pos_weight = float(neg / pos) if pos else 1.0

    params = dict(config["binary_detection"])
    early_stopping_rounds = params.pop("early_stopping_rounds")
    params["scale_pos_weight"] = scale_pos_weight
    params["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.XGBClassifier(**params)
    model.fit(F_train_sample, y_train_sample, eval_set=[(F_val_sample, y_val_sample)], verbose=True)
    model.save_model(artifact_dir / "xgboost_model.json")

    y_val_prob = _predict_prob(model, F_val_sample)
    threshold = optimize_threshold(
        y_val_sample,
        y_val_prob,
        min_t=config["threshold"]["min"],
        max_t=config["threshold"]["max"],
        steps=config["threshold"]["steps"],
    )
    booster = model.get_booster()
    importance = {
        "gain": booster.get_score(importance_type="gain"),
        "weight": booster.get_score(importance_type="weight"),
        "cover": booster.get_score(importance_type="cover"),
    }
    feature_schema = read_json(feature_dir / "memae_feature_schema.json")
    log = {
        "experiment": experiment,
        "feature_set": feature_set,
        "feature_dims": int(F_train.shape[1]),
        "feature_schema": feature_schema,
        "train_samples_used": int(len(y_train_sample)),
        "val_samples_used": int(len(y_val_sample)),
        "class_counts_train": {"benign": neg, "malicious": pos},
        "family_counts_train_used": _value_counts(family_train_sample),
        "family_balance_enabled": bool(train_cfg.get("family_balance") and family_train is not None),
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", 0.0)),
        "threshold": threshold,
        "evals_result": model.evals_result(),
    }
    write_json(artifact_dir / "training_log.json", log)
    write_json(artifact_dir / "feature_importance.json", importance)
    write_json(artifact_dir / "threshold.json", threshold)
    return {"model": model, "threshold": threshold["threshold"], "artifact_dir": str(artifact_dir)}

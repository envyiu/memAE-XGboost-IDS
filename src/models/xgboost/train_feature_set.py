from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.models.xgboost.threshold_optimizer import optimize_threshold
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.scoring import predict_prob
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


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


def _qualified_attack_family_counts(
    family: np.ndarray | None,
    y: np.ndarray,
    min_samples_per_family: int,
) -> dict[str, int]:
    if family is None:
        return {}
    attack_family = family[np.asarray(y) == 1]
    counts = _value_counts(attack_family)
    return {
        name: count
        for name, count in counts.items()
        if name != "benign" and count >= min_samples_per_family
    }


def _select_features(
    model: xgb.XGBClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = 0.0,
    n_repeats: int = 3,
    seed: int = 42,
    protected_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    selected = np.flatnonzero(result.importances_mean > threshold)
    protected = np.asarray(protected_indices if protected_indices is not None else [], dtype=np.int64)
    protected = protected[(protected >= 0) & (protected < X_val.shape[1])]
    if protected.size:
        selected = np.union1d(selected, protected)
    if selected.size == 0:
        logger.warning("Feature selection selected zero features; keeping all %d features.", X_val.shape[1])
        selected = np.arange(X_val.shape[1])
    metadata = {
        "enabled": True,
        "method": "permutation_importance",
        "scoring": "average_precision",
        "threshold": float(threshold),
        "n_repeats": int(n_repeats),
        "selected_indices": selected.astype(int).tolist(),
        "selected_feature_count": int(selected.size),
        "original_feature_count": int(X_val.shape[1]),
        "protected_indices": protected.astype(int).tolist(),
        "importances_mean": result.importances_mean.astype(float).tolist(),
        "importances_std": result.importances_std.astype(float).tolist(),
    }
    return selected.astype(np.int64), metadata


def _memae_protected_feature_indices(feature_schema: dict, train_cfg: dict) -> np.ndarray:
    configured = train_cfg.get("feature_selection_protected_indices")
    if configured is not None:
        return np.asarray(configured, dtype=np.int64)
    if not train_cfg.get("feature_selection_keep_memae_scalars", True):
        return np.array([], dtype=np.int64)
    memae_dim = int(feature_schema.get("memae_feature_dim", feature_schema.get("total_dims_numeric", 0)) or 0)
    input_dim = int(feature_schema.get("D_value", 0) or 0)
    latent_dim = int(feature_schema.get("C_value", 0) or 0)
    protected = [0]
    scalar_start = 1 + 2 * input_dim + 3 * latent_dim
    protected.extend([scalar_start, scalar_start + 1, scalar_start + 2])
    return np.asarray([idx for idx in protected if 0 <= idx < memae_dim], dtype=np.int64)


def _validation_split_name(feature_dir: Path, processed_dir: Path) -> str:
    if (
        (feature_dir / "F_model_selection_val.npy").exists()
        and (processed_dir / "y_model_selection_val.npy").exists()
        and (processed_dir / "family_model_selection_val.npy").exists()
    ):
        return "model_selection_val"
    return "val"


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
    validation_split = _validation_split_name(feature_dir, processed_dir)

    F_train = np.load(feature_dir / "F_train.npy", mmap_mode="r")
    F_val = np.load(feature_dir / f"F_{validation_split}.npy", mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / f"y_{validation_split}.npy")

    train_cfg = config.get("training", {})
    family_train = np.load(processed_dir / "family_train.npy", allow_pickle=True) if (processed_dir / "family_train.npy").exists() else None
    family_val_path = processed_dir / f"family_{validation_split}.npy"
    family_val = np.load(family_val_path, allow_pickle=True) if family_val_path.exists() else None
    F_train_sample, y_train_sample, train_idx = _sample_xy(
        F_train,
        y_train,
        train_cfg.get("max_train_samples"),
        seed,
        family_arr=family_train,
        cfg=train_cfg,
    )
    F_val_sample, y_val_sample, val_idx = _sample_xy(F_val, y_val, train_cfg.get("max_val_samples"), seed + 1)
    family_train_sample = family_train[train_idx] if family_train is not None and train_idx is not None else family_train
    family_val_sample = family_val[val_idx] if family_val is not None and val_idx is not None else family_val
    eval_positive_count = int((y_val_sample == 1).sum())
    min_eval_positive = int(train_cfg.get("min_eval_attack_samples", 50))
    min_eval_attack_families = int(train_cfg.get("min_eval_attack_families", 2))
    min_eval_samples_per_family = int(train_cfg.get("min_eval_samples_per_attack_family", 10))
    qualified_eval_families = _qualified_attack_family_counts(
        family_val_sample,
        y_val_sample,
        min_eval_samples_per_family,
    )
    eval_source = validation_split
    eval_fallback_reason = None
    F_fit = F_train_sample
    y_fit = y_train_sample
    F_eval = F_val_sample
    y_eval = y_val_sample
    use_internal_eval = bool(train_cfg.get("always_use_internal_eval", False))
    if eval_positive_count < min_eval_positive:
        use_internal_eval = True
        eval_fallback_reason = (
            f"{validation_split} has {eval_positive_count} positive samples (< {min_eval_positive})"
        )
    elif family_val_sample is not None and len(qualified_eval_families) < min_eval_attack_families:
        use_internal_eval = True
        eval_fallback_reason = (
            f"{validation_split} has only "
            f"{len(qualified_eval_families)} attack families with >= {min_eval_samples_per_family} samples "
            f"(< {min_eval_attack_families})"
        )

    if use_internal_eval and int((y_train_sample == 1).sum()) >= 2:
        eval_fraction = float(train_cfg.get("internal_eval_fraction", 0.2))
        F_fit, F_eval, y_fit, y_eval = train_test_split(
            F_train_sample,
            y_train_sample,
            test_size=eval_fraction,
            random_state=seed + 3,
            stratify=y_train_sample,
        )
        eval_source = "train_stratified_holdout"
        logger.warning(
            "Using train stratified holdout for XGBoost eval: %s.",
            eval_fallback_reason or "always_use_internal_eval=true",
        )
    neg = int((y_fit == 0).sum())
    pos = int((y_fit == 1).sum())
    scale_pos_weight = float(neg / pos) if pos else 1.0

    feature_schema = read_json(feature_dir / "memae_feature_schema.json")
    protected_feature_indices = _memae_protected_feature_indices(feature_schema, train_cfg)

    params = dict(config["binary_detection"])
    early_stopping_rounds = params.pop("early_stopping_rounds")
    params["scale_pos_weight"] = scale_pos_weight
    params["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.XGBClassifier(**params)
    model.fit(F_fit, y_fit, eval_set=[(F_eval, y_eval)], verbose=True)
    selected_indices = None
    feature_selection_metadata = {
        "enabled": False,
        "selected_indices": None,
        "selected_feature_count": int(F_train.shape[1]),
        "original_feature_count": int(F_train.shape[1]),
    }
    if train_cfg.get("feature_selection"):
        selected_indices, feature_selection_metadata = _select_features(
            model,
            F_eval,
            y_eval,
            threshold=float(train_cfg.get("feature_selection_threshold", 0.0)),
            n_repeats=int(train_cfg.get("feature_selection_repeats", 3)),
            seed=seed + 2,
            protected_indices=protected_feature_indices,
        )
        logger.info(
            "Retraining XGBoost with %d/%d selected features.",
            selected_indices.size,
            F_train_sample.shape[1],
        )
        model = xgb.XGBClassifier(**params)
        model.fit(
            F_fit[:, selected_indices],
            y_fit,
            eval_set=[(F_eval[:, selected_indices], y_eval)],
            verbose=True,
        )
        setattr(model, "_selected_feature_indices", selected_indices)

    best_iteration = int(getattr(model, "best_iteration", -1))
    if best_iteration == 0:
        logger.warning(
            "XGBoost failed to learn (best_iteration=0). Consider adjusting hyperparameters or features."
        )
    model.save_model(artifact_dir / "xgboost_model.json")
    write_json(artifact_dir / "feature_selection.json", feature_selection_metadata)

    y_val_prob = predict_prob(model, F_val_sample)
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
    log = {
        "experiment": experiment,
        "feature_set": feature_set,
        "train_split": "train",
        "validation_split": validation_split,
        "feature_dims": int(F_train.shape[1]),
        "model_feature_dims": int(feature_selection_metadata["selected_feature_count"]),
        "feature_schema": feature_schema,
        "train_samples_used": int(len(y_train_sample)),
        "fit_samples_used": int(len(y_fit)),
        "val_samples_used": int(len(y_val_sample)),
        "eval_samples_used": int(len(y_eval)),
        "eval_source": eval_source,
        "eval_fallback_reason": eval_fallback_reason,
        "validation_positive_samples": eval_positive_count,
        "validation_qualified_attack_family_counts": qualified_eval_families,
        "official_val_positive_samples": eval_positive_count,
        "official_val_qualified_attack_family_counts": qualified_eval_families,
        "eval_positive_samples": int((y_eval == 1).sum()),
        "class_counts_train": {"benign": int((y_train_sample == 0).sum()), "malicious": int((y_train_sample == 1).sum())},
        "class_counts_fit": {"benign": neg, "malicious": pos},
        "family_counts_train_used": _value_counts(family_train_sample),
        "family_balance_enabled": bool(train_cfg.get("family_balance") and family_train is not None),
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": best_iteration,
        "best_score": float(getattr(model, "best_score", 0.0)),
        "feature_selection": feature_selection_metadata,
        "threshold": threshold,
        "evals_result": model.evals_result(),
    }
    write_json(artifact_dir / "training_log.json", log)
    write_json(artifact_dir / "feature_importance.json", importance)
    write_json(artifact_dir / "threshold.json", threshold)
    return {"model": model, "threshold": threshold["threshold"], "artifact_dir": str(artifact_dir)}

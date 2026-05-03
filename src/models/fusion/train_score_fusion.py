from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from src.utils.io import ensure_dir, write_json


def _predict_prob(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict_proba(X, iteration_range=(0, best_iteration + 1))[:, 1]
    return model.predict_proba(X)[:, 1]


def _predict_prob_batched(model: xgb.XGBClassifier, X: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        out[start:end] = _predict_prob(model, X[start:end]).astype(np.float32)
    return out


def _fusion_features(xgb_score: np.ndarray, memae_score: np.ndarray) -> np.ndarray:
    xgb_score = np.asarray(xgb_score, dtype=np.float32).reshape(-1, 1)
    memae_score = np.asarray(memae_score, dtype=np.float32).reshape(-1, 1)
    logit_xgb = np.log(np.clip(xgb_score, 1e-6, 1 - 1e-6) / np.clip(1 - xgb_score, 1e-6, 1 - 1e-6))
    log_memae = np.log1p(np.maximum(memae_score, 0.0))
    return np.concatenate(
        [
            xgb_score,
            logit_xgb.astype(np.float32),
            memae_score,
            log_memae.astype(np.float32),
            (xgb_score * log_memae).astype(np.float32),
            np.maximum(xgb_score, np.tanh(log_memae / 10.0)).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def train_score_fusion(
    experiment: str,
    feature_set: str,
    xgboost_artifact: str,
    fusion_artifact: str,
) -> Path:
    processed_dir = Path("data/processed") / experiment
    feature_dir = Path("data/features") / feature_set
    xgb_dir = Path("artifacts/xgboost") / xgboost_artifact
    artifact_dir = ensure_dir(Path("artifacts/fusion") / fusion_artifact)

    model = xgb.XGBClassifier()
    model.load_model(xgb_dir / "xgboost_model.json")
    F_train = np.load(feature_dir / "F_train.npy", mmap_mode="r")
    F_val = np.load(feature_dir / "F_val.npy", mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")

    xgb_train = _predict_prob_batched(model, F_train)
    xgb_val = _predict_prob_batched(model, F_val)
    memae_train = F_train[:, 0]
    memae_val = F_val[:, 0]

    X_train = _fusion_features(xgb_train, memae_train)
    X_val = _fusion_features(xgb_val, memae_val)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, artifact_dir / "fusion_model.joblib")
    write_json(
        artifact_dir / "training_log.json",
        {
            "experiment": experiment,
            "feature_set": feature_set,
            "xgboost_artifact": xgboost_artifact,
            "fusion_artifact": fusion_artifact,
            "fusion_feature_names": [
                "xgb_score",
                "xgb_logit",
                "memae_score",
                "log1p_memae_score",
                "xgb_times_log1p_memae",
                "max_xgb_tanh_memae",
            ],
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
        },
    )
    np.save(artifact_dir / "val_score.npy", clf.predict_proba(X_val)[:, 1].astype(np.float32))
    return artifact_dir

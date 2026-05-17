from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.utils.io import ensure_dir, write_json
from src.utils.scoring import attach_selected_feature_indices, predict_prob_batched


def _fusion_features(xgb_score: np.ndarray, memae_score: np.ndarray) -> np.ndarray:
    xgb_score = np.asarray(xgb_score, dtype=np.float32).reshape(-1, 1)
    memae_score = np.asarray(memae_score, dtype=np.float32).reshape(-1, 1)
    logit_xgb = np.log(np.clip(xgb_score, 1e-6, 1 - 1e-6) / np.clip(1 - xgb_score, 1e-6, 1 - 1e-6))
    log_memae = np.log1p(np.maximum(memae_score, 0.0))
    tanh_memae = np.tanh(log_memae / 10.0).astype(np.float32)
    return np.concatenate(
        [
            xgb_score,
            logit_xgb.astype(np.float32),
            memae_score,
            log_memae.astype(np.float32),
            (xgb_score * log_memae).astype(np.float32),
            np.maximum(xgb_score, tanh_memae).astype(np.float32),
            np.abs(xgb_score - memae_score).astype(np.float32),
            np.minimum(xgb_score, tanh_memae).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _validation_split_name(feature_dir: Path, processed_dir: Path) -> str:
    if (
        (feature_dir / "F_model_selection_val.npy").exists()
        and (processed_dir / "y_model_selection_val.npy").exists()
    ):
        return "model_selection_val"
    return "val"


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
    attach_selected_feature_indices(model, xgb_dir)
    validation_split = _validation_split_name(feature_dir, processed_dir)
    F_train = np.load(feature_dir / "F_train.npy", mmap_mode="r")
    F_val = np.load(feature_dir / f"F_{validation_split}.npy", mmap_mode="r")
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / f"y_{validation_split}.npy")

    xgb_train = predict_prob_batched(model, F_train)
    xgb_val = predict_prob_batched(model, F_val)
    memae_train = F_train[:, 0]
    memae_val = F_val[:, 0]

    X_train = _fusion_features(xgb_train, memae_train)
    X_val = _fusion_features(xgb_val, memae_val)
    base = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    clf.fit(X_train, y_train)
    joblib.dump(clf, artifact_dir / "fusion_model.joblib")
    write_json(
        artifact_dir / "training_log.json",
        {
            "experiment": experiment,
            "feature_set": feature_set,
            "xgboost_artifact": xgboost_artifact,
            "fusion_artifact": fusion_artifact,
            "train_split": "train",
            "validation_split": validation_split,
            "fusion_feature_names": [
                "xgb_score",
                "xgb_logit",
                "memae_score",
                "log1p_memae_score",
                "xgb_times_log1p_memae",
                "max_xgb_tanh_memae",
                "abs_xgb_memae_diff",
                "min_xgb_tanh_memae",
            ],
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
        },
    )
    np.save(artifact_dir / "val_score.npy", clf.predict_proba(X_val)[:, 1].astype(np.float32))
    return artifact_dir

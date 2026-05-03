from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.models.fusion.train_score_fusion import _fusion_features
from src.utils.io import ensure_dir, read_json, write_json

DEFAULT_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)


def _predict_prob(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict_proba(X, iteration_range=(0, best_iteration + 1))[:, 1]
    return model.predict_proba(X)[:, 1]


def _threshold_for_fpr(benign_score: np.ndarray, target_fpr: float) -> dict[str, float]:
    sorted_score = np.sort(benign_score)
    thresholds = np.unique(benign_score)
    thresholds.sort()
    ge_counts = benign_score.size - np.searchsorted(sorted_score, thresholds, side="left")
    fprs = ge_counts / benign_score.size
    valid = np.flatnonzero(fprs <= target_fpr)
    if valid.size == 0:
        threshold = float(np.nextafter(float(benign_score.max()), np.inf))
        actual_fpr = 0.0
    else:
        threshold = float(thresholds[valid[0]])
        actual_fpr = float(fprs[valid[0]])
    return {"threshold": threshold, "calibration_fpr": actual_fpr, "target_fpr": float(target_fpr)}


def _metrics(y_true: np.ndarray, family: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    attack_mask = family != "benign"
    return {
        "threshold": float(threshold),
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


def _fpr_drift_ratio(test_fpr: float, calibration_fpr: float) -> float:
    return float(test_fpr / max(float(calibration_fpr), 1e-12))


def _calibration_scores(calibration_mode: str, data: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    val_benign = data["val"]["score"][data["val"]["family"] == "benign"]
    if calibration_mode == "val_only":
        return val_benign
    if calibration_mode == "val_plus_test_seen_benign":
        seen_benign = data["test_seen"]["score"][data["test_seen"]["family"] == "benign"]
        return np.concatenate([val_benign, seen_benign])
    raise ValueError(f"Unknown calibration_mode: {calibration_mode}")


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        rendered = []
        for col in columns:
            value = row.get(col, "")
            rendered.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _render_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        test = row["test_zero_day"]
        out.append(
            {
                "model": row["model_name"],
                "target_fpr": row["target_fpr"],
                "threshold": row["threshold"],
                "cal_fpr": row["calibration_fpr"],
                "val_fpr": row["validation_fpr"],
                "test_fpr": test["fpr"],
                "fpr_drift_ratio": row["fpr_drift_ratio"],
                "zdr": test["z_dr"],
                "f1": test["f1"],
                "status": row["fpr_status"],
            }
        )
    return out


def generate_fusion_calibration_report(
    experiment: str,
    feature_set: str,
    xgboost_artifact: str,
    fusion_artifact: str,
    calibration_mode: str = "val_plus_test_seen_benign",
    fpr_budgets: tuple[float, ...] = DEFAULT_FPR_BUDGETS,
    max_observed_test_fpr: float = 0.05,
    report_dir: Path | None = None,
) -> Path:
    processed_dir = Path("data/processed") / experiment
    feature_dir = Path("data/features") / feature_set
    xgb_dir = Path("artifacts/xgboost") / xgboost_artifact
    fusion_dir = Path("artifacts/fusion") / fusion_artifact
    report_dir = ensure_dir(report_dir or (Path("reports/metrics") / experiment))
    processed_schema_path = processed_dir / "feature_schema.json"
    feature_schema_path = feature_dir / "memae_feature_schema.json"
    processed_schema = read_json(processed_schema_path) if processed_schema_path.exists() else {}
    feature_schema = read_json(feature_schema_path) if feature_schema_path.exists() else {}

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_dir / "xgboost_model.json")
    fusion = joblib.load(fusion_dir / "fusion_model.joblib")

    data = {}
    for split in ("val", "test_seen", "test_zero_day"):
        F = np.load(feature_dir / f"F_{split}.npy", mmap_mode="r")
        y = np.load(processed_dir / f"y_{split}.npy")
        family = np.load(processed_dir / f"family_{split}.npy", allow_pickle=True)
        xgb_score = _predict_prob(xgb_model, F)
        memae_score = np.asarray(F[:, 0], dtype=np.float32)
        fusion_score = fusion.predict_proba(_fusion_features(xgb_score, memae_score))[:, 1]
        data[split] = {"score": fusion_score, "y": y, "family": family}

    calibration_score = _calibration_scores(calibration_mode, data)
    rows = []
    for target_fpr in fpr_budgets:
        selected = _threshold_for_fpr(calibration_score, target_fpr)
        validation = _metrics(data["val"]["y"], data["val"]["family"], data["val"]["score"], selected["threshold"])
        test = _metrics(
            data["test_zero_day"]["y"],
            data["test_zero_day"]["family"],
            data["test_zero_day"]["score"],
            selected["threshold"],
        )
        row = {
            "model_name": "logistic_fusion",
            "score_key": "fusion_score",
            "selection_rule": f"logistic fusion threshold selected at calibration FPR budget <= {target_fpr:.1%}",
            "threshold": selected["threshold"],
            "calibration_fpr": selected["calibration_fpr"],
            "validation_fpr": validation["fpr"],
            "target_fpr": selected["target_fpr"],
            "validation": validation,
            "test_seen": _metrics(
                data["test_seen"]["y"],
                data["test_seen"]["family"],
                data["test_seen"]["score"],
                selected["threshold"],
            ),
            "test_zero_day": test,
            "observed_test_fpr": test["fpr"],
            "fpr_drift_ratio": _fpr_drift_ratio(test["fpr"], selected["calibration_fpr"]),
            "fpr_cap": float(max_observed_test_fpr),
            "fpr_status": "PASS" if test["fpr"] <= max_observed_test_fpr else "FAIL",
        }
        rows.append(row)

    report = {
        "experiment": experiment,
        "feature_set": feature_set,
        "xgboost_artifact": xgboost_artifact,
        "fusion_artifact": fusion_artifact,
        "benchmark_mode": processed_schema.get("benchmark_mode") or feature_schema.get("processed_benchmark_mode"),
        "processed_feature_count": len(processed_schema.get("feature_order", [])),
        "memae_input_dim": feature_schema.get("D_value"),
        "threshold_fit_scope": "calibration benign only",
        "calibration_mode": calibration_mode,
        "fpr_budgets": list(fpr_budgets),
        "max_observed_test_fpr": float(max_observed_test_fpr),
        "rows": rows,
        "candidate_rows": rows,
    }
    suffix = f"_{fusion_artifact}"
    json_path = report_dir / f"fusion_calibration_report{suffix}.json"
    md_path = report_dir / f"fusion_calibration_report{suffix}.md"
    write_json(json_path, report)
    md_path.write_text(
        "# Fusion Calibration Report: "
        + experiment
        + "\n\n"
        + f"- Calibration mode: `{calibration_mode}`\n"
        + f"- FPR cap: `{max_observed_test_fpr:.6f}`\n\n"
        + _markdown_table(
            _render_rows(rows),
            ["model", "target_fpr", "threshold", "cal_fpr", "val_fpr", "test_fpr", "fpr_drift_ratio", "zdr", "f1", "status"],
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path

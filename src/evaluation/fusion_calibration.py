from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from src.models.fusion.train_score_fusion import _fusion_features
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.reporting import markdown_table, render_calibration_rows, with_fpr_status
from src.utils.scoring import (
    DEFAULT_FPR_BUDGETS,
    attach_selected_feature_indices,
    calibration_benign_scores,
    metrics_at_threshold,
    predict_prob,
    threshold_for_fpr,
)


def _validation_split_name(processed_dir: Path, feature_dir: Path) -> str:
    if (
        (feature_dir / "F_model_selection_val.npy").exists()
        and (processed_dir / "y_model_selection_val.npy").exists()
        and (processed_dir / "family_model_selection_val.npy").exists()
    ):
        return "model_selection_val"
    return "val"


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
    attach_selected_feature_indices(xgb_model, xgb_dir)
    fusion = joblib.load(fusion_dir / "fusion_model.joblib")

    data = {}
    validation_split = _validation_split_name(processed_dir, feature_dir)
    for split in (validation_split, "test_seen", "test_zero_day"):
        F = np.load(feature_dir / f"F_{split}.npy", mmap_mode="r")
        y = np.load(processed_dir / f"y_{split}.npy")
        family = np.load(processed_dir / f"family_{split}.npy", allow_pickle=True)
        xgb_score = predict_prob(xgb_model, F)
        memae_score = np.asarray(F[:, 0], dtype=np.float32)
        fusion_score = fusion.predict_proba(_fusion_features(xgb_score, memae_score))[:, 1]
        data[split] = {"score": fusion_score, "y": y, "family": family}
    data["val"] = data[validation_split]

    calibration_score = calibration_benign_scores(
        calibration_mode,
        data["val"],
        data["test_seen"],
        data["val"]["score"],
        data["test_seen"]["score"],
    )
    rows = []
    for target_fpr in fpr_budgets:
        selected = threshold_for_fpr(calibration_score, target_fpr, fallback_mode="nextafter")
        validation = metrics_at_threshold(data["val"]["y"], data["val"]["family"], data["val"]["score"], selected["threshold"])
        test = metrics_at_threshold(
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
            "test_seen": metrics_at_threshold(
                data["test_seen"]["y"],
                data["test_seen"]["family"],
                data["test_seen"]["score"],
                selected["threshold"],
            ),
            "test_zero_day": test,
        }
        rows.append(with_fpr_status(row, max_observed_test_fpr))

    report = {
        "experiment": experiment,
        "feature_set": feature_set,
        "xgboost_artifact": xgboost_artifact,
        "fusion_artifact": fusion_artifact,
        "benchmark_mode": processed_schema.get("benchmark_mode") or feature_schema.get("processed_benchmark_mode"),
        "validation_split": validation_split,
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
        + f"- Validation split: `{validation_split}`\n"
        + f"- FPR cap: `{max_observed_test_fpr:.6f}`\n\n"
        + markdown_table(
            render_calibration_rows(rows),
            ["model", "target_fpr", "threshold", "cal_fpr", "val_fpr", "test_fpr", "fpr_drift_ratio", "zdr", "f1", "status"],
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path

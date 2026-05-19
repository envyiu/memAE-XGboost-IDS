from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from src.utils.io import ensure_dir, read_json, write_json
from src.utils.reporting import markdown_table, render_calibration_rows, with_fpr_status
from src.utils.scoring import (
    DEFAULT_FPR_BUDGETS,
    attach_selected_feature_indices,
    calibration_benign_scores,
    metrics_from_pred,
    predict_prob,
    threshold_for_fpr,
)


def _load_split(processed_dir: Path, feature_dir: Path, split: str) -> dict[str, np.ndarray]:
    return {
        "F": np.load(feature_dir / f"F_{split}.npy", mmap_mode="r"),
        "y": np.load(processed_dir / f"y_{split}.npy"),
        "family": np.load(processed_dir / f"family_{split}.npy", allow_pickle=True),
    }


def _validation_split_name(processed_dir: Path, feature_dir: Path) -> str:
    if (
        (feature_dir / "F_model_selection_val.npy").exists()
        and (processed_dir / "y_model_selection_val.npy").exists()
        and (processed_dir / "family_model_selection_val.npy").exists()
    ):
        return "model_selection_val"
    return "val"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Detector Calibration Report: {report['experiment']}",
        "",
        f"- Calibration mode: `{report['calibration_mode']}`",
        f"- Validation split: `{report['validation_split']}`",
        f"- FPR cap: `{report['max_observed_test_fpr']:.6f}`",
        "- Thresholds are selected from calibration benign scores only and evaluated unchanged on `test_zero_day`.",
        "",
    ]

    columns = ["model", "target_fpr", "threshold", "cal_fpr", "val_fpr", "test_fpr", "fpr_drift_ratio", "zdr", "f1", "status"]
    lines.append("## OR Fusion Budget Grid")
    lines.append(markdown_table(render_calibration_rows(report["or_fusion_budget_grid"]), columns))
    lines.append("")

    fine_rows = []
    for row in report["or_fusion_fine_grid_top_by_validation_recall"]:
        test = row["test_zero_day"]
        fine_rows.append(
            {
                "total_budget": row["total_budget"],
                "memae_budget": row["memae_budget"],
                "xgb_budget": row["xgboost_budget"],
                "val_fpr": row["validation"]["fpr"],
                "test_fpr": test["fpr"],
                "zdr": test["z_dr"],
                "f1": test["f1"],
                "status": row["fpr_status"],
            }
        )
    lines.append("## OR Fusion Fine Grid: Top By Validation Recall")
    lines.append(markdown_table(fine_rows, ["total_budget", "memae_budget", "xgb_budget", "val_fpr", "test_fpr", "zdr", "f1", "status"]))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_xgboost_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# XGBoost Calibration Report: {report['experiment']}",
        "",
        f"- Calibration mode: `{report['calibration_mode']}`",
        f"- Validation split: `{report['validation_split']}`",
        f"- FPR cap: `{report['max_observed_test_fpr']:.6f}`",
        "- Thresholds are selected from calibration benign scores only and evaluated unchanged on `test_zero_day`.",
        "",
    ]
    columns = ["model", "target_fpr", "threshold", "cal_fpr", "val_fpr", "test_fpr", "fpr_drift_ratio", "zdr", "f1", "status"]
    lines.append("## XGBoost Budget Grid")
    lines.append(markdown_table(render_calibration_rows(report["xgboost_budget_grid"]), columns))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _feature_schema_path(feature_dir: Path) -> Path:
    generic = feature_dir / "representation_feature_schema.json"
    if generic.exists():
        return generic
    return feature_dir / "memae_feature_schema.json"


def generate_xgboost_calibration_report(
    experiment: str = "zero_day_dos",
    feature_set: str | None = None,
    xgboost_artifact: str | None = None,
    calibration_mode: str = "val_plus_test_seen_benign",
    fpr_budgets: tuple[float, ...] = DEFAULT_FPR_BUDGETS,
    max_observed_test_fpr: float = 0.05,
    report_dir: Path | None = None,
) -> Path:
    feature_set = feature_set or experiment
    xgboost_artifact = xgboost_artifact or feature_set
    processed_dir = Path("data/processed") / experiment
    feature_dir = Path("data/features") / feature_set
    xgb_dir = Path("artifacts/xgboost") / xgboost_artifact
    report_dir = ensure_dir(report_dir or (Path("reports/metrics") / experiment))

    model = xgb.XGBClassifier()
    model.load_model(xgb_dir / "xgboost_model.json")
    attach_selected_feature_indices(model, xgb_dir)
    processed_schema_path = processed_dir / "feature_schema.json"
    feature_schema_path = _feature_schema_path(feature_dir)
    processed_schema = read_json(processed_schema_path) if processed_schema_path.exists() else {}
    feature_schema = read_json(feature_schema_path) if feature_schema_path.exists() else {}

    validation_split = _validation_split_name(processed_dir, feature_dir)
    val = _load_split(processed_dir, feature_dir, validation_split)
    test_seen = _load_split(processed_dir, feature_dir, "test_seen")
    test_zero_day = _load_split(processed_dir, feature_dir, "test_zero_day")
    xgb_val_score = predict_prob(model, val["F"])
    xgb_seen_score = predict_prob(model, test_seen["F"])
    xgb_test_score = predict_prob(model, test_zero_day["F"])
    xgb_calibration = calibration_benign_scores(calibration_mode, val, test_seen, xgb_val_score, xgb_seen_score)

    rows = []
    for target_fpr in fpr_budgets:
        selected = threshold_for_fpr(xgb_calibration, target_fpr, add_jitter=True, fallback_mode="percentile")
        threshold = float(selected["threshold"])
        val_pred = (xgb_val_score >= threshold).astype("int64")
        seen_pred = (xgb_seen_score >= threshold).astype("int64")
        test_pred = (xgb_test_score >= threshold).astype("int64")
        row = {
            "model_name": "xgboost",
            "score_key": "xgboost_probability",
            "selection_rule": f"XGBoost threshold selected at calibration FPR budget <= {target_fpr:.1%}",
            "target_fpr": float(target_fpr),
            "threshold": threshold,
            "calibration_fpr": float(selected["calibration_fpr"]),
            "validation": metrics_from_pred(val["y"], val["family"], val_pred),
            "test_seen": metrics_from_pred(test_seen["y"], test_seen["family"], seen_pred),
            "test_zero_day": metrics_from_pred(test_zero_day["y"], test_zero_day["family"], test_pred),
        }
        row["validation_fpr"] = float(row["validation"]["fpr"])
        rows.append(with_fpr_status(row, max_observed_test_fpr))

    report = {
        "experiment": experiment,
        "feature_set": feature_set,
        "xgboost_artifact": xgboost_artifact,
        "benchmark_mode": processed_schema.get("benchmark_mode") or feature_schema.get("processed_benchmark_mode"),
        "architecture": feature_schema.get("architecture", "unknown"),
        "validation_split": validation_split,
        "processed_feature_count": len(processed_schema.get("feature_order", [])),
        "representation_input_dim": feature_schema.get("D_value"),
        "representation_feature_dim": feature_schema.get("representation_feature_dim"),
        "threshold_fit_scope": "calibration benign only",
        "calibration_mode": calibration_mode,
        "fpr_budgets": list(fpr_budgets),
        "max_observed_test_fpr": float(max_observed_test_fpr),
        "xgboost_budget_grid": rows,
        "candidate_rows": rows,
    }

    suffix = "" if feature_set == experiment and xgboost_artifact == experiment else f"_{xgboost_artifact}"
    json_path = report_dir / f"xgboost_calibration_report{suffix}.json"
    md_path = report_dir / f"xgboost_calibration_report{suffix}.md"
    write_json(json_path, report)
    _write_xgboost_markdown(md_path, report)
    return json_path


def generate_detector_calibration_report(
    experiment: str = "zero_day_dos",
    feature_set: str | None = None,
    xgboost_artifact: str | None = None,
    calibration_mode: str = "val_plus_test_seen_benign",
    fpr_budgets: tuple[float, ...] = DEFAULT_FPR_BUDGETS,
    max_observed_test_fpr: float = 0.05,
    report_dir: Path | None = None,
) -> Path:
    feature_set = feature_set or experiment
    xgboost_artifact = xgboost_artifact or feature_set
    processed_dir = Path("data/processed") / experiment
    feature_dir = Path("data/features") / feature_set
    xgb_dir = Path("artifacts/xgboost") / xgboost_artifact
    report_dir = ensure_dir(report_dir or (Path("reports/metrics") / experiment))

    model = xgb.XGBClassifier()
    model.load_model(xgb_dir / "xgboost_model.json")
    attach_selected_feature_indices(model, xgb_dir)
    processed_schema_path = processed_dir / "feature_schema.json"
    feature_schema_path = _feature_schema_path(feature_dir)
    processed_schema = read_json(processed_schema_path) if processed_schema_path.exists() else {}
    feature_schema = read_json(feature_schema_path) if feature_schema_path.exists() else {}

    validation_split = _validation_split_name(processed_dir, feature_dir)
    val = _load_split(processed_dir, feature_dir, validation_split)
    test_seen = _load_split(processed_dir, feature_dir, "test_seen")
    test_zero_day = _load_split(processed_dir, feature_dir, "test_zero_day")
    xgb_val_score = predict_prob(model, val["F"])
    xgb_seen_score = predict_prob(model, test_seen["F"])
    xgb_test_score = predict_prob(model, test_zero_day["F"])
    memae_val_score = np.asarray(val["F"][:, 0], dtype=np.float32)
    memae_seen_score = np.asarray(test_seen["F"][:, 0], dtype=np.float32)
    memae_test_score = np.asarray(test_zero_day["F"][:, 0], dtype=np.float32)

    xgb_calibration = calibration_benign_scores(calibration_mode, val, test_seen, xgb_val_score, xgb_seen_score)
    memae_calibration = calibration_benign_scores(calibration_mode, val, test_seen, memae_val_score, memae_seen_score)

    budget_pairs = []
    for total_budget in fpr_budgets:
        for ratio in np.linspace(0.0, 1.0, 11):
            budget_pairs.append(
                (total_budget * ratio, total_budget * (1.0 - ratio), total_budget)
            )

    fusion_rows = []

    def make_fusion_row(memae_budget: float, xgb_budget: float, total_budget: float) -> dict[str, Any]:
        tm = threshold_for_fpr(memae_calibration, memae_budget, add_jitter=True, fallback_mode="percentile")
        tx = threshold_for_fpr(xgb_calibration, xgb_budget, add_jitter=True, fallback_mode="percentile")
        val_pred = ((xgb_val_score >= tx["threshold"]) | (memae_val_score >= tm["threshold"])).astype("int64")
        seen_pred = ((xgb_seen_score >= tx["threshold"]) | (memae_seen_score >= tm["threshold"])).astype("int64")
        test_pred = ((xgb_test_score >= tx["threshold"]) | (memae_test_score >= tm["threshold"])).astype("int64")
        row = {
            "model_name": "or_fusion",
            "score_key": "xgb_or_memae",
            "selection_rule": f"OR fusion threshold selected at calibration FPR budget <= {total_budget:.1%}",
            "target_fpr": float(total_budget),
            "memae_budget": float(memae_budget),
            "xgboost_budget": float(xgb_budget),
            "memae_threshold": float(tm["threshold"]),
            "xgboost_threshold": float(tx["threshold"]),
            "calibration_fpr": float(tm["calibration_fpr"] + tx["calibration_fpr"]),
            "validation": metrics_from_pred(val["y"], val["family"], val_pred),
            "test_seen": metrics_from_pred(test_seen["y"], test_seen["family"], seen_pred),
            "test_zero_day": metrics_from_pred(test_zero_day["y"], test_zero_day["family"], test_pred),
            "total_budget": float(total_budget),
        }
        row["validation_fpr"] = float(row["validation"]["fpr"])
        return with_fpr_status(row, max_observed_test_fpr)

    for memae_budget, xgb_budget, total_budget in budget_pairs:
        fusion_rows.append(make_fusion_row(memae_budget, xgb_budget, total_budget))

    valid_fusion = [row for row in fusion_rows if row["validation"]["fpr"] <= 0.01]
    best_fusion = max(valid_fusion, key=lambda row: row["validation"]["recall"], default=None)

    fine_rows = []
    for total_budget in fpr_budgets:
        for memae_budget in np.linspace(0.0, total_budget, 21):
            xgb_budget = float(total_budget - memae_budget)
            fine_rows.append(make_fusion_row(float(memae_budget), xgb_budget, float(total_budget)))
    fine_valid = [row for row in fine_rows if row["validation"]["fpr"] <= max(fpr_budgets)]
    fine_top_by_val_recall = sorted(fine_valid, key=lambda row: row["validation"]["recall"], reverse=True)[:10]

    candidate_rows = fusion_rows
    report = {
        "experiment": experiment,
        "feature_set": feature_set,
        "xgboost_artifact": xgboost_artifact,
        "benchmark_mode": processed_schema.get("benchmark_mode") or feature_schema.get("processed_benchmark_mode"),
        "validation_split": validation_split,
        "processed_feature_count": len(processed_schema.get("feature_order", [])),
        "memae_input_dim": feature_schema.get("D_value"),
        "threshold_fit_scope": "calibration benign only",
        "calibration_mode": calibration_mode,
        "fpr_budgets": list(fpr_budgets),
        "max_observed_test_fpr": float(max_observed_test_fpr),
        "or_fusion_rule": "malicious if xgboost_score >= tx OR memae_reconstruction_error >= tm",
        "or_fusion_budget_grid": fusion_rows,
        "best_or_fusion_under_1pct_val_fpr": best_fusion,
        "or_fusion_fine_grid_top_by_validation_recall": fine_top_by_val_recall,
        "candidate_rows": candidate_rows,
    }

    suffix = "" if feature_set == experiment and xgboost_artifact == experiment else f"_{xgboost_artifact}"
    json_path = report_dir / f"detector_calibration_report{suffix}.json"
    md_path = report_dir / f"detector_calibration_report{suffix}.md"
    write_json(json_path, report)
    _write_markdown(md_path, report)
    return json_path

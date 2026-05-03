from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.utils.io import ensure_dir, read_json, write_json

DEFAULT_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)


def _load_split(processed_dir: Path, feature_dir: Path, split: str) -> dict[str, np.ndarray]:
    return {
        "F": np.load(feature_dir / f"F_{split}.npy", mmap_mode="r"),
        "y": np.load(processed_dir / f"y_{split}.npy"),
        "family": np.load(processed_dir / f"family_{split}.npy", allow_pickle=True),
    }


def _predict_prob(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict_proba(X, iteration_range=(0, best_iteration + 1))[:, 1]
    return model.predict_proba(X)[:, 1]


def _threshold_for_fpr(benign_score: np.ndarray, target_fpr: float) -> dict[str, float]:
    if np.unique(benign_score).size <= 10:
        rng = np.random.default_rng(42)
        score_to_use = benign_score + rng.uniform(0, 1e-7, size=benign_score.size)
    else:
        score_to_use = benign_score

    sorted_score = np.sort(score_to_use)
    thresholds = np.unique(score_to_use)
    thresholds.sort()
    ge_counts = score_to_use.size - np.searchsorted(sorted_score, thresholds, side="left")
    fprs = ge_counts / score_to_use.size
    valid = np.flatnonzero(fprs <= target_fpr)
    
    if valid.size == 0:
        threshold = float(np.percentile(score_to_use, 100.0 * (1.0 - target_fpr)))
        actual_fpr = float((benign_score >= threshold).mean())
    else:
        threshold = float(thresholds[valid[0]])
        actual_fpr = float((benign_score >= threshold).mean())
    return {"threshold": threshold, "calibration_fpr": actual_fpr, "target_fpr": float(target_fpr)}


def _metrics_from_pred(y_true: np.ndarray, family: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
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


def _metrics_at_threshold(y_true: np.ndarray, family: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype("int64")
    return {"threshold": float(threshold), **_metrics_from_pred(y_true, family, pred)}


def _fpr_drift_ratio(test_fpr: float, calibration_fpr: float) -> float:
    denom = max(float(calibration_fpr), 1e-12)
    return float(test_fpr / denom)


def _with_status(row: dict[str, Any], max_observed_test_fpr: float) -> dict[str, Any]:
    test_fpr = float(row["test_zero_day"]["fpr"])
    calibration_fpr = float(row.get("calibration_fpr", row.get("validation_fpr", 0.0)))
    row["observed_test_fpr"] = test_fpr
    row["fpr_drift_ratio"] = _fpr_drift_ratio(test_fpr, calibration_fpr)
    row["fpr_cap"] = float(max_observed_test_fpr)
    row["fpr_status"] = "PASS" if test_fpr <= max_observed_test_fpr else "FAIL"
    return row


def _make_score_row(
    model_name: str,
    score_key: str,
    threshold: float,
    target_fpr: float,
    calibration_fpr: float,
    val: dict[str, np.ndarray],
    test_seen: dict[str, np.ndarray],
    test_zero_day: dict[str, np.ndarray],
    val_score: np.ndarray,
    test_seen_score: np.ndarray,
    test_score: np.ndarray,
    max_observed_test_fpr: float,
) -> dict[str, Any]:
    validation = _metrics_at_threshold(val["y"], val["family"], val_score, threshold)
    row = {
        "model_name": model_name,
        "score_key": score_key,
        "selection_rule": f"{model_name} threshold selected at calibration FPR budget <= {target_fpr:.1%}",
        "threshold": float(threshold),
        "calibration_fpr": float(calibration_fpr),
        "validation_fpr": float(validation["fpr"]),
        "target_fpr": float(target_fpr),
        "validation": validation,
        "test_seen": _metrics_at_threshold(test_seen["y"], test_seen["family"], test_seen_score, threshold),
        "test_zero_day": _metrics_at_threshold(test_zero_day["y"], test_zero_day["family"], test_score, threshold),
    }
    return _with_status(row, max_observed_test_fpr)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _render_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered = []
    for row in rows:
        test = row["test_zero_day"]
        rendered.append(
            {
                "model": row["model_name"],
                "target_fpr": row["target_fpr"],
                "threshold": row.get("threshold", ""),
                "cal_fpr": row.get("calibration_fpr", ""),
                "val_fpr": row.get("validation_fpr", row.get("validation", {}).get("fpr", "")),
                "test_fpr": test["fpr"],
                "fpr_drift_ratio": row.get("fpr_drift_ratio", ""),
                "zdr": test["z_dr"],
                "f1": test["f1"],
                "status": row.get("fpr_status", ""),
            }
        )
    return rendered


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Detector Calibration Report: {report['experiment']}",
        "",
        f"- Calibration mode: `{report['calibration_mode']}`",
        f"- FPR cap: `{report['max_observed_test_fpr']:.6f}`",
        "- Thresholds are selected from calibration benign scores only and evaluated unchanged on `test_zero_day`.",
        "",
    ]

    columns = ["model", "target_fpr", "threshold", "cal_fpr", "val_fpr", "test_fpr", "fpr_drift_ratio", "zdr", "f1", "status"]
    lines.append("## XGBoost Fixed FPR")
    lines.append(_markdown_table(_render_rows(report["xgboost_fixed_fpr"]), columns))
    lines.append("")
    lines.append("## MemAE Fixed FPR")
    lines.append(_markdown_table(_render_rows(report["memae_fixed_fpr"]), columns))
    lines.append("")
    lines.append("## OR Fusion Budget Grid")
    lines.append(_markdown_table(_render_rows(report["or_fusion_budget_grid"]), columns))
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
    lines.append(_markdown_table(fine_rows, ["total_budget", "memae_budget", "xgb_budget", "val_fpr", "test_fpr", "zdr", "f1", "status"]))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _calibration_benign_scores(
    calibration_mode: str,
    val: dict[str, np.ndarray],
    test_seen: dict[str, np.ndarray],
    val_score: np.ndarray,
    test_seen_score: np.ndarray,
) -> np.ndarray:
    val_benign = val_score[val["family"] == "benign"]
    if calibration_mode == "val_only":
        return val_benign
    if calibration_mode == "val_plus_test_seen_benign":
        seen_benign = test_seen_score[test_seen["family"] == "benign"]
        return np.concatenate([val_benign, seen_benign])
    raise ValueError(f"Unknown calibration_mode: {calibration_mode}")


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
    processed_schema_path = processed_dir / "feature_schema.json"
    feature_schema_path = feature_dir / "memae_feature_schema.json"
    processed_schema = read_json(processed_schema_path) if processed_schema_path.exists() else {}
    feature_schema = read_json(feature_schema_path) if feature_schema_path.exists() else {}

    val = _load_split(processed_dir, feature_dir, "val")
    test_seen = _load_split(processed_dir, feature_dir, "test_seen")
    test_zero_day = _load_split(processed_dir, feature_dir, "test_zero_day")
    xgb_val_score = _predict_prob(model, val["F"])
    xgb_seen_score = _predict_prob(model, test_seen["F"])
    xgb_test_score = _predict_prob(model, test_zero_day["F"])
    memae_val_score = np.asarray(val["F"][:, 0], dtype=np.float32)
    memae_seen_score = np.asarray(test_seen["F"][:, 0], dtype=np.float32)
    memae_test_score = np.asarray(test_zero_day["F"][:, 0], dtype=np.float32)

    xgb_calibration = _calibration_benign_scores(calibration_mode, val, test_seen, xgb_val_score, xgb_seen_score)
    memae_calibration = _calibration_benign_scores(calibration_mode, val, test_seen, memae_val_score, memae_seen_score)

    xgb_fixed_rows = []
    memae_fixed_rows = []
    for target_fpr in fpr_budgets:
        xgb_selected = _threshold_for_fpr(xgb_calibration, target_fpr)
        xgb_fixed_rows.append(
            _make_score_row(
                "xgboost",
                "xgb_score",
                xgb_selected["threshold"],
                target_fpr,
                xgb_selected["calibration_fpr"],
                val,
                test_seen,
                test_zero_day,
                xgb_val_score,
                xgb_seen_score,
                xgb_test_score,
                max_observed_test_fpr,
            )
        )
        memae_selected = _threshold_for_fpr(memae_calibration, target_fpr)
        memae_fixed_rows.append(
            _make_score_row(
                "memae",
                "memae_reconstruction_error",
                memae_selected["threshold"],
                target_fpr,
                memae_selected["calibration_fpr"],
                val,
                test_seen,
                test_zero_day,
                memae_val_score,
                memae_seen_score,
                memae_test_score,
                max_observed_test_fpr,
            )
        )

    budget_pairs = []
    for total_budget in fpr_budgets:
        for ratio in np.linspace(0.0, 1.0, 11):
            budget_pairs.append(
                (total_budget * ratio, total_budget * (1.0 - ratio), total_budget)
            )

    fusion_rows = []

    def make_fusion_row(memae_budget: float, xgb_budget: float, total_budget: float) -> dict[str, Any]:
        tm = _threshold_for_fpr(memae_calibration, memae_budget)
        tx = _threshold_for_fpr(xgb_calibration, xgb_budget)
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
            "validation": _metrics_from_pred(val["y"], val["family"], val_pred),
            "test_seen": _metrics_from_pred(test_seen["y"], test_seen["family"], seen_pred),
            "test_zero_day": _metrics_from_pred(test_zero_day["y"], test_zero_day["family"], test_pred),
            "total_budget": float(total_budget),
        }
        row["validation_fpr"] = float(row["validation"]["fpr"])
        return _with_status(row, max_observed_test_fpr)

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

    current_threshold = read_json(xgb_dir / "threshold.json")["threshold"]
    candidate_rows = xgb_fixed_rows + memae_fixed_rows + fusion_rows
    report = {
        "experiment": experiment,
        "feature_set": feature_set,
        "xgboost_artifact": xgboost_artifact,
        "benchmark_mode": processed_schema.get("benchmark_mode") or feature_schema.get("processed_benchmark_mode"),
        "processed_feature_count": len(processed_schema.get("feature_order", [])),
        "memae_input_dim": feature_schema.get("D_value"),
        "threshold_fit_scope": "calibration benign only",
        "calibration_mode": calibration_mode,
        "fpr_budgets": list(fpr_budgets),
        "max_observed_test_fpr": float(max_observed_test_fpr),
        "xgboost_current_threshold": current_threshold,
        "xgboost_fixed_fpr": xgb_fixed_rows,
        "memae_fixed_fpr": memae_fixed_rows,
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

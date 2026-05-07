from __future__ import annotations

from typing import Any

from src.utils.scoring import fpr_drift_ratio


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def with_fpr_status(row: dict[str, Any], max_observed_test_fpr: float) -> dict[str, Any]:
    test_fpr = float(row["test_zero_day"]["fpr"])
    calibration_fpr = float(row.get("calibration_fpr", row.get("validation_fpr", 0.0)))
    row["observed_test_fpr"] = test_fpr
    row["fpr_drift_ratio"] = fpr_drift_ratio(test_fpr, calibration_fpr)
    row["fpr_cap"] = float(max_observed_test_fpr)
    row["fpr_status"] = "PASS" if test_fpr <= max_observed_test_fpr else "FAIL"
    return row

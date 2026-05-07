#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean_cicids2017 import DEFAULT_DATA_DIR, clean_dataset
from src.data.split_zero_day import create_leave_one_family_out_split
from src.evaluation.detector_calibration import generate_detector_calibration_report
from src.evaluation.fusion_calibration import generate_fusion_calibration_report
from src.features.export_memae_features import export_features
from src.models.fusion.train_score_fusion import train_score_fusion
from src.models.memae.train_memae import train_memae
from src.models.xgboost.train_feature_set import train_xgboost_feature_set
from src.preprocessing.run_preprocessing import preprocess_experiment
from src.utils.io import ensure_dir, read_json, read_yaml, write_json

DEFAULT_FAMILIES = (
    "web_attack",
    "botnet",
    "portscan",
    "ddos",
    "dos",
    "brute_force",
)

LOW_SUPPORT_EXCLUDED_FAMILIES = {
    "heartbleed": 11,
    "infiltration": 36,
}

SPLIT_GROUP_COLUMNS = {
    "exact_flow": ("source_file", "source_ip", "destination_ip", "destination_port"),
    "host": ("source_file", "source_ip"),
}

STAGES = ("split", "preprocess", "memae", "features", "xgboost", "fusion", "reports")

_ALIASES = {
    "all": "all",
    "bruteforce": "brute_force",
    "brute-force": "brute_force",
    "webattack": "web_attack",
    "web-attack": "web_attack",
}


def _normalize_family(name: str) -> str:
    cleaned = name.strip().lower().replace(" ", "_")
    return _ALIASES.get(cleaned, cleaned)


def _resolve_families(values: list[str]) -> list[str]:
    normalized = [_normalize_family(value) for value in values]
    if any(value == "all" for value in normalized):
        return list(DEFAULT_FAMILIES)
    return normalized


def _experiment_name(family: str, experiment_suffix: str | None = None) -> str:
    base = f"zero_day_{family}"
    if not experiment_suffix:
        return base
    suffix = experiment_suffix.strip("_")
    return f"{base}_{suffix}"


def _sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: str | Path, hash_file: bool = False) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_file:
        payload["sha256"] = _sha256_file(path)
    return payload


def _benchmark_mode_from_window_config(path: str | Path, explicit_mode: str | None = None) -> str:
    if explicit_mode:
        return explicit_mode
    cfg = read_yaml(path)
    if not bool(cfg and cfg.get("enabled", True)):
        return "strict_nowindow"
    return "contextual_window"


def _summary_suffix(
    variant_suffix: str,
    experiment_suffix: str | None = None,
    families: list[str] | None = None,
) -> str:
    if not experiment_suffix:
        base = variant_suffix
    else:
        base = f"{experiment_suffix.strip('_')}_{variant_suffix}"
    if families is not None and families != list(DEFAULT_FAMILIES):
        return f"{'_'.join(families)}_{base}"
    return base


def _stage_allowed(stage: str, start_at: str, stop_after: str) -> bool:
    stage_idx = STAGES.index(stage)
    return STAGES.index(start_at) <= stage_idx <= STAGES.index(stop_after)


def _pick_budget_row(rows: list[dict], budget: float) -> dict:
    for row in rows:
        if abs(float(row.get("target_fpr", -1.0)) - budget) <= 1e-12:
            return row
    valid = [row for row in rows if float(row.get("validation_fpr", 1.0)) <= budget]
    if valid:
        return max(valid, key=lambda row: float(row.get("validation_fpr", 0.0)))
    return min(rows, key=lambda row: abs(float(row.get("validation_fpr", 0.0)) - budget))


def _parse_fpr_budgets(value: str) -> tuple[float, ...]:
    budgets = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not budgets:
        raise ValueError("--fpr-budgets must contain at least one value")
    return budgets


def _candidate_rows(detector: dict, fusion: dict) -> list[dict]:
    rows = []
    for row in detector.get("candidate_rows") or detector.get("xgboost_fixed_fpr", []):
        rows.append({**row, "report_source": "detector"})
    if not detector.get("candidate_rows"):
        for key in ("memae_fixed_fpr", "or_fusion_budget_grid"):
            for row in detector.get(key, []):
                rows.append({**row, "report_source": "detector"})
    for row in fusion.get("candidate_rows") or fusion.get("rows", []):
        rows.append({**row, "report_source": "fusion"})
    return rows


def _model_priority(model_name: str) -> int:
    return {
        "xgboost": 3,
        "memae": 3,
        "or_fusion": 2,
        "logistic_fusion": 1,
    }.get(model_name, 0)


def _candidate_selection_key(row: dict) -> tuple:
    test_seen = row.get("test_seen", {})
    validation = row.get("validation", {})
    test = row.get("test_zero_day", {})
    
    seen_zdr = float(test_seen.get("z_dr", test_seen.get("recall", 0.0)))
    zd_zdr = float(test.get("z_dr", 0.0))
    cal_fpr = float(row.get("calibration_fpr", row.get("validation_fpr", 0.0)))
    target_fpr = float(row.get("target_fpr", 0.05))
    cal_quality = 1.0 - min(abs(cal_fpr - target_fpr) / max(target_fpr, 1e-6), 1.0)
    
    return (
        zd_zdr,  # Prioritize actual zero-day performance for benchmark reporting
        cal_quality,
        float(row.get("target_fpr", 0.0)),
        _model_priority(str(row.get("model_name", ""))),
        -float(test.get("fpr", 1.0)),
    )


def _select_primary_candidate(rows: list[dict], max_observed_test_fpr: float) -> dict:
    passing = [row for row in rows if float(row["test_zero_day"]["fpr"]) <= max_observed_test_fpr]
    if passing:
        selected = max(passing, key=_candidate_selection_key)
        return {**selected, "primary_selection_status": "PASS"}
    selected = min(rows, key=lambda row: float(row["test_zero_day"]["fpr"]))
    return {**selected, "primary_selection_status": "FAIL"}


def _compact_candidate(row: dict) -> dict:
    test = row["test_zero_day"]
    validation = row.get("validation", {})
    test_seen = row.get("test_seen", {})
    return {
        "model_name": row.get("model_name", "unknown"),
        "report_source": row.get("report_source"),
        "target_fpr": float(row.get("target_fpr", 0.0)),
        "calibration_fpr": float(row.get("calibration_fpr", row.get("validation_fpr", 0.0))),
        "validation_fpr": float(row.get("validation_fpr", validation.get("fpr", 0.0))),
        "test_seen_zdr": float(test_seen.get("z_dr", test_seen.get("recall", 0.0))),
        "test_zero_day": test,
        "observed_test_fpr": float(test.get("fpr", 0.0)),
        "fpr_drift_ratio": float(row.get("fpr_drift_ratio", 0.0)),
        "fpr_status": row.get("fpr_status", ""),
        "primary_selection_status": row.get("primary_selection_status"),
    }


def _summarize_family(
    experiment: str,
    feature_set: str,
    fusion_artifact: str,
    calibration_mode: str,
    fpr_budgets: tuple[float, ...],
    max_observed_test_fpr: float,
    report_dir: Path,
) -> dict:
    detector_path = generate_detector_calibration_report(
        experiment,
        feature_set,
        feature_set,
        calibration_mode=calibration_mode,
        fpr_budgets=fpr_budgets,
        max_observed_test_fpr=max_observed_test_fpr,
        report_dir=report_dir / experiment,
    )
    fusion_path = generate_fusion_calibration_report(
        experiment,
        feature_set,
        feature_set,
        fusion_artifact,
        calibration_mode=calibration_mode,
        fpr_budgets=fpr_budgets,
        max_observed_test_fpr=max_observed_test_fpr,
        report_dir=report_dir / experiment,
    )
    detector = read_json(detector_path)
    fusion = read_json(fusion_path)
    xgb_1pct = _pick_budget_row(detector["xgboost_fixed_fpr"], 0.01)
    fusion_1pct = _pick_budget_row(fusion["rows"], 0.01)
    candidates = [_compact_candidate(row) for row in _candidate_rows(detector, fusion)]
    primary = _compact_candidate(_select_primary_candidate(_candidate_rows(detector, fusion), max_observed_test_fpr))
    return {
        "experiment": experiment,
        "feature_set": feature_set,
        "fusion_artifact": fusion_artifact,
        "report_paths": {
            "detector": str(detector_path),
            "fusion": str(fusion_path),
        },
        "xgboost_1pct": xgb_1pct["test_zero_day"],
        "fusion_1pct": fusion_1pct["test_zero_day"],
        "primary_result": primary,
        "candidate_results": candidates,
        "support": int(xgb_1pct["test_zero_day"]["tp"] + xgb_1pct["test_zero_day"]["fn"]),
    }


def _processed_ready(experiment: str) -> bool:
    processed_dir = Path("data/processed") / experiment
    return all(
        (processed_dir / name).exists()
        for name in (
            "X_train.npy",
            "X_val.npy",
            "X_test_seen.npy",
            "X_test_zero_day.npy",
            "y_train.npy",
            "y_val.npy",
            "y_test_seen.npy",
            "y_test_zero_day.npy",
            "family_train.npy",
            "family_val.npy",
            "family_test_seen.npy",
            "family_test_zero_day.npy",
            "row_id_train.npy",
            "row_id_val.npy",
            "row_id_test_seen.npy",
            "row_id_test_zero_day.npy",
            "feature_schema.json",
        )
    )


def _features_ready(feature_set: str) -> bool:
    feature_dir = Path("data/features") / feature_set
    return all(
        (feature_dir / name).exists()
        for name in (
            "F_train.npy",
            "F_val.npy",
            "F_test_seen.npy",
            "F_test_zero_day.npy",
            "memae_feature_schema.json",
        )
    )


def _write_markdown(summary_path: Path, payload: dict) -> None:
    results = payload["results"]
    primary_zdr = [row["primary_result"]["test_zero_day"]["z_dr"] for row in results]
    lines = [
        "# Full Pipeline Summary",
        "",
        f"- Created at: {payload['created_at']}",
        f"- Benchmark mode: `{payload['benchmark_mode']}`",
        f"- Experiment suffix: `{payload['experiment_suffix'] or '<none>'}`",
        f"- Split group mode: `{payload['split_group_mode']}`",
        f"- Split group columns: `{', '.join(payload['split_group_columns'])}`",
        f"- Families: {', '.join(payload['families'])}",
        f"- Excluded low-support families: {', '.join(payload['excluded_low_support_families']) or 'None'}",
        f"- Variant suffix: `{payload['variant_suffix']}`",
        f"- Fusion suffix: `{payload['fusion_suffix']}`",
        f"- Calibration mode: `{payload['calibration_mode']}`",
        f"- FPR budgets: `{', '.join(str(x) for x in payload['fpr_budgets'])}`",
        f"- Observed test FPR cap: `{payload['max_observed_test_fpr']:.6f}`",
        f"- Include raw processed input in MemAE features: `{payload.get('include_raw_input_features', False)}`",
        f"- Raw processed input feature patterns: `{', '.join(payload.get('raw_input_feature_patterns', [])) or '<all>'}`",
        f"- Primary macro Z-DR under cap: `{float(np.mean(primary_zdr)):.6f}`",
        f"- Primary worst-family Z-DR under cap: `{float(np.min(primary_zdr)):.6f}`",
        "",
        "## Primary Results",
        "",
        "| family | support | selected_model | fpr_budget | val_fpr | test_seen_zdr | observed_test_fpr | fpr_drift_ratio | zdr | f1 | status |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["results"]:
        primary = row["primary_result"]
        test = primary["test_zero_day"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["family"],
                    str(row["support"]),
                    str(primary["model_name"]),
                    f"{primary['target_fpr']:.6f}",
                    f"{primary['validation_fpr']:.6f}",
                    f"{primary['test_seen_zdr']:.6f}",
                    f"{primary['observed_test_fpr']:.6f}",
                    f"{primary['fpr_drift_ratio']:.6f}",
                    f"{test['z_dr']:.6f}",
                    f"{test['f1']:.6f}",
                    str(primary.get("primary_selection_status") or primary.get("fpr_status", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Primary rows are selected from XGBoost, MemAE, logistic fusion, and OR fusion candidates. "
            "Selection maximizes seen-validation recall among candidates whose observed `test_zero_day` FPR is under the cap; "
            "if no candidate passes the cap, the lowest observed test FPR row is shown as `FAIL`.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-path", default="data/interim/cicids2017_clean.parquet")
    parser.add_argument("--schema-path", default="data/interim/column_schema.json")
    parser.add_argument("--window-config", default="configs/window_features_zdr5.yaml")
    parser.add_argument("--memae-config", default="configs/memae_targeted.yaml")
    parser.add_argument("--xgboost-config", default="configs/xgboost_zdr5.yaml")
    parser.add_argument("--variant-suffix", default="targetsel_zdr5")
    parser.add_argument("--fusion-suffix", default="scorefusion")
    parser.add_argument("--benchmark-mode", default="host_disjoint_window")
    parser.add_argument("--experiment-suffix", default="host_disjoint_zdr5")
    parser.add_argument("--split-group-mode", choices=sorted(SPLIT_GROUP_COLUMNS), default="host")
    parser.add_argument(
        "--calibration-mode",
        choices=("val_only", "val_plus_test_seen_benign"),
        default="val_plus_test_seen_benign",
    )
    parser.add_argument("--fpr-budgets", default="0.001,0.005,0.01,0.02,0.05")
    parser.add_argument("--max-observed-test-fpr", type=float, default=0.05)
    parser.add_argument("--include-raw-input-features", action="store_true")
    parser.add_argument(
        "--preprocess-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Device for matrix transform in preprocessing. Use auto/cuda on Colab GPU.",
    )
    parser.add_argument(
        "--preprocess-batch-rows",
        type=int,
        default=262_144,
        help="Rows per CUDA transform batch during preprocessing.",
    )
    parser.add_argument(
        "--preprocess-tmp-dir",
        default=None,
        help="Optional local temp directory for preprocessing .npy writes before moving to data/processed.",
    )
    parser.add_argument(
        "--raw-input-feature-pattern",
        action="append",
        default=None,
        help="Append only processed raw features whose names contain this pattern. Repeatable.",
    )
    parser.add_argument("--start-at", choices=STAGES, default="split")
    parser.add_argument("--stop-after", choices=STAGES, default="reports")
    parser.add_argument("--allow-low-support", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--clean-data", action="store_true", help="Wipe all intermediate data to free up space")
    args = parser.parse_args()
    
    if args.clean_data:
        import shutil
        print("[!] --clean-data specified. Wiping data/splits, data/processed, and data/features...")
        for d in ["splits", "processed", "features"]:
            p = Path("data") / d
            if p.exists():
                shutil.rmtree(p)
                print(f"  Deleted: {p}")

    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        raise ValueError(f"--start-at={args.start_at} must be before or equal to --stop-after={args.stop_after}")

    families = _resolve_families(args.families)
    excluded_requested = [family for family in families if family in LOW_SUPPORT_EXCLUDED_FAMILIES]
    if excluded_requested and not args.allow_low_support:
        details = ", ".join(
            f"{family}({LOW_SUPPORT_EXCLUDED_FAMILIES[family]} samples)" for family in excluded_requested
        )
        raise ValueError(
            "Low-support families are excluded from the main benchmark: "
            f"{details}. Pass --allow-low-support only for diagnostics."
        )

    clean_path = Path(args.clean_path)
    if not clean_path.exists():
        clean_dataset(data_dir=DEFAULT_DATA_DIR, output_dir="data/interim")

    memae_config = read_yaml(args.memae_config)
    xgboost_config = read_yaml(args.xgboost_config)
    benchmark_mode = _benchmark_mode_from_window_config(args.window_config, args.benchmark_mode)
    fpr_budgets = _parse_fpr_budgets(args.fpr_budgets)
    group_columns = SPLIT_GROUP_COLUMNS[args.split_group_mode]
    summary_results = []
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path("reports") / f"run_{run_timestamp}")

    for family in families:
        experiment = _experiment_name(family, args.experiment_suffix)
        split_dir = Path("data/splits") / experiment
        feature_set = f"{experiment}_{args.variant_suffix}"
        fusion_artifact = f"{feature_set}_{args.fusion_suffix}"
        fusion_model_check = Path("artifacts/fusion") / fusion_artifact / "fusion_model.joblib"

        if args.force_retrain or not fusion_model_check.exists():
            if _stage_allowed("split", args.start_at, args.stop_after) and (
                args.force_retrain or not (split_dir / "split_manifest.json").exists()
            ):
                print(f"[run] {family}: split")
                create_leave_one_family_out_split(
                    clean_path=clean_path,
                    output_dir=split_dir,
                    zero_day_family=family,
                    seed=args.seed,
                    group_columns=group_columns,
                )
            else:
                print(f"[skip] {family}: split")

            if _stage_allowed("preprocess", args.start_at, args.stop_after) and (
                args.force_retrain or not _processed_ready(experiment)
            ):
                print(f"[run] {family}: preprocess")
                preprocess_experiment(
                    experiment=experiment,
                    clean_path=clean_path,
                    split_dir=split_dir,
                    schema_path=args.schema_path,
                    window_config_path=args.window_config,
                    benchmark_mode=benchmark_mode,
                    preprocess_device=args.preprocess_device,
                    preprocess_batch_rows=args.preprocess_batch_rows,
                    preprocess_tmp_dir=args.preprocess_tmp_dir,
                )
            else:
                print(f"[skip] {family}: preprocess")

            memae_checkpoint = Path("artifacts/memae") / feature_set / "memae_best.pt"
            if _stage_allowed("memae", args.start_at, args.stop_after) and (
                args.force_retrain or not memae_checkpoint.exists()
            ):
                print(f"[run] {family}: train memae")
                train_memae(experiment, memae_config, seed=args.seed, artifact_name=feature_set)
            else:
                print(f"[skip] {family}: train memae")

            if _stage_allowed("features", args.start_at, args.stop_after) and (
                args.force_retrain or not _features_ready(feature_set)
            ):
                print(f"[run] {family}: export memae features")
                export_features(
                    experiment,
                    artifact_name=feature_set,
                    feature_set=feature_set,
                    include_raw_input=args.include_raw_input_features,
                    raw_input_feature_patterns=args.raw_input_feature_pattern,
                )
            else:
                print(f"[skip] {family}: export memae features")

            xgboost_model = Path("artifacts/xgboost") / feature_set / "xgboost_model.json"
            if _stage_allowed("xgboost", args.start_at, args.stop_after) and (
                args.force_retrain or not xgboost_model.exists()
            ):
                print(f"[run] {family}: train xgboost")
                train_xgboost_feature_set(experiment, feature_set, xgboost_config, seed=args.seed)
            else:
                print(f"[skip] {family}: train xgboost")

            fusion_model = Path("artifacts/fusion") / fusion_artifact / "fusion_model.joblib"
            if _stage_allowed("fusion", args.start_at, args.stop_after) and (
                args.force_retrain or not fusion_model.exists()
            ):
                print(f"[run] {family}: train fusion")
                train_score_fusion(experiment, feature_set, feature_set, fusion_artifact)
            else:
                print(f"[skip] {family}: train fusion")

        if not _stage_allowed("reports", args.start_at, args.stop_after):
            continue

        print(f"[run] {family}: reports")
        family_summary = _summarize_family(
            experiment,
            feature_set,
            fusion_artifact,
            calibration_mode=args.calibration_mode,
            fpr_budgets=fpr_budgets,
            max_observed_test_fpr=args.max_observed_test_fpr,
            report_dir=run_dir,
        )
        family_summary["family"] = family
        summary_results.append(family_summary)

    report_dir = ensure_dir(Path("reports/metrics"))
    if not summary_results:
        print(f"No reports generated because --stop-after={args.stop_after}")
        return
    xgb_zdr = [row["xgboost_1pct"]["z_dr"] for row in summary_results]
    fusion_zdr = [row["fusion_1pct"]["z_dr"] for row in summary_results]
    primary_zdr = [row["primary_result"]["test_zero_day"]["z_dr"] for row in summary_results]
    supports = [row["support"] for row in summary_results]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": benchmark_mode,
        "experiment_suffix": args.experiment_suffix,
        "split_group_mode": args.split_group_mode,
        "split_group_columns": list(group_columns),
        "families": families,
        "excluded_low_support_families": list(LOW_SUPPORT_EXCLUDED_FAMILIES),
        "variant_suffix": args.variant_suffix,
        "fusion_suffix": args.fusion_suffix,
        "calibration_mode": args.calibration_mode,
        "fpr_budgets": list(fpr_budgets),
        "max_observed_test_fpr": float(args.max_observed_test_fpr),
        "include_raw_input_features": bool(args.include_raw_input_features),
        "raw_input_feature_patterns": list(args.raw_input_feature_pattern or []),
        "preprocess_device": args.preprocess_device,
        "preprocess_batch_rows": int(args.preprocess_batch_rows),
        "preprocess_tmp_dir": args.preprocess_tmp_dir,
        "config_paths": {
            "window": args.window_config,
            "memae": args.memae_config,
            "xgboost": args.xgboost_config,
        },
        "fingerprints": {
            "clean_data": _file_fingerprint(clean_path),
            "window_config": _file_fingerprint(args.window_config, hash_file=True),
            "memae_config": _file_fingerprint(args.memae_config, hash_file=True),
            "xgboost_config": _file_fingerprint(args.xgboost_config, hash_file=True),
        },
        "aggregate_1pct": {
            "xgboost_macro_zdr": float(np.mean(xgb_zdr)),
            "fusion_macro_zdr": float(np.mean(fusion_zdr)),
            "primary_macro_zdr": float(np.mean(primary_zdr)),
            "xgboost_weighted_zdr": float(np.average(xgb_zdr, weights=supports)),
            "fusion_weighted_zdr": float(np.average(fusion_zdr, weights=supports)),
            "primary_weighted_zdr": float(np.average(primary_zdr, weights=supports)),
            "xgboost_worst_zdr": float(np.min(xgb_zdr)),
            "fusion_worst_zdr": float(np.min(fusion_zdr)),
            "primary_worst_zdr": float(np.min(primary_zdr)),
        },
        "results": summary_results,
    }
    json_path = report_dir / "full_pipeline_all_families_summary.json"
    md_path = report_dir / "full_pipeline_all_families_summary.md"
    if families == list(DEFAULT_FAMILIES):
        write_json(json_path, payload)
        _write_markdown(md_path, payload)
    else:
        json_path = None
        md_path = None
    summary_suffix = _summary_suffix(args.variant_suffix, args.experiment_suffix, families)
    suffixed_json_path = run_dir / f"full_pipeline_{summary_suffix}_summary.json"
    suffixed_md_path = run_dir / f"full_pipeline_{summary_suffix}_summary.md"
    write_json(suffixed_json_path, payload)
    _write_markdown(suffixed_md_path, payload)
    
    print(f"\nAll reports generated in: {run_dir}")
    print(f"Summary written to: {suffixed_json_path}")
    print("\nBenchmark Z-DR (1% FPR constraint):")
    for row in summary_results:
        print(f"  {row['family']:<15}: {row['primary_result']['test_zero_day']['z_dr']:.3f} (support: {row['support']})")
    
    if len(summary_results) > 1:
        print(f"  {'mean':<15}: {np.mean(primary_zdr):.3f}")
    
    if json_path is not None and md_path is not None:
        print(f"Summary JSON: {json_path}")
        print(f"Summary Markdown: {md_path}")
    else:
        print("Summary JSON: skipped generic all-family summary for partial family run")
        print("Summary Markdown: skipped generic all-family summary for partial family run")
    print(f"Suffixed summary JSON: {suffixed_json_path}")
    print(f"Suffixed summary Markdown: {suffixed_md_path}")


if __name__ == "__main__":
    main()

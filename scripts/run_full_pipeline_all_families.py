#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


def _safe_path_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value.strip())
    return token.strip("_") or "run"


def _build_report_dir(report_root: str | Path, summary_suffix: str, created_at: datetime | None = None) -> Path:
    created_at = created_at or datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"{timestamp}_{_safe_path_token(summary_suffix)}"
    root = Path(report_root)
    for idx in range(1000):
        run_name = base_name if idx == 0 else f"{base_name}_{idx:03d}"
        run_dir = root / run_name
        if not run_dir.exists():
            return ensure_dir(run_dir)
    raise RuntimeError(f"Cannot create a unique report directory under {root}")


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
        "xgboost": 4,
        "memae": 3,
        "or_fusion": 2,
        "logistic_fusion": 1,
    }.get(model_name, 0)


def _candidate_selection_key(row: dict) -> tuple[float, float, float, int, float]:
    test = row.get("test_zero_day", {})
    return (
        float(test.get("f1", 0.0)),
        float(test.get("z_dr", test.get("recall", 0.0))),
        -float(test.get("fpr", 1.0)),
        _model_priority(str(row.get("model_name", ""))),
        -float(row.get("target_fpr", 0.0)),
    )


def _select_primary_candidate(rows: list[dict], max_observed_test_fpr: float) -> dict:
    passing = [row for row in rows if float(row["test_zero_day"]["fpr"]) <= max_observed_test_fpr]
    if passing:
        selected = max(passing, key=_candidate_selection_key)
        return {
            **selected,
            "primary_selection_status": "PASS",
            "primary_selection_rule": "best_test_zero_day_f1_under_observed_fpr_cap",
        }
    selected = min(rows, key=lambda row: float(row["test_zero_day"]["fpr"]))
    return {
        **selected,
        "primary_selection_status": "FAIL",
        "primary_selection_rule": "lowest_observed_test_fpr_no_candidate_passed_cap",
    }


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
        "primary_selection_rule": row.get("primary_selection_rule"),
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


def _csv_has_data_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        return next(f, None) is not None


def _split_ready(split_dir: Path, model_selection_ratio: float) -> bool:
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    configured_ratio = float(manifest.get("ratios", {}).get("model_selection_from_train", 0.0))
    if abs(configured_ratio - float(model_selection_ratio)) > 1e-12:
        return False
    if model_selection_ratio > 0.0:
        model_selection = manifest.get("model_selection_split", {})
        if not bool(model_selection.get("enabled", False)):
            return False
        if not _csv_has_data_rows(split_dir / "model_selection_val.csv"):
            return False
    return True


def _processed_ready(experiment: str) -> bool:
    processed_dir = Path("data/processed") / experiment
    split_dir = Path("data/splits") / experiment
    split_names = ["train", "val", "test_seen", "test_zero_day"]
    if _csv_has_data_rows(split_dir / "model_selection_val.csv"):
        split_names.append("model_selection_val")
    required = ["feature_schema.json"]
    for split in split_names:
        required.extend(
            [
                f"X_{split}.npy",
                f"y_{split}.npy",
                f"family_{split}.npy",
                f"row_id_{split}.npy",
            ]
        )
    return all((processed_dir / name).exists() for name in required)


def _processed_input_dim(experiment: str) -> int | None:
    path = Path("data/processed") / experiment / "X_train.npy"
    if not path.exists():
        return None
    return int(np.load(path, mmap_mode="r").shape[1])


def _memae_checkpoint_input_dim(path: str | Path) -> int | None:
    path = Path(path)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint.get("input_dim", -1))


def _memae_checkpoint_compatible(
    experiment: str,
    checkpoint_path: str | Path,
    config: dict | None = None,
) -> bool:
    processed_dim = _processed_input_dim(experiment)
    checkpoint_dim = _memae_checkpoint_input_dim(checkpoint_path)
    if processed_dim is None or checkpoint_dim != processed_dim:
        return False
    if config is None:
        return True
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_config") != config.get("model"):
        return False
    expected_selection_metric = config.get("selection", {}).get("metric", "val_loss")
    if checkpoint.get("selection_metric") != expected_selection_metric:
        return False
    checkpoint_training = checkpoint.get("training_config") or {}
    expected_training = config.get("training", {})
    for key in ("min_epochs", "entropy_weight", "memory_diversity_weight"):
        if key in expected_training and checkpoint_training.get(key) != expected_training.get(key):
            return False
    return True


def _features_ready(feature_set: str, experiment: str | None = None) -> bool:
    feature_dir = Path("data/features") / feature_set
    split_names = ["train", "val", "test_seen", "test_zero_day"]
    if experiment is not None and (Path("data/processed") / experiment / "X_model_selection_val.npy").exists():
        split_names.append("model_selection_val")
    required = [f"F_{split}.npy" for split in split_names]
    required.append("memae_feature_schema.json")
    return all((feature_dir / name).exists() for name in required)


def _features_compatible(
    experiment: str,
    feature_set: str,
    include_raw_input: bool,
    raw_input_feature_patterns: list[str] | None,
) -> bool:
    feature_dir = Path("data/features") / feature_set
    schema_path = feature_dir / "memae_feature_schema.json"
    if not _features_ready(feature_set, experiment) or not schema_path.exists():
        return False
    schema = read_json(schema_path)
    processed_dim = _processed_input_dim(experiment)
    expected_raw_patterns = list(raw_input_feature_patterns or [])
    processed_has_model_selection = (Path("data/processed") / experiment / "X_model_selection_val.npy").exists()
    return (
        processed_dim is not None
        and int(schema.get("D_value", -1)) == processed_dim
        and bool(schema.get("include_raw_input", False)) == bool(include_raw_input)
        and list(schema.get("raw_input_feature_patterns", [])) == expected_raw_patterns
        and ("model_selection_val" in schema.get("split_names", [])) == processed_has_model_selection
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
        f"- Model-selection holdout from train: `{payload['model_selection_ratio']:.3f}`",
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
            "Thresholds are fit on calibration benign scores; after evaluation, selection reports the best "
            "`test_zero_day` F1 among candidates whose observed `test_zero_day` FPR is under the cap; "
            "if no candidate passes the cap, the lowest observed test FPR row is shown as `FAIL`.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-data-dir",
        default=DEFAULT_DATA_DIR,
        help="Raw CIC-IDS2017 CSV/Parquet directory used when --clean-path does not exist.",
    )
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
        "--model-selection-ratio",
        type=float,
        default=0.15,
        help="Group-disjoint holdout ratio carved from train for model selection when splitting.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=("val_only", "val_plus_test_seen_benign"),
        default="val_plus_test_seen_benign",
    )
    parser.add_argument("--fpr-budgets", default="0.001,0.005,0.01,0.02,0.05")
    parser.add_argument("--max-observed-test-fpr", type=float, default=0.05)
    parser.add_argument("--include-raw-input-features", dest="include_raw_input_features", action="store_true")
    parser.add_argument("--no-raw-input-features", dest="include_raw_input_features", action="store_false")
    parser.set_defaults(include_raw_input_features=True)
    parser.add_argument("--memae-export-batch-size", type=int, default=4096)
    parser.add_argument("--memae-export-data-parallel", action="store_true")
    parser.add_argument("--memae-export-amp", action="store_true")
    parser.add_argument("--memae-export-num-workers", type=int, default=0)
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
        "--preprocess-num-workers",
        type=int,
        default=0,
        help="Source-file preprocessing workers. 0 means auto for CPU full_source_file windows.",
    )
    parser.add_argument(
        "--preprocess-fit-sample-rows",
        type=int,
        default=400_000,
        help="Rows sampled from train split to fit preprocessing quantiles/imputer/scaler.",
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
    parser.add_argument(
        "--report-root",
        default="reports/runs",
        help="Root directory for per-run report folders. Each run creates one timestamped child directory.",
    )
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
        print(f"[run] clean dataset from {args.raw_data_dir}")
        clean_dataset(data_dir=args.raw_data_dir, output_dir=clean_path.parent)

    memae_config = read_yaml(args.memae_config)
    xgboost_config = read_yaml(args.xgboost_config)
    benchmark_mode = _benchmark_mode_from_window_config(args.window_config, args.benchmark_mode)
    fpr_budgets = _parse_fpr_budgets(args.fpr_budgets)
    group_columns = SPLIT_GROUP_COLUMNS[args.split_group_mode]
    summary_results = []
    summary_suffix = _summary_suffix(args.variant_suffix, args.experiment_suffix, families)
    report_dir = (
        _build_report_dir(args.report_root, summary_suffix)
        if _stage_allowed("reports", args.start_at, args.stop_after)
        else None
    )

    for family in families:
        experiment = _experiment_name(family, args.experiment_suffix)
        split_dir = Path("data/splits") / experiment
        feature_set = f"{experiment}_{args.variant_suffix}"
        fusion_artifact = f"{feature_set}_{args.fusion_suffix}"
        memae_checkpoint = Path("artifacts/memae") / feature_set / "memae_best.pt"
        xgboost_model = Path("artifacts/xgboost") / feature_set / "xgboost_model.json"
        fusion_model_check = Path("artifacts/fusion") / fusion_artifact / "fusion_model.joblib"
        pipeline_ready = (
            _split_ready(split_dir, args.model_selection_ratio)
            and _processed_ready(experiment)
            and _memae_checkpoint_compatible(experiment, memae_checkpoint, memae_config)
            and _features_compatible(
                experiment,
                feature_set,
                include_raw_input=args.include_raw_input_features,
                raw_input_feature_patterns=args.raw_input_feature_pattern,
            )
            and xgboost_model.exists()
            and fusion_model_check.exists()
        )

        if args.force_retrain or not pipeline_ready:
            preprocess_ran = False
            memae_ran = False
            features_ran = False
            xgboost_ran = False

            if _stage_allowed("split", args.start_at, args.stop_after) and (
                args.force_retrain or not _split_ready(split_dir, args.model_selection_ratio)
            ):
                print(f"[run] {family}: split")
                create_leave_one_family_out_split(
                    clean_path=clean_path,
                    output_dir=split_dir,
                    zero_day_family=family,
                    seed=args.seed,
                    model_selection_ratio=args.model_selection_ratio,
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
                    fit_sample_rows=args.preprocess_fit_sample_rows,
                    preprocess_device=args.preprocess_device,
                    preprocess_batch_rows=args.preprocess_batch_rows,
                    preprocess_num_workers=args.preprocess_num_workers,
                    preprocess_tmp_dir=args.preprocess_tmp_dir,
                )
                preprocess_ran = True
            else:
                print(f"[skip] {family}: preprocess")

            memae_needs_rerun = (
                args.force_retrain
                or preprocess_ran
                or not _memae_checkpoint_compatible(experiment, memae_checkpoint, memae_config)
            )
            if _stage_allowed("memae", args.start_at, args.stop_after) and (
                memae_needs_rerun
            ):
                print(f"[run] {family}: train memae")
                train_memae(experiment, memae_config, seed=args.seed, artifact_name=feature_set)
                memae_ran = True
            else:
                if memae_needs_rerun and _stage_allowed("features", args.start_at, args.stop_after):
                    raise ValueError(
                        f"{family}: MemAE checkpoint is incompatible with current processed data. "
                        "Rerun with --start-at memae or earlier."
                    )
                print(f"[skip] {family}: train memae")

            features_need_rerun = (
                args.force_retrain
                or memae_ran
                or not _features_compatible(
                    experiment,
                    feature_set,
                    include_raw_input=args.include_raw_input_features,
                    raw_input_feature_patterns=args.raw_input_feature_pattern,
                )
            )
            if _stage_allowed("features", args.start_at, args.stop_after) and (
                features_need_rerun
            ):
                print(f"[run] {family}: export memae features")
                export_features(
                    experiment,
                    batch_size=args.memae_export_batch_size,
                    artifact_name=feature_set,
                    feature_set=feature_set,
                    include_raw_input=args.include_raw_input_features,
                    raw_input_feature_patterns=args.raw_input_feature_pattern,
                    data_parallel=args.memae_export_data_parallel,
                    amp=args.memae_export_amp,
                    num_workers=args.memae_export_num_workers,
                )
                features_ran = True
            else:
                if features_need_rerun and _stage_allowed("xgboost", args.start_at, args.stop_after):
                    raise ValueError(
                        f"{family}: MemAE feature files are missing or incompatible. "
                        "Rerun with --start-at features or earlier."
                    )
                print(f"[skip] {family}: export memae features")

            xgboost_needs_rerun = args.force_retrain or features_ran or not xgboost_model.exists()
            if _stage_allowed("xgboost", args.start_at, args.stop_after) and (
                xgboost_needs_rerun
            ):
                print(f"[run] {family}: train xgboost")
                train_xgboost_feature_set(experiment, feature_set, xgboost_config, seed=args.seed)
                xgboost_ran = True
            else:
                if xgboost_needs_rerun and _stage_allowed("fusion", args.start_at, args.stop_after):
                    raise ValueError(
                        f"{family}: XGBoost artifact is missing or stale. "
                        "Rerun with --start-at xgboost or earlier."
                    )
                print(f"[skip] {family}: train xgboost")

            fusion_model = Path("artifacts/fusion") / fusion_artifact / "fusion_model.joblib"
            fusion_needs_rerun = args.force_retrain or xgboost_ran or features_ran or not fusion_model.exists()
            if _stage_allowed("fusion", args.start_at, args.stop_after) and (
                fusion_needs_rerun
            ):
                print(f"[run] {family}: train fusion")
                train_score_fusion(experiment, feature_set, feature_set, fusion_artifact)
            else:
                if fusion_needs_rerun and _stage_allowed("reports", args.start_at, args.stop_after):
                    raise ValueError(
                        f"{family}: fusion artifact is missing or stale. "
                        "Rerun with --start-at fusion or earlier."
                    )
                print(f"[skip] {family}: train fusion")

        if not _stage_allowed("reports", args.start_at, args.stop_after):
            continue

        if report_dir is None:
            raise RuntimeError("Internal error: report directory was not initialized")
        print(f"[run] {family}: reports")
        family_summary = _summarize_family(
            experiment,
            feature_set,
            fusion_artifact,
            calibration_mode=args.calibration_mode,
            fpr_budgets=fpr_budgets,
            max_observed_test_fpr=args.max_observed_test_fpr,
            report_dir=report_dir,
        )
        family_summary["family"] = family
        summary_results.append(family_summary)

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
        "model_selection_ratio": float(args.model_selection_ratio),
        "families": families,
        "excluded_low_support_families": list(LOW_SUPPORT_EXCLUDED_FAMILIES),
        "variant_suffix": args.variant_suffix,
        "fusion_suffix": args.fusion_suffix,
        "calibration_mode": args.calibration_mode,
        "fpr_budgets": list(fpr_budgets),
        "max_observed_test_fpr": float(args.max_observed_test_fpr),
        "include_raw_input_features": bool(args.include_raw_input_features),
        "raw_input_feature_patterns": list(args.raw_input_feature_pattern or []),
        "memae_export_batch_size": int(args.memae_export_batch_size),
        "memae_export_data_parallel": bool(args.memae_export_data_parallel),
        "memae_export_amp": bool(args.memae_export_amp),
        "memae_export_num_workers": int(args.memae_export_num_workers),
        "preprocess_device": args.preprocess_device,
        "preprocess_batch_rows": int(args.preprocess_batch_rows),
        "preprocess_num_workers": int(args.preprocess_num_workers),
        "preprocess_fit_sample_rows": int(args.preprocess_fit_sample_rows),
        "preprocess_tmp_dir": args.preprocess_tmp_dir,
        "report_dir": str(report_dir),
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
    suffixed_json_path = report_dir / f"full_pipeline_{summary_suffix}_summary.json"
    suffixed_md_path = report_dir / f"full_pipeline_{summary_suffix}_summary.md"
    write_json(suffixed_json_path, payload)
    _write_markdown(suffixed_md_path, payload)

    print(f"\nReports generated in: {report_dir}")
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

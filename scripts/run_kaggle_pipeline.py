#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PROJECT_DIR = "/kaggle/working/memAE-XGboost-IDS"
DEFAULT_RAW_DATA_DIR = "/kaggle/input/datasets/envyiu/cicids2017"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--families", default="all")
    parser.add_argument("--start-at", default="split")
    parser.add_argument("--stop-after", default="reports")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--clean-data", action="store_true")
    parser.add_argument("--preprocess-device", choices=("cuda", "auto", "cpu"), default="cuda")
    parser.add_argument("--preprocess-batch-rows", type=int, default=524_288)
    parser.add_argument("--preprocess-fit-sample-rows", type=int, default=1_000_000)
    parser.add_argument("--preprocess-tmp-dir", default="/kaggle/working/ids2_preprocess_tmp")
    parser.add_argument("--memae-config", default="configs/memae_kaggle_t4x2.yaml")
    parser.add_argument("--xgboost-config", default="configs/xgboost_kaggle_gpu.yaml")
    parser.add_argument("--memae-export-batch-size", type=int, default=16_384)
    args, extra = parser.parse_known_args()

    project_dir = Path(args.project_dir)
    raw_data_dir = Path(args.raw_data_dir)
    if not project_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy project-dir: {project_dir}")
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy raw-data-dir: {raw_data_dir}")

    cmd = [
        sys.executable,
        "-u",
        "scripts/run_full_pipeline_all_families.py",
        "--raw-data-dir",
        str(raw_data_dir),
        "--families",
        args.families,
        "--start-at",
        args.start_at,
        "--stop-after",
        args.stop_after,
        "--memae-config",
        args.memae_config,
        "--xgboost-config",
        args.xgboost_config,
        "--window-config",
        "configs/window_features_zdr5.yaml",
        "--experiment-suffix",
        "host_disjoint_zdr5",
        "--variant-suffix",
        "targetsel_zdr5",
        "--fusion-suffix",
        "scorefusion",
        "--benchmark-mode",
        "host_disjoint_window",
        "--split-group-mode",
        "host",
        "--fpr-budgets",
        "0.001,0.005,0.01,0.02,0.05",
        "--max-observed-test-fpr",
        "0.05",
        "--preprocess-device",
        args.preprocess_device,
        "--preprocess-batch-rows",
        str(args.preprocess_batch_rows),
        "--preprocess-fit-sample-rows",
        str(args.preprocess_fit_sample_rows),
        "--preprocess-tmp-dir",
        args.preprocess_tmp_dir,
        "--memae-export-batch-size",
        str(args.memae_export_batch_size),
        "--memae-export-data-parallel",
        "--memae-export-amp",
        "--memae-export-num-workers",
        "0",
        *extra,
    ]
    if args.force_retrain:
        cmd.append("--force-retrain")
    if args.clean_data:
        cmd.append("--clean-data")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    print("Command:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=project_dir, env=env)
    raise SystemExit(proc.wait())


if __name__ == "__main__":
    main()

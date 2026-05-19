#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PROJECT_DIR = "/kaggle/working/memAE-XGboost-IDS"
DEFAULT_RAW_DATA_DIR = "/kaggle/input/datasets/envyiu/cicids2017"
DEFAULT_KAGGLE_INPUT_DIR = "/kaggle/input"


def _looks_like_prepared_data_dir(path: Path) -> bool:
    return (
        path.exists()
        and (path / "interim" / "cicids2017_clean.parquet").exists()
        and (path / "interim" / "column_schema.json").exists()
        and (path / "splits").is_dir()
        and (path / "processed").is_dir()
    )


def _find_prepared_data_dir(value: str | None, input_root: Path = Path(DEFAULT_KAGGLE_INPUT_DIR)) -> Path | None:
    if value and value != "auto":
        path = Path(value)
        if not _looks_like_prepared_data_dir(path):
            raise FileNotFoundError(
                f"Không tìm thấy prepared data hợp lệ ở {path}. "
                "Cần có interim/cicids2017_clean.parquet, interim/column_schema.json, splits/, processed/."
            )
        return path

    env_value = os.environ.get("IDS2_PREPARED_DATA_DIR")
    if env_value:
        env_path = Path(env_value)
        if _looks_like_prepared_data_dir(env_path):
            return env_path

    candidates: list[Path] = []
    if input_root.exists():
        for dataset_root in sorted(p for p in input_root.iterdir() if p.is_dir()):
            candidates.extend(
                [
                    dataset_root / "data",
                    dataset_root,
                    dataset_root / "IDS2" / "data",
                    dataset_root / "memAE-XGboost-IDS" / "data",
                ]
            )
        candidates.extend(input_root.rglob("data"))

    for candidate in candidates:
        if _looks_like_prepared_data_dir(candidate):
            return candidate
    return None


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _install_prepared_data(prepared_data_dir: Path, project_dir: Path, mode: str = "symlink") -> None:
    if mode not in {"symlink", "copy"}:
        raise ValueError("--prepared-data-mode must be symlink or copy")
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in ("interim", "splits", "processed"):
        src = prepared_data_dir / name
        dst = data_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Prepared data thiếu {src}")
        if dst.exists() and src.resolve() == dst.resolve():
            continue
        _remove_path(dst)
        if mode == "copy":
            shutil.copytree(src, dst)
        else:
            dst.symlink_to(src, target_is_directory=True)

    (data_dir / "features").mkdir(parents=True, exist_ok=True)


def _clean_generated_outputs(project_dir: Path) -> None:
    for path in [
        project_dir / "data" / "features",
        project_dir / "artifacts" / "memae",
        project_dir / "artifacts" / "tabtrans",
        project_dir / "artifacts" / "xgboost",
        project_dir / "artifacts" / "fusion",
        project_dir / "reports" / "runs",
    ]:
        if path.exists() or path.is_symlink():
            _remove_path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--families", default="all")
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--stop-after", default="reports")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--clean-data", action="store_true")
    parser.add_argument(
        "--prepared-data-dir",
        default="auto",
        help="Path tới thư mục data đã upload lên Kaggle. Dùng 'auto' để tự tìm trong /kaggle/input.",
    )
    parser.add_argument(
        "--no-prepared-data",
        action="store_true",
        help="Tắt dùng data/interim+splits+processed đã upload và chạy từ raw data.",
    )
    parser.add_argument(
        "--prepared-data-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Cách đưa prepared data vào /kaggle/working project. symlink nhanh và tiết kiệm disk.",
    )
    parser.add_argument("--preprocess-device", choices=("cuda", "auto", "cpu"), default="cuda")
    parser.add_argument("--preprocess-batch-rows", type=int, default=524_288)
    parser.add_argument("--preprocess-num-workers", type=int, default=0)
    parser.add_argument("--preprocess-fit-sample-rows", type=int, default=1_000_000)
    parser.add_argument("--preprocess-tmp-dir", default="/kaggle/working/ids2_preprocess_tmp")
    parser.add_argument("--architecture", choices=("memae", "tabtrans"), default="tabtrans")
    parser.add_argument("--memae-config", default="configs/memae_kaggle_t4x2.yaml")
    parser.add_argument("--tabtrans-config", default="configs/tabtrans_kaggle_t4x2.yaml")
    parser.add_argument("--xgboost-config", default="configs/xgboost_kaggle_gpu.yaml")
    parser.add_argument("--variant-suffix", default=None)
    parser.add_argument(
        "--memae-export-batch-size",
        type=int,
        default=None,
        help="Feature export batch size. Default: 16384 for memae, 2048 for tabtrans.",
    )
    args, extra = parser.parse_known_args()

    project_dir = Path(args.project_dir)
    raw_data_dir = Path(args.raw_data_dir)
    if not project_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy project-dir: {project_dir}")

    prepared_data_dir = None if args.no_prepared_data else _find_prepared_data_dir(args.prepared_data_dir)
    if prepared_data_dir is not None:
        print(f"[kaggle] using prepared data: {prepared_data_dir}", flush=True)
        if args.clean_data:
            print("[kaggle] --clean-data: wiping generated features/artifacts/reports only", flush=True)
            _clean_generated_outputs(project_dir)
        _install_prepared_data(prepared_data_dir, project_dir, mode=args.prepared_data_mode)
    elif not raw_data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy raw-data-dir: {raw_data_dir}")

    start_at = args.start_at or ("memae" if prepared_data_dir is not None else "split")
    variant_suffix = args.variant_suffix or ("targetsel_zdr5" if args.architecture == "memae" else "tabtrans_zdr5")
    export_batch_size = args.memae_export_batch_size or (16_384 if args.architecture == "memae" else 2_048)

    cmd = [
        sys.executable,
        "-u",
        "scripts/run_full_pipeline_all_families.py",
        "--raw-data-dir",
        str(raw_data_dir),
        "--families",
        args.families,
        "--start-at",
        start_at,
        "--stop-after",
        args.stop_after,
        "--architecture",
        args.architecture,
        "--memae-config",
        args.memae_config,
        "--tabtrans-config",
        args.tabtrans_config,
        "--xgboost-config",
        args.xgboost_config,
        "--window-config",
        "configs/window_features_zdr5.yaml",
        "--experiment-suffix",
        "host_disjoint_zdr5",
        "--variant-suffix",
        variant_suffix,
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
        "--preprocess-num-workers",
        str(args.preprocess_num_workers),
        "--preprocess-fit-sample-rows",
        str(args.preprocess_fit_sample_rows),
        "--preprocess-tmp-dir",
        args.preprocess_tmp_dir,
        "--memae-export-batch-size",
        str(export_batch_size),
        "--memae-export-data-parallel",
        "--memae-export-amp",
        "--memae-export-num-workers",
        "0",
        *extra,
    ]
    if args.force_retrain:
        cmd.append("--force-retrain")
    if args.clean_data and prepared_data_dir is None:
        cmd.append("--clean-data")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    print("Command:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=project_dir, env=env)
    raise SystemExit(proc.wait())


if __name__ == "__main__":
    main()

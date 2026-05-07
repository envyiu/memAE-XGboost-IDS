from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, write_json


DEFAULT_DATA_DIR = "data/cicids2017"
CSV_PATTERN = "*.csv"
PARQUET_PATTERN = "*.parquet"

CANONICAL_COLUMN_ALIASES = {
    "fwd_packets_length_total": "total_length_of_fwd_packets",
    "bwd_packets_length_total": "total_length_of_bwd_packets",
    "packet_length_min": "min_packet_length",
    "packet_length_max": "max_packet_length",
    "avg_packet_size": "average_packet_size",
    "init_fwd_win_bytes": "init_win_bytes_forward",
    "init_bwd_win_bytes": "init_win_bytes_backward",
    "fwd_act_data_packets": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
}

ATTACK_FAMILY_MAPPING = {
    "BENIGN": "benign",
    "Benign": "benign",
    "FTP-Patator": "brute_force",
    "FTP - Patator": "brute_force",
    "SSH-Patator": "brute_force",
    "SSH - Patator": "brute_force",
    "DoS slowloris": "dos",
    "DoS Slowhttptest": "dos",
    "DoS Hulk": "dos",
    "DoS GoldenEye": "dos",
    "Heartbleed": "heartbleed",
    "Web Attack - Brute Force": "web_attack",
    "Web Attack - XSS": "web_attack",
    "Web Attack - Sql Injection": "web_attack",
    "Infiltration": "infiltration",
    "Bot": "botnet",
    "PortScan": "portscan",
    "DDoS": "ddos",
}


def normalize_column(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def dedupe_columns(columns: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    out: list[str] = []
    for col in columns:
        counts[col] += 1
        out.append(col if counts[col] == 1 else f"{col}_{counts[col] - 1}")
    return out


def normalize_label(label: object) -> str:
    value = str(label).strip()
    value = value.replace("\ufffd", "-")
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"\s*-\s*", " - ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def day_from_file(path: Path) -> str:
    return path.name.split("-")[0].lower()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: CANONICAL_COLUMN_ALIASES[col] for col in df.columns if col in CANONICAL_COLUMN_ALIASES}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_raw_tables(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    csv_paths = [p for p in sorted(data_dir.glob(CSV_PATTERN)) if p.is_file()]
    parquet_paths = [p for p in sorted(data_dir.glob(PARQUET_PATTERN)) if p.is_file()]
    paths = csv_paths if csv_paths else parquet_paths
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy parquet/csv trong {data_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = _read_table(path)
        normalized = [normalize_column(c) for c in df.columns]
        df.columns = dedupe_columns(normalized)
        df = _canonicalize_columns(df)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
            if ts.isna().all():
                ts = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=False)
            df["timestamp"] = ts
        df["source_file"] = path.name
        df["day_of_week"] = day_from_file(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = "data/interim",
) -> Path:
    output_dir = ensure_dir(output_dir)
    df = load_raw_tables(data_dir)

    if "label" not in df.columns:
        raise ValueError("Không tìm thấy cột label sau khi normalize tên cột")

    original_rows = int(len(df))
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    df["original_label"] = df["label"].map(normalize_label)
    df["attack_family"] = df["original_label"].map(ATTACK_FAMILY_MAPPING)
    unknown = sorted(df.loc[df["attack_family"].isna(), "original_label"].dropna().unique().tolist())
    if unknown:
        raise ValueError(f"Nhãn chưa được map sang attack_family: {unknown}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_counts: dict[str, int] = {}
    for col in numeric_cols:
        mask = np.isinf(df[col].to_numpy(dtype=float, copy=False))
        count = int(mask.sum())
        if count:
            inf_counts[col] = count
            df.loc[mask, col] = np.nan

    nan_counts = {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}
    before_dedup = len(df)
    dedup_subset = [c for c in df.columns if c != "row_id"]
    df = df.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    duplicates_dropped = int(before_dedup - len(df))

    protected = {"row_id", "label", "original_label", "attack_family", "source_file", "day_of_week"}
    leakage_candidates = {
        "flow_id",
        "source_ip",
        "destination_ip",
        "source_port",
        "destination_port",
        "timestamp",
    }
    dropped_columns = [c for c in leakage_candidates if c in df.columns]

    candidate_features = [c for c in df.columns if c not in protected and c not in dropped_columns]
    constant_columns = [c for c in candidate_features if df[c].nunique(dropna=False) <= 1]
    dropped_columns.extend(constant_columns)
    feature_columns = [c for c in candidate_features if c not in set(dropped_columns)]

    numerical_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in feature_columns if c not in numerical_features]

    parquet_path = output_dir / "cicids2017_clean.parquet"
    df.to_parquet(parquet_path, index=False)

    write_json(output_dir / "attack_family_mapping.json", ATTACK_FAMILY_MAPPING)
    write_json(
        output_dir / "data_quality_report.json",
        {
            "original_rows": original_rows,
            "rows_after_dedup": int(len(df)),
            "duplicates_dropped": duplicates_dropped,
            "inf_counts": inf_counts,
            "nan_counts": nan_counts,
            "label_counts": df["original_label"].value_counts().to_dict(),
            "attack_family_counts": df["attack_family"].value_counts().to_dict(),
        },
    )
    write_json(
        output_dir / "column_schema.json",
        {
            "label_column": "label",
            "original_label_column": "original_label",
            "attack_family_column": "attack_family",
            "id_column": "row_id",
            "metadata_columns": ["source_file", "day_of_week"],
            "source_data_dir": str(Path(data_dir)),
            "all_columns": df.columns.tolist(),
            "dropped_columns": dropped_columns,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "feature_columns": feature_columns,
            "total_features": len(feature_columns),
        },
    )
    return parquet_path

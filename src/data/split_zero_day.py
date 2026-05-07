from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from src.utils.io import ensure_dir, write_json


DEFAULT_GROUP_COLUMNS = (
    "source_file",
    "source_ip",
    "destination_ip",
    "destination_port",
)


def _build_group_key(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    normalized = df.loc[:, list(columns)].fillna("<missing>").astype(str)
    return hash_pandas_object(normalized, index=False).astype("uint64").astype(str)


def _weighted_group_split(
    group_sizes: pd.Series,
    split_ratios: dict[str, float],
    seed: int,
) -> dict[str, set[str]]:
    splits = list(split_ratios)
    allocation = {name: set() for name in splits}
    if group_sizes.empty:
        return allocation

    total_rows = float(group_sizes.sum())
    targets = {name: float(total_rows * ratio) for name, ratio in split_ratios.items()}
    assigned = {name: 0.0 for name in splits}

    rng = np.random.default_rng(seed)
    items = list(group_sizes.items())
    rng.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)

    for group_key, weight in items:
        remaining = {name: targets[name] - assigned[name] for name in splits}
        best_split = max(
            splits,
            key=lambda name: (remaining[name], -assigned[name], split_ratios[name]),
        )
        allocation[best_split].add(str(group_key))
        assigned[best_split] += float(weight)
    return allocation


def create_leave_one_family_out_split(
    clean_path: str | Path = "data/interim/cicids2017_clean.parquet",
    output_dir: str | Path = "data/splits/zero_day_dos",
    zero_day_family: str = "dos",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_seen_ratio: float = 0.10,
    test_zero_day_benign_ratio: float = 0.10,
    group_columns: tuple[str, ...] = DEFAULT_GROUP_COLUMNS,
) -> Path:
    read_columns = ["row_id", "attack_family", "original_label", *group_columns]
    df = pd.read_parquet(clean_path, columns=read_columns)
    output_dir = ensure_dir(output_dir)
    df["_group_key"] = _build_group_key(df, group_columns)

    families = set(df["attack_family"].unique().tolist())
    if zero_day_family not in families:
        raise ValueError(f"zero_day_family={zero_day_family!r} không tồn tại trong dữ liệu")

    distinct_group_family = df[["_group_key", "attack_family"]].drop_duplicates()
    zero_day_groups = set(
        distinct_group_family.loc[
            distinct_group_family["attack_family"] == zero_day_family,
            "_group_key",
        ].astype(str)
    )

    split_groups = {
        "train": set(),
        "val": set(),
        "test_seen": set(),
        "test_zero_day": set(zero_day_groups),
    }

    seen_df = df[
        (df["attack_family"] != "benign")
        & (df["attack_family"] != zero_day_family)
        & (~df["_group_key"].isin(zero_day_groups))
    ].copy()

    for family, part in seen_df.groupby("attack_family", sort=True):
        family_group_sizes = part.groupby("_group_key").size().sort_values(ascending=False)
        allocations = _weighted_group_split(
            family_group_sizes,
            {"train": train_ratio, "val": val_ratio, "test_seen": test_seen_ratio},
            seed=seed + sum(ord(ch) for ch in family),
        )
        if len(family_group_sizes) == 1:
            only_group = next(iter(family_group_sizes.index.tolist()))
            allocations = {"train": {only_group}, "val": set(), "test_seen": set()}
        elif len(family_group_sizes) == 2 and not allocations["train"]:
            first, second = family_group_sizes.index.tolist()
            allocations = {"train": {first}, "val": {second}, "test_seen": set()}
        for split_name in ("train", "val", "test_seen"):
            split_groups[split_name].update(allocations[split_name])

    assigned_non_benign_groups = (
        split_groups["train"]
        | split_groups["val"]
        | split_groups["test_seen"]
        | split_groups["test_zero_day"]
    )

    benign_df = df[
        (df["attack_family"] == "benign")
        & (~df["_group_key"].isin(assigned_non_benign_groups))
    ].copy()
    benign_group_sizes = benign_df.groupby("_group_key").size().sort_values(ascending=False)
    benign_allocations = _weighted_group_split(
        benign_group_sizes,
        {
            "train": train_ratio,
            "val": val_ratio,
            "test_seen": test_seen_ratio,
            "test_zero_day": test_zero_day_benign_ratio,
        },
        seed=seed,
    )
    for split_name in split_groups:
        split_groups[split_name].update(benign_allocations[split_name])

    split_ids = {}
    for split_name, groups in split_groups.items():
        ids = df.loc[df["_group_key"].isin(groups), "row_id"].to_numpy(copy=True)
        rng = np.random.default_rng(seed + len(split_name))
        rng.shuffle(ids)
        split_ids[split_name] = ids

    all_ids = np.concatenate(list(split_ids.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError("Có row_id trùng giữa các split")

    for left in split_groups:
        for right in split_groups:
            if left >= right:
                continue
            overlap = split_groups[left] & split_groups[right]
            if overlap:
                raise AssertionError(f"Có group trùng giữa {left} và {right}")

    manifests: dict[str, dict] = {}
    for name, ids in split_ids.items():
        split_df = df[df["row_id"].isin(ids)][["row_id", "attack_family", "original_label"]].copy()
        excluded_rows = 0
        if name == "test_zero_day":
            keep_mask = split_df["attack_family"].isin(["benign", zero_day_family])
            excluded_rows = int((~keep_mask).sum())
            split_df = split_df.loc[keep_mask].copy()
        split_df["binary_label"] = (split_df["attack_family"] != "benign").astype(int)
        split_df.to_csv(output_dir / f"{name}.csv", index=False)
        manifests[name] = {
            "rows": int(len(split_df)),
            "groups": int(len(split_groups[name])),
            "excluded_non_target_attack_rows": excluded_rows,
            "attack_family_counts": split_df["attack_family"].value_counts().to_dict(),
            "binary_label_counts": split_df["binary_label"].value_counts().to_dict(),
        }

    for name in ("train", "val"):
        families_in_split = set(pd.read_csv(output_dir / f"{name}.csv")["attack_family"].unique())
        if zero_day_family in families_in_split:
            raise AssertionError(f"{zero_day_family} xuất hiện trong {name}")

    write_json(
        output_dir / "split_manifest.json",
        {
            "dataset": "CIC-IDS2017",
            "experiment_name": f"zero_day_{zero_day_family}",
            "zero_day_family": zero_day_family,
            "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split_rule": "leave-one-attack-family-out",
            "group_columns": list(group_columns),
            "ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "test_seen": test_seen_ratio,
                "test_zero_day_benign": test_zero_day_benign_ratio,
            },
            "splits": manifests,
        },
    )
    return output_dir

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.split_zero_day import create_leave_one_family_out_split
from src.utils.io import read_json


class ZeroDaySplitTests(unittest.TestCase):
    def test_model_selection_val_is_disjoint_and_excludes_zero_day_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.parquet"
            split_dir = root / "splits"
            rows = []
            row_id = 0
            for family, group_count in {
                "benign": 10,
                "botnet": 8,
                "ddos": 8,
                "dos": 5,
            }.items():
                for group_idx in range(group_count):
                    for replica in range(2):
                        rows.append(
                            {
                                "row_id": row_id,
                                "attack_family": family,
                                "original_label": family,
                                "source_file": f"{family}_{group_idx % 2}.parquet",
                                "source_ip": f"10.{group_idx}.{replica}.1",
                                "destination_ip": f"172.16.{group_idx}.10",
                                "destination_port": 80 + group_idx,
                            }
                        )
                        row_id += 1
            pd.DataFrame(rows).to_parquet(clean_path, index=False)

            create_leave_one_family_out_split(
                clean_path=clean_path,
                output_dir=split_dir,
                zero_day_family="dos",
                seed=7,
                train_ratio=0.6,
                val_ratio=0.2,
                test_seen_ratio=0.2,
                test_zero_day_benign_ratio=0.1,
                model_selection_ratio=0.25,
            )

            split_ids = {}
            for split in ("train", "model_selection_val", "val", "test_seen", "test_zero_day"):
                frame = pd.read_csv(split_dir / f"{split}.csv")
                split_ids[split] = set(frame["row_id"].tolist())
                if split in {"train", "model_selection_val", "val"}:
                    self.assertNotIn("dos", set(frame["attack_family"]))

            self.assertGreater(len(split_ids["model_selection_val"]), 0)
            all_ids = set()
            for split, ids in split_ids.items():
                self.assertFalse(all_ids & ids, f"{split} overlaps another split")
                all_ids |= ids

            manifest = read_json(split_dir / "split_manifest.json")
            self.assertTrue(manifest["model_selection_split"]["enabled"])
            self.assertIn("model_selection_val", manifest["splits"])
            self.assertGreater(manifest["splits"]["model_selection_val"]["rows"], 0)


if __name__ == "__main__":
    unittest.main()

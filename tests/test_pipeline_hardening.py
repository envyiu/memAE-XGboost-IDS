from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.run_full_pipeline_all_families import (
    DEFAULT_FAMILIES,
    _compact_candidate,
    _select_primary_candidate,
    _summary_suffix,
)
from src.features.export_memae_features import _memae_feature_dim, export_features
from src.features.window_features import add_window_features, window_feature_names
from src.models.memae.model import MemAE
from src.utils.io import read_json, write_json


class PipelineHardeningTests(unittest.TestCase):
    def test_boosted_window_features_are_finite_and_named(self) -> None:
        cfg = {
            "enabled": True,
            "group_by": ["source_file", "source_ip"],
            "order_by": ["timestamp", "row_id"],
            "window_sizes": [3],
            "time_window_seconds": [60],
            "include_periodicity": True,
            "include_botnet_context": True,
            "include_low_slow": True,
        }
        df = pd.DataFrame(
            {
                "row_id": [1, 2, 3, 4, 5],
                "source_file": ["a"] * 5,
                "source_ip": ["10.0.0.1"] * 5,
                "source_port": [8080, 8080, 4242, 8080, 51515],
                "destination_ip": ["x", "x", "y", "x", "x"],
                "destination_port": [80, 80, 443, 80, 8080],
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="500ms"),
                "syn_flag_count": [1, 1, 0, 1, 0],
                "ack_flag_count": [1, 1, 1, 1, 1],
                "rst_flag_count": [0, 0, 0, 0, 0],
                "fin_flag_count": [0, 0, 0, 0, 0],
                "total_fwd_packets": [1, 2, 10, 1, 2],
                "total_backward_packets": [0, 0, 1, 0, 0],
                "total_length_of_fwd_packets": [100, 120, 2000, 100, 110],
                "total_length_of_bwd_packets": [0, 0, 300, 0, 0],
            }
        )
        out, cols = add_window_features(df, cfg)
        self.assertEqual(set(window_feature_names(cfg)), set(cols))
        for col in (
            "win_dest_repeat_ratio_3",
            "time_dest_repeat_ratio_60s",
            "win_burst_count_3",
            "time_burst_count_60s",
            "win_low_slow_repeat_count_3",
            "time_low_slow_repeat_count_60s",
        ):
            self.assertIn(col, out.columns)
        self.assertTrue(np.isfinite(out[cols].to_numpy(dtype=np.float32)).all())

    def test_primary_candidate_respects_fpr_cap(self) -> None:
        rows = [
            {
                "model_name": "high_zdr_bad_fpr",
                "target_fpr": 0.05,
                "calibration_fpr": 0.05,
                "validation": {"z_dr": 0.9, "fpr": 0.05},
                "test_seen": {"z_dr": 0.8},
                "test_zero_day": {"z_dr": 0.7, "fpr": 0.06, "f1": 0.1},
            },
            {
                "model_name": "lower_zdr_good_fpr",
                "target_fpr": 0.02,
                "calibration_fpr": 0.02,
                "validation": {"z_dr": 0.4, "fpr": 0.02},
                "test_seen": {"z_dr": 0.3},
                "test_zero_day": {"z_dr": 0.2, "fpr": 0.03, "f1": 0.1},
            },
        ]
        selected = _compact_candidate(_select_primary_candidate(rows, 0.05))
        self.assertEqual(selected["model_name"], "lower_zdr_good_fpr")
        self.assertEqual(selected["primary_selection_status"], "PASS")

    def test_single_family_summary_suffix_is_not_generic(self) -> None:
        self.assertEqual(_summary_suffix("targetsel", "host", ["botnet"]), "botnet_host_targetsel")
        self.assertEqual(_summary_suffix("targetsel", "host", list(DEFAULT_FAMILIES)), "host_targetsel")

    def test_memae_export_can_append_raw_processed_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                experiment = "tiny_exp"
                artifact = "tiny_artifact"
                processed_dir = Path("data/processed") / experiment
                artifact_dir = Path("artifacts/memae") / artifact
                processed_dir.mkdir(parents=True)
                artifact_dir.mkdir(parents=True)

                rng = np.random.default_rng(123)
                for split, n_rows in {
                    "train": 5,
                    "val": 3,
                    "test_seen": 4,
                    "test_zero_day": 2,
                }.items():
                    X = rng.normal(size=(n_rows, 3)).astype("float32")
                    np.save(processed_dir / f"X_{split}.npy", X)
                write_json(processed_dir / "feature_schema.json", {"feature_order": ["a", "b", "c"]})

                model_config = {"latent_dim": 2, "memory_size": 4, "shrink_threshold": 0.0}
                model = MemAE(3, **model_config)
                torch.save(
                    {
                        "input_dim": 3,
                        "model_config": model_config,
                        "model_state_dict": model.state_dict(),
                    },
                    artifact_dir / "memae_best.pt",
                )

                plain_dir = export_features(
                    experiment,
                    batch_size=2,
                    artifact_name=artifact,
                    feature_set="plain_features",
                )
                raw_dir = export_features(
                    experiment,
                    batch_size=2,
                    artifact_name=artifact,
                    feature_set="raw_features",
                    include_raw_input=True,
                )

                plain_schema = read_json(plain_dir / "memae_feature_schema.json")
                raw_schema = read_json(raw_dir / "memae_feature_schema.json")
                memae_dim = _memae_feature_dim(3, 2)

                self.assertFalse(plain_schema["include_raw_input"])
                self.assertEqual(plain_schema["total_dims_numeric"], memae_dim)
                self.assertTrue(raw_schema["include_raw_input"])
                self.assertEqual(raw_schema["raw_input_dim"], 3)
                self.assertEqual(raw_schema["memae_feature_dim"], memae_dim)
                self.assertEqual(raw_schema["total_dims_numeric"], memae_dim + 3)
                self.assertIn("raw_processed_input", raw_schema["feature_blocks"])
                self.assertEqual(np.load(raw_dir / "F_train.npy", mmap_mode="r").shape, (5, memae_dim + 3))

                selected_dir = export_features(
                    experiment,
                    batch_size=2,
                    artifact_name=artifact,
                    feature_set="selected_raw_features",
                    include_raw_input=True,
                    raw_input_feature_patterns=["b"],
                )
                selected_schema = read_json(selected_dir / "memae_feature_schema.json")
                self.assertEqual(selected_schema["raw_input_dim"], 1)
                self.assertEqual(selected_schema["raw_input_feature_names"], ["b"])
                self.assertEqual(np.load(selected_dir / "F_train.npy", mmap_mode="r").shape, (5, memae_dim + 1))
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()

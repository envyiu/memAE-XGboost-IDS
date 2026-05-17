from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.preprocessor import IDSPreprocessor
from src.preprocessing.run_preprocessing import preprocess_experiment
from src.utils.io import write_json


class IDSPreprocessorTests(unittest.TestCase):
    def test_all_nan_feature_is_preserved_after_transform(self) -> None:
        train = np.array(
            [
                [np.nan, 1.0],
                [np.nan, 2.0],
                [np.nan, 3.0],
            ],
            dtype=np.float32,
        )
        test = np.array([[np.nan, 2.0], [np.inf, 4.0]], dtype=np.float32)
        preprocessor = IDSPreprocessor(["all_nan", "value"])

        preprocessor.fit(train)
        transformed = preprocessor.transform(test)

        self.assertEqual(transformed.shape, (2, 2))
        self.assertEqual(transformed.dtype, np.float32)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(transformed[:, 0], np.zeros(2, dtype=np.float32))

    def test_numpy_input_must_match_feature_count(self) -> None:
        preprocessor = IDSPreprocessor(["a", "b"])

        with self.assertRaisesRegex(ValueError, "Expected 2 feature columns"):
            preprocessor.fit(np.zeros((3, 1), dtype=np.float32))

    def test_dataframe_input_reports_missing_feature_columns(self) -> None:
        preprocessor = IDSPreprocessor(["a", "b"])
        frame = pd.DataFrame({"a": [1.0, 2.0]})

        with self.assertRaisesRegex(ValueError, "Missing feature columns"):
            preprocessor.fit(frame)

    def test_full_source_preprocessing_parallel_matches_sequential(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                root = Path(tmp)
                clean_path = root / "clean.parquet"
                schema_path = root / "schema.json"
                split_dir = root / "splits"
                split_dir.mkdir()

                frame = pd.DataFrame(
                    {
                        "row_id": np.arange(8, dtype=np.int64),
                        "attack_family": [
                            "benign",
                            "benign",
                            "dos",
                            "benign",
                            "benign",
                            "botnet",
                            "benign",
                            "dos",
                        ],
                        "source_file": ["a.parquet"] * 4 + ["b.parquet"] * 4,
                        "source_ip": ["10.0.0.1"] * 4 + ["10.0.0.2"] * 4,
                        "timestamp": pd.date_range("2024-01-01", periods=8, freq="s"),
                        "duration": np.linspace(1.0, 8.0, 8, dtype=np.float32),
                    }
                )
                frame.to_parquet(clean_path, index=False)
                write_json(
                    schema_path,
                    {
                        "all_columns": frame.columns.tolist(),
                        "numerical_features": ["duration"],
                    },
                )
                for name, row_ids in {
                    "train": [0, 1, 4],
                    "model_selection_val": [2, 5],
                    "val": [3],
                    "test_seen": [6],
                    "test_zero_day": [7],
                }.items():
                    pd.DataFrame({"row_id": row_ids}).to_csv(split_dir / f"{name}.csv", index=False)

                window_config = {
                    "enabled": True,
                    "window_scope": "full_source_file",
                    "group_by": ["source_file", "source_ip"],
                    "order_by": ["timestamp", "row_id"],
                    "window_sizes": [2],
                    "time_window_seconds": [],
                    "include_flow_count": True,
                    "include_unique_destination_port": False,
                    "include_destination_port_count": False,
                    "include_unique_destination_ip": False,
                    "include_destination_ip_count": False,
                    "include_destination_service_count": False,
                    "include_flag_ratios": False,
                    "include_packet_byte_sums": False,
                    "include_unique_port_ratio": False,
                    "include_service_context": False,
                    "include_behavior_proxies": False,
                    "include_periodicity": False,
                    "include_botnet_context": False,
                    "include_low_slow": False,
                    "include_source_port_context": False,
                    "include_watched_ports": False,
                    "include_port_diversity": False,
                    "include_timing_regularity": False,
                    "include_dest_concentration": False,
                    "include_beaconing_detection": False,
                }

                preprocess_experiment(
                    experiment="seq",
                    clean_path=clean_path,
                    split_dir=split_dir,
                    schema_path=schema_path,
                    window_config=window_config,
                    fit_sample_rows=100,
                    preprocess_device="cpu",
                    preprocess_num_workers=1,
                )
                preprocess_experiment(
                    experiment="par",
                    clean_path=clean_path,
                    split_dir=split_dir,
                    schema_path=schema_path,
                    window_config=window_config,
                    fit_sample_rows=100,
                    preprocess_device="cpu",
                    preprocess_num_workers=2,
                )

                for split in ("train", "model_selection_val", "val", "test_seen", "test_zero_day"):
                    seq_dir = Path("data/processed/seq")
                    par_dir = Path("data/processed/par")
                    np.testing.assert_allclose(
                        np.load(seq_dir / f"X_{split}.npy"),
                        np.load(par_dir / f"X_{split}.npy"),
                    )
                    np.testing.assert_array_equal(
                        np.load(seq_dir / f"row_id_{split}.npy"),
                        np.load(par_dir / f"row_id_{split}.npy"),
                    )
                    np.testing.assert_array_equal(
                        np.load(seq_dir / f"y_{split}.npy"),
                        np.load(par_dir / f"y_{split}.npy"),
                    )
                schema = pd.read_json(Path("data/processed/par/feature_schema.json"), typ="series")
                self.assertIn("model_selection_val", schema["split_names"])
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from src.models.xgboost.train_feature_set import _qualified_attack_family_counts, _validation_split_name


class XGBoostTrainingTests(unittest.TestCase):
    def test_qualified_attack_family_counts_ignores_tiny_families(self) -> None:
        family = np.array(["benign", "botnet", "botnet", "ddos", "ddos", "ddos", "dos"], dtype=object)
        y = np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int64)

        counts = _qualified_attack_family_counts(family, y, min_samples_per_family=3)

        self.assertEqual(counts, {"ddos": 3})

    def test_validation_split_prefers_model_selection_holdout_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            processed_dir = root / "processed"
            feature_dir.mkdir()
            processed_dir.mkdir()
            (feature_dir / "F_model_selection_val.npy").touch()
            (processed_dir / "y_model_selection_val.npy").touch()
            (processed_dir / "family_model_selection_val.npy").touch()

            self.assertEqual(_validation_split_name(feature_dir, processed_dir), "model_selection_val")


if __name__ == "__main__":
    unittest.main()

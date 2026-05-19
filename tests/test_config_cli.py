from __future__ import annotations

import importlib
import unittest

from src.preprocessing.preprocessor import IDSPreprocessor
from src.utils.io import read_yaml


class ConfigAndCliTests(unittest.TestCase):
    def test_default_configs_load_required_sections(self) -> None:
        memae = read_yaml("configs/memae_targeted.yaml")
        tabtrans = read_yaml("configs/tabtrans_zdr5.yaml")
        tabtrans_kaggle = read_yaml("configs/tabtrans_kaggle_t4x2.yaml")
        xgboost = read_yaml("configs/xgboost_zdr5.yaml")
        window = read_yaml("configs/window_features_zdr5.yaml")

        self.assertIn("model", memae)
        self.assertIn("training", memae)
        self.assertIn("model", tabtrans)
        self.assertIn("training", tabtrans)
        self.assertTrue(tabtrans_kaggle["training"]["data_parallel"])
        self.assertTrue(tabtrans_kaggle["training"]["amp"])
        self.assertIn("binary_detection", xgboost)
        self.assertIn("threshold", xgboost)
        self.assertTrue(window["enabled"])

    def test_preprocess_auto_device_resolves_to_available_backend(self) -> None:
        self.assertIn(IDSPreprocessor.resolve_device("auto"), {"cpu", "cuda"})

    def test_cli_modules_import_without_running_pipeline(self) -> None:
        full = importlib.import_module("scripts.run_full_pipeline_all_families")
        kaggle = importlib.import_module("scripts.run_kaggle_pipeline")

        self.assertIn("reports", full.STAGES)
        self.assertTrue(hasattr(kaggle, "main"))


if __name__ == "__main__":
    unittest.main()

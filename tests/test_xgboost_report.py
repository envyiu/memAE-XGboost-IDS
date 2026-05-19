from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.evaluation.detector_calibration import generate_xgboost_calibration_report
from src.utils.io import read_json, write_json


class XGBoostReportTests(unittest.TestCase):
    def test_xgboost_only_report_uses_generic_representation_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                experiment = "tiny_exp"
                feature_set = "tiny_features"
                processed_dir = Path("data/processed") / experiment
                feature_dir = Path("data/features") / feature_set
                xgb_dir = Path("artifacts/xgboost") / feature_set
                processed_dir.mkdir(parents=True)
                feature_dir.mkdir(parents=True)
                xgb_dir.mkdir(parents=True)

                X_fit = np.array(
                    [
                        [0.0, 0.0],
                        [0.1, 0.0],
                        [0.0, 0.1],
                        [1.0, 1.0],
                        [1.1, 1.0],
                        [1.0, 1.1],
                    ],
                    dtype=np.float32,
                )
                y_fit = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
                model = xgb.XGBClassifier(
                    n_estimators=3,
                    max_depth=1,
                    learning_rate=0.5,
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                )
                model.fit(X_fit, y_fit, verbose=False)
                model.save_model(xgb_dir / "xgboost_model.json")
                write_json(xgb_dir / "feature_selection.json", {"enabled": False, "selected_indices": None})

                splits = {
                    "model_selection_val": (X_fit, y_fit, np.array(["benign", "benign", "benign", "dos", "dos", "dos"], dtype=object)),
                    "test_seen": (X_fit, y_fit, np.array(["benign", "benign", "benign", "botnet", "botnet", "botnet"], dtype=object)),
                    "test_zero_day": (X_fit, y_fit, np.array(["benign", "benign", "benign", "web_attack", "web_attack", "web_attack"], dtype=object)),
                }
                for split, (X, y, family) in splits.items():
                    np.save(feature_dir / f"F_{split}.npy", X)
                    np.save(processed_dir / f"y_{split}.npy", y)
                    np.save(processed_dir / f"family_{split}.npy", family)

                write_json(
                    processed_dir / "feature_schema.json",
                    {"feature_order": ["a", "b"], "benchmark_mode": "tiny"},
                )
                write_json(
                    feature_dir / "representation_feature_schema.json",
                    {
                        "architecture": "tabtrans",
                        "D_value": 2,
                        "representation_feature_dim": 2,
                        "total_dims_numeric": 2,
                    },
                )

                path = generate_xgboost_calibration_report(
                    experiment,
                    feature_set,
                    feature_set,
                    fpr_budgets=(0.5,),
                )
                report = read_json(path)

                self.assertEqual(report["architecture"], "tabtrans")
                self.assertEqual(report["candidate_rows"][0]["model_name"], "xgboost")
                self.assertIn("test_zero_day", report["candidate_rows"][0])
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()

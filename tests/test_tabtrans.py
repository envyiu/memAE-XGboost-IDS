from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.features.export_tabtrans_features import export_tabtrans_features
from src.models.tabtrans.model import NumericTabTransformer
from src.models.tabtrans.train_tabtrans import train_tabtrans
from src.utils.io import read_json, write_json


class TabTransformerTests(unittest.TestCase):
    def test_model_forward_and_extract_features_shapes(self) -> None:
        model = NumericTabTransformer(
            5,
            embed_dim=8,
            depth=1,
            heads=2,
            latent_dim=6,
            attn_dropout=0.0,
            ff_dropout=0.0,
            classifier_dropout=0.0,
        )
        x = torch.zeros((3, 5), dtype=torch.float32)

        logits, features = model(x, return_features=True)

        self.assertEqual(logits.shape, (3,))
        self.assertEqual(features.shape, (3, 6))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(features).all())

    def test_train_and_export_tabtrans_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                experiment = "tiny_tab"
                artifact = "tiny_tab_artifact"
                processed_dir = Path("data/processed") / experiment
                processed_dir.mkdir(parents=True)
                rng = np.random.default_rng(123)
                X_train = rng.normal(size=(12, 5)).astype("float32")
                y_train = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
                X_val = rng.normal(size=(6, 5)).astype("float32")
                y_val = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
                for split, (X, y) in {
                    "train": (X_train, y_train),
                    "model_selection_val": (X_val, y_val),
                    "val": (X_val + 0.1, y_val),
                    "test_seen": (X_val + 0.2, y_val),
                    "test_zero_day": (X_val + 0.3, y_val),
                }.items():
                    np.save(processed_dir / f"X_{split}.npy", X)
                    np.save(processed_dir / f"y_{split}.npy", y)
                write_json(
                    processed_dir / "feature_schema.json",
                    {
                        "feature_order": ["a", "b", "c", "d", "e"],
                        "split_names": ["train", "model_selection_val", "val", "test_seen", "test_zero_day"],
                    },
                )

                cfg = {
                    "model": {
                        "embed_dim": 8,
                        "depth": 1,
                        "heads": 2,
                        "latent_dim": 6,
                        "attn_dropout": 0.0,
                        "ff_dropout": 0.0,
                        "classifier_dropout": 0.0,
                    },
                    "training": {
                        "epochs": 2,
                        "min_epochs": 1,
                        "batch_size": 4,
                        "eval_batch_size": 3,
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "patience": 2,
                        "device": "cpu",
                        "num_workers": 0,
                    },
                    "selection": {"metric": "val_aucpr"},
                }
                checkpoint_path = train_tabtrans(experiment, cfg, seed=123, artifact_name=artifact)
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.assertEqual(checkpoint["architecture"], "tabtrans")
                self.assertEqual(checkpoint["input_dim"], 5)
                self.assertEqual(checkpoint["validation_split"], "model_selection_val")

                feature_dir = export_tabtrans_features(
                    experiment,
                    batch_size=2,
                    artifact_name=artifact,
                    feature_set="tiny_features",
                    include_raw_input=True,
                    raw_input_feature_patterns=["b", "d"],
                )
                schema = read_json(feature_dir / "representation_feature_schema.json")
                self.assertEqual(schema["architecture"], "tabtrans")
                self.assertEqual(schema["D_value"], 5)
                self.assertEqual(schema["representation_feature_dim"], 6)
                self.assertEqual(schema["raw_input_dim"], 2)
                self.assertEqual(schema["total_dims_numeric"], 8)
                self.assertIn("model_selection_val", schema["split_names"])
                self.assertEqual(np.load(feature_dir / "F_train.npy", mmap_mode="r").shape, (12, 8))
                self.assertEqual(np.load(feature_dir / "F_model_selection_val.npy", mmap_mode="r").shape, (6, 8))
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()

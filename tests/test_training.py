from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.memae.model import MemAE
from src.models.memae.train_memae import _reconstruction_errors, train_memae
from src.utils.io import read_json


class MemAETrainingTests(unittest.TestCase):
    def test_reconstruction_errors_are_per_sample_on_cpu(self) -> None:
        model = MemAE(3, latent_dim=2, memory_size=2, hidden_dims=[4], shrink_threshold=0.0)
        X = np.zeros((5, 3), dtype=np.float32)

        errors = _reconstruction_errors(
            model,
            X,
            torch.device("cpu"),
            batch_size=2,
            reduction="mean",
        )

        self.assertEqual(errors.shape, (5,))
        self.assertTrue(np.isfinite(errors).all())

    def test_train_memae_saves_checkpoint_metadata_and_handles_last_batch_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                processed_dir = Path("data/processed/tiny")
                processed_dir.mkdir(parents=True)
                X_train = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0],
                        [0.0, 0.1, 0.0],
                        [0.0, 0.0, 0.1],
                        [3.0, 3.0, 3.0],
                    ],
                    dtype=np.float32,
                )
                y_train = np.array([0, 0, 0, 0, 1], dtype=np.int64)
                X_val = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, 0.1, 0.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 2.0, 3.0],
                    ],
                    dtype=np.float32,
                )
                y_val = np.array([0, 0, 1, 1], dtype=np.int64)
                np.save(processed_dir / "X_train.npy", X_train)
                np.save(processed_dir / "y_train.npy", y_train)
                np.save(processed_dir / "X_val.npy", X_val)
                np.save(processed_dir / "y_val.npy", y_val)
                np.save(processed_dir / "X_model_selection_val.npy", X_val + 0.01)
                np.save(processed_dir / "y_model_selection_val.npy", y_val)

                checkpoint_path = train_memae(
                    "tiny",
                    {
                        "model": {
                            "latent_dim": 2,
                            "memory_size": 2,
                            "shrink_threshold": 0.0,
                            "hidden_dims": [4],
                        },
                        "training": {
                            "epochs": 2,
                            "batch_size": 3,
                            "eval_batch_size": 2,
                            "learning_rate": 0.01,
                            "weight_decay": 0.0,
                            "entropy_weight": 0.0,
                            "patience": 2,
                            "min_epochs": 2,
                            "num_workers": 0,
                            "device": "cpu",
                        },
                        "selection": {"metric": "val_loss"},
                    },
                    seed=123,
                    artifact_name="tiny_artifact",
                )

                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.assertEqual(checkpoint["input_dim"], 3)
                self.assertIn("optimizer_state_dict", checkpoint)
                self.assertIn("scheduler_state_dict", checkpoint)
                self.assertIn("best_epoch", checkpoint)
                self.assertIn("best_val_loss", checkpoint)
                self.assertGreaterEqual(checkpoint["best_epoch"], 2)
                self.assertEqual(checkpoint["training_config"]["batch_size"], 3)
                self.assertEqual(checkpoint["seed"], 123)
                self.assertEqual(checkpoint["validation_split"], "model_selection_val")
                log = read_json(Path("artifacts/memae/tiny_artifact/training_log.json"))
                self.assertEqual(log["validation_split"], "model_selection_val")
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from src.utils.scoring import metrics_at_threshold, metrics_from_pred, threshold_for_fpr


class ScoringTests(unittest.TestCase):
    def test_zero_fpr_budget_selects_threshold_above_max_score(self) -> None:
        scores = np.array([0.1, 0.2, 0.2], dtype=np.float32)

        selected = threshold_for_fpr(scores, 0.0, fallback_mode="percentile")

        self.assertGreater(selected["threshold"], float(scores.max()))
        self.assertEqual(selected["calibration_fpr"], 0.0)

    def test_percentile_fallback_does_not_exceed_budget_for_tied_scores(self) -> None:
        scores = np.zeros(5, dtype=np.float32)

        selected = threshold_for_fpr(scores, 0.2, fallback_mode="percentile")

        self.assertLessEqual(selected["calibration_fpr"], 0.2)
        self.assertGreater(selected["threshold"], float(scores.max()))

    def test_threshold_rejects_non_finite_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            threshold_for_fpr(np.array([0.0, np.nan], dtype=np.float32), 0.1)

    def test_metrics_reject_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "lengths"):
            metrics_from_pred(
                np.array([0, 1]),
                np.array(["benign", "dos"]),
                np.array([0]),
            )

    def test_metrics_at_threshold_rejects_score_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "lengths"):
            metrics_at_threshold(
                np.array([0, 1]),
                np.array(["benign", "dos"]),
                np.array([0.1]),
                0.5,
            )


if __name__ == "__main__":
    unittest.main()

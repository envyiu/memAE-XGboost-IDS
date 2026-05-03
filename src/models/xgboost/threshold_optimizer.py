from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_t: float = 0.01, max_t: float = 0.99, steps: int = 99):
    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in np.linspace(min_t, max_t, steps):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best["f1"]:
            best = {"threshold": float(threshold), "f1": float(score)}
    return best

from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("empty arrays")

    denom = np.abs(y_true) + np.abs(y_pred)
    if np.any(denom == 0.0):
        raise ZeroDivisionError("sMAPE undefined when y_true and y_pred are both zero")

    return float(np.mean(200.0 * np.abs(y_true - y_pred) / denom))

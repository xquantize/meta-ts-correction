"""Mean Absolute Scaled Error (seasonal-naive denominator)."""

from __future__ import annotations

import numpy as np


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Point-forecast MASE with seasonal-naive scale from the training window.

    scale = mean(|y_train[t] - y_train[t - seasonality]|) over t >= seasonality
    MASE  = mean(|y_true - y_pred|) / scale
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if seasonality < 1:
        raise ValueError("seasonality must be >= 1")
    if len(y_train) <= seasonality:
        raise ValueError("y_train must be longer than seasonality")

    diffs = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = float(np.mean(diffs))
    if scale == 0.0:
        raise ZeroDivisionError("seasonal-naive scale is zero; series may be constant")

    return float(np.mean(np.abs(y_true - y_pred)) / scale)

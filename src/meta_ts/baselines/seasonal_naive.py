from __future__ import annotations

import numpy as np


def seasonal_naive(
    train: np.ndarray,
    horizon: int,
    seasonality: int = 1,
) -> np.ndarray:
    y = np.asarray(train, dtype=float).ravel()

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if seasonality < 1:
        raise ValueError("seasonality must be >= 1")
    if len(y) < seasonality:
        raise ValueError(f"train length {len(y)} is shorter than seasonality {seasonality}")

    last_season = y[-seasonality:]
    reps = int(np.ceil(horizon / seasonality))
    return np.tile(last_season, reps)[:horizon]

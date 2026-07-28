from __future__ import annotations

import numpy as np


def series_meta_features(
    train: np.ndarray,
    *,
    seasonality: int,
    horizon: int,
) -> dict[str, float | int]:
    y = np.asarray(train, dtype=float).ravel()
    if len(y) < 2:
        raise ValueError("train series too short for meta-features")

    mean = float(np.mean(y))
    std = float(np.std(y))
    t = np.arange(len(y), dtype=float)
    trend = float(np.corrcoef(t, y)[0, 1]) if std > 0 else 0.0
    if not np.isfinite(trend):
        trend = 0.0

    seasonal_strength = 0.0
    if seasonality > 1 and len(y) > seasonality + 1:
        centered = y - mean
        lag = centered[seasonality:]
        lead = centered[:-seasonality]
        denom = float(np.sqrt(np.sum(lead**2) * np.sum(lag**2)))
        if denom > 0:
            seasonal_strength = float(np.dot(lead, lag) / denom)

    return {
        "seasonality": int(seasonality),
        "horizon": int(horizon),
        "n_train": len(y),
        "mean": mean,
        "std": std,
        "cv": float(std / abs(mean)) if mean != 0.0 else 0.0,
        "trend_corr": trend,
        "seasonal_corr": float(seasonal_strength),
        "abs_diff_mean": float(np.mean(np.abs(np.diff(y)))),
    }

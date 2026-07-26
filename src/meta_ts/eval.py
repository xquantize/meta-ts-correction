from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meta_ts.baselines.seasonal_naive import seasonal_naive
from meta_ts.data.windows import ForecastWindow, rolling_origin_windows
from meta_ts.metrics.mase import mase


@dataclass(frozen=True)
class WindowScore:
    origin: int
    mase: float


def score_seasonal_naive(
    series: np.ndarray,
    *,
    horizon: int,
    min_train_size: int,
    seasonality: int,
    stride: int = 1,
) -> list[WindowScore]:
    windows = rolling_origin_windows(
        series,
        horizon=horizon,
        min_train_size=min_train_size,
        stride=stride,
    )
    return [_score_window(w, seasonality) for w in windows]


def mean_mase(scores: list[WindowScore]) -> float:
    if not scores:
        raise ValueError("scores is empty")
    return float(np.mean([s.mase for s in scores]))


def _score_window(window: ForecastWindow, seasonality: int) -> WindowScore:
    y_pred = seasonal_naive(window.train, len(window.test), seasonality)
    return WindowScore(
        origin=window.origin,
        mase=mase(window.test, y_pred, window.train, seasonality=seasonality),
    )

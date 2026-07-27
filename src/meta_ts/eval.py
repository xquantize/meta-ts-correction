from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from meta_ts.data.windows import ForecastWindow, rolling_origin_windows
from meta_ts.metrics.mase import mase


class Predictor(Protocol):
    def __call__(
        self,
        train: np.ndarray,
        horizon: int,
        seasonality: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class WindowScore:
    origin: int
    mase: float


def score_predictor(
    series: np.ndarray,
    predict: Predictor | Callable[..., np.ndarray],
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
    return [_score_window(w, predict, seasonality) for w in windows]


def score_seasonal_naive(
    series: np.ndarray,
    *,
    horizon: int,
    min_train_size: int,
    seasonality: int,
    stride: int = 1,
) -> list[WindowScore]:
    from meta_ts.baselines.seasonal_naive import seasonal_naive

    return score_predictor(
        series,
        seasonal_naive,
        horizon=horizon,
        min_train_size=min_train_size,
        seasonality=seasonality,
        stride=stride,
    )


def mean_mase(scores: list[WindowScore]) -> float:
    if not scores:
        raise ValueError("scores is empty")
    return float(np.mean([s.mase for s in scores]))


def _score_window(
    window: ForecastWindow,
    predict: Predictor | Callable[..., np.ndarray],
    seasonality: int,
) -> WindowScore:
    y_pred = np.asarray(predict(window.train, len(window.test), seasonality), dtype=float)
    return WindowScore(
        origin=window.origin,
        mase=mase(window.test, y_pred, window.train, seasonality=seasonality),
    )

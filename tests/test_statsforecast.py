from __future__ import annotations

import numpy as np
import pytest

statsforecast = pytest.importorskip("statsforecast")

from meta_ts.baselines.statsforecast_models import auto_arima, auto_ets
from meta_ts.eval import mean_mase, score_predictor, score_seasonal_naive


def _seasonal_series(n: int = 96, period: int = 12) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return 10.0 + 0.05 * t + 2.0 * np.sin(2 * np.pi * t / period)


def test_auto_arima_returns_horizon_length():
    train = _seasonal_series(80)
    pred = auto_arima(train, horizon=6, seasonality=12)
    assert pred.shape == (6,)
    assert np.isfinite(pred).all()


def test_auto_ets_returns_horizon_length():
    train = _seasonal_series(80)
    pred = auto_ets(train, horizon=6, seasonality=12)
    assert pred.shape == (6,)
    assert np.isfinite(pred).all()


def test_auto_ets_beats_naive_on_trending_seasonal():
    series = _seasonal_series(120, period=12)
    kwargs = {"horizon": 12, "min_train_size": 60, "seasonality": 12, "stride": 12}
    naive = mean_mase(score_seasonal_naive(series, **kwargs))
    ets = mean_mase(score_predictor(series, auto_ets, **kwargs))
    assert ets < naive

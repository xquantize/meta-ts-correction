from __future__ import annotations

import numpy as np
import pytest

from meta_ts.baselines.seasonal_naive import seasonal_naive
from meta_ts.data.windows import rolling_origin_windows
from meta_ts.eval import mean_mase, score_seasonal_naive


def test_seasonal_naive_period_one_repeats_last():
    train = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(seasonal_naive(train, horizon=3, seasonality=1), [4.0, 4.0, 4.0])


def test_seasonal_naive_repeats_last_season():
    train = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    pred = seasonal_naive(train, horizon=5, seasonality=3)
    assert np.array_equal(pred, [40.0, 50.0, 60.0, 40.0, 50.0])


def test_seasonal_naive_rejects_short_train():
    with pytest.raises(ValueError):
        seasonal_naive(np.array([1.0, 2.0]), horizon=2, seasonality=3)


def test_seasonal_naive_matches_pure_seasonal_windows():
    season = np.array([1.0, 2.0, 3.0, 4.0])
    series = np.tile(season, 6)
    windows = rolling_origin_windows(series, horizon=4, min_train_size=8, stride=4)
    for window in windows:
        pred = seasonal_naive(window.train, horizon=4, seasonality=4)
        assert np.array_equal(pred, window.test)


def test_score_seasonal_naive_returns_finite_mase():
    rng = np.random.default_rng(0)
    season = np.array([1.0, 2.0, 3.0, 4.0])
    series = np.tile(season, 8) + 0.05 * rng.normal(size=32)
    scores = score_seasonal_naive(
        series,
        horizon=4,
        min_train_size=8,
        seasonality=4,
        stride=4,
    )
    assert len(scores) > 0
    assert np.isfinite(mean_mase(scores))
    assert mean_mase(scores) > 0.0

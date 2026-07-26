from __future__ import annotations

import numpy as np
import pytest

from meta_ts.metrics.mase import mase


def test_mase_perfect_forecast_is_zero():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_true = np.array([7.0, 8.0])
    y_pred = y_true.copy()
    assert mase(y_true, y_pred, y_train, seasonality=1) == 0.0


def test_mase_matches_hand_calculation():
    # train: 1,2,3,4  seasonality=1 → |diffs| = 1,1,1 → scale=1
    # true: 5,6  pred: 4,8 → abs err = 1,2 → mean=1.5 → MASE=1.5
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = np.array([5.0, 6.0])
    y_pred = np.array([4.0, 8.0])
    assert mase(y_true, y_pred, y_train, seasonality=1) == pytest.approx(1.5)


def test_mase_rejects_bad_shapes():
    with pytest.raises(ValueError):
        mase(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0, 3.0]), 1)

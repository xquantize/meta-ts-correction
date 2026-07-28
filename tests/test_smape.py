from __future__ import annotations

import numpy as np
import pytest

from meta_ts.metrics.smape import smape


def test_smape_perfect_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y) == 0.0


def test_smape_hand_calculation():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 90.0])
    # 200*|10|/210 + 200*|10|/190  → mean
    expected = np.mean([200.0 * 10.0 / 210.0, 200.0 * 10.0 / 190.0])
    assert smape(y_true, y_pred) == pytest.approx(expected)


def test_smape_rejects_both_zero():
    with pytest.raises(ZeroDivisionError):
        smape(np.array([0.0]), np.array([0.0]))

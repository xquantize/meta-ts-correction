from __future__ import annotations

import numpy as np
import pytest

from meta_ts.data.windows import rolling_origin_windows


def test_rolling_origin_basic_counts_and_cuts():
    series = np.arange(20, dtype=float)
    windows = rolling_origin_windows(series, horizon=3, min_train_size=10, stride=2)

    assert len(windows) == 4
    assert [w.cut for w in windows] == [10, 12, 14, 16]
    assert [w.origin for w in windows] == [0, 1, 2, 3]


def test_rolling_origin_no_train_test_leakage():
    series = np.arange(15, dtype=float)
    windows = rolling_origin_windows(series, horizon=2, min_train_size=8, stride=1)

    for w in windows:
        assert len(w.train) == w.cut
        assert len(w.test) == 2
        assert w.train[-1] == series[w.cut - 1]
        assert w.test[0] == series[w.cut]
        assert np.array_equal(w.train, series[: w.cut])
        assert np.array_equal(w.test, series[w.cut : w.cut + 2])


def test_rolling_origin_last_window_reaches_end():
    series = np.arange(12, dtype=float)
    windows = rolling_origin_windows(series, horizon=3, min_train_size=6, stride=1)

    last = windows[-1]
    assert last.cut + len(last.test) == len(series)
    assert np.array_equal(last.test, series[-3:])


def test_rolling_origin_copies_are_independent():
    series = np.arange(10, dtype=float)
    windows = rolling_origin_windows(series, horizon=2, min_train_size=5, stride=1)
    windows[0].train[0] = -1.0
    assert series[0] == 0.0


def test_rolling_origin_rejects_short_series():
    with pytest.raises(ValueError):
        rolling_origin_windows(np.arange(5, dtype=float), horizon=3, min_train_size=4)

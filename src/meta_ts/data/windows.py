from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastWindow:
    origin: int
    train: np.ndarray
    test: np.ndarray
    cut: int


def rolling_origin_windows(
    series: np.ndarray,
    *,
    horizon: int,
    min_train_size: int,
    stride: int = 1,
) -> list[ForecastWindow]:
    y = np.asarray(series, dtype=float).ravel()
    n = len(y)

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if min_train_size < 1:
        raise ValueError("min_train_size must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if n < min_train_size + horizon:
        raise ValueError(
            f"series length {n} is shorter than min_train_size + horizon "
            f"({min_train_size + horizon})"
        )

    windows: list[ForecastWindow] = []
    origin = 0
    cut = min_train_size
    while cut + horizon <= n:
        windows.append(
            ForecastWindow(
                origin=origin,
                train=y[:cut].copy(),
                test=y[cut : cut + horizon].copy(),
                cut=cut,
            )
        )
        origin += 1
        cut += stride

    return windows

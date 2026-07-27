from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4, M4Info


@dataclass(frozen=True)
class M4Series:
    series_id: str
    train: np.ndarray
    test: np.ndarray
    seasonality: int
    horizon: int


def load_m4_group(group: str, directory: str = "data/raw") -> list[M4Series]:
    info = M4Info[group]
    df, *_ = M4.load(directory, group)
    horizon = int(info.horizon)
    seasonality = int(info.seasonality)
    series: list[M4Series] = []
    for series_id, frame in df.groupby("unique_id", sort=False):
        y = frame["y"].to_numpy(dtype=float)
        if len(y) <= horizon:
            raise ValueError(f"{series_id} shorter than horizon {horizon}")
        series.append(
            M4Series(
                series_id=str(series_id),
                train=y[:-horizon].copy(),
                test=y[-horizon:].copy(),
                seasonality=seasonality,
                horizon=horizon,
            )
        )
    return series


def load_naive2_forecasts(
    directory: str = "data/raw",
    path: str | None = None,
) -> pd.DataFrame:
    csv_path = path or f"{directory}/m4/datasets/submission-Naive2.csv"
    return pd.read_csv(csv_path)

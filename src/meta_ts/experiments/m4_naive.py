from __future__ import annotations

import numpy as np

from meta_ts.baselines.seasonal_naive import seasonal_naive
from meta_ts.data.m4 import M4Series
from meta_ts.experiments.m4_forecast import run_m4_point_forecast


def run_m4_seasonal_naive(
    config_path: str,
    *,
    base: str = "outputs",
    data_dir: str = "data/raw",
    use_cache: bool = True,
) -> str:
    return run_m4_point_forecast(
        config_path,
        predict_batch=_predict_seasonal_naive,
        base=base,
        data_dir=data_dir,
        use_cache=use_cache,
    )


def _predict_seasonal_naive(series: list[M4Series]) -> dict[str, np.ndarray]:
    return {
        item.series_id: seasonal_naive(item.train, item.horizon, item.seasonality)
        for item in series
    }

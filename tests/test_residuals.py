from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta_ts.data.m4 import M4Series
from meta_ts.data.meta_features import series_meta_features
from meta_ts.data.residuals import build_residual_tables, write_residual_dataset
from meta_ts.results.paths import residual_paths


def test_meta_features_basic():
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    meta = series_meta_features(train, seasonality=1, horizon=2)
    assert meta["n_train"] == 6
    assert meta["mean"] == pytest.approx(3.5)
    assert meta["trend_corr"] > 0.9


def test_build_residual_tables():
    series = [
        M4Series(
            series_id="H1",
            train=np.arange(10, dtype=float),
            test=np.array([10.0, 11.0]),
            seasonality=1,
            horizon=2,
        )
    ]
    forecasts = pd.DataFrame(
        {
            "series_id": ["H1", "H1"],
            "model": ["chronos", "chronos"],
            "step": [1, 2],
            "y_true": [10.0, 11.0],
            "y_pred": [9.5, 12.0],
        }
    )
    residuals, meta = build_residual_tables(series, forecasts)
    assert len(residuals) == 2
    assert residuals["residual"].tolist() == pytest.approx([0.5, -1.0])
    assert len(meta) == 1
    assert meta.iloc[0]["series_id"] == "H1"


def test_write_residual_dataset(tmp_path):
    residuals = pd.DataFrame(
        {
            "series_id": ["H1"],
            "model": ["chronos"],
            "step": [1],
            "y_true": [1.0],
            "y_pred": [0.8],
            "residual": [0.2],
            "seasonality": [1],
            "horizon": [1],
            "n_train": [5],
            "mean": [1.0],
            "std": [0.1],
            "cv": [0.1],
            "trend_corr": [0.0],
            "seasonal_corr": [0.0],
            "abs_diff_mean": [0.1],
        }
    )
    series_meta = residuals.drop(columns=["model", "step", "y_true", "y_pred", "residual"])
    paths = residual_paths("demo", base=tmp_path)
    write_residual_dataset(
        paths,
        residuals=residuals,
        series_meta=series_meta,
        manifest={"name": "demo", "n_rows": 1},
    )
    assert paths.residuals.exists()
    assert paths.series_meta.exists()
    assert paths.manifest.exists()

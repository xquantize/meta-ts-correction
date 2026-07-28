from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from meta_ts.data.m4 import M4Series, load_m4_group
from meta_ts.metrics.mase import mase
from meta_ts.metrics.smape import smape
from meta_ts.results.fingerprint import forecast_fingerprint
from meta_ts.results.manifest import init_run, mark_completed, mark_failed
from meta_ts.results.paths import cache_paths
from meta_ts.results.store import (
    read_forecast_cache,
    summarize_scores,
    write_forecast_cache,
    write_run_artifacts,
)

_METRICS = {
    "mase": lambda item, pred: mase(item.test, pred, item.train, seasonality=item.seasonality),
    "smape": lambda item, pred: smape(item.test, pred),
}

PredictBatch = Callable[[list[M4Series]], dict[str, np.ndarray]]


def run_m4_point_forecast(
    config_path: str,
    *,
    predict_batch: PredictBatch,
    fingerprint_extras: dict[str, Any] | None = None,
    base: str = "outputs",
    data_dir: str = "data/raw",
    use_cache: bool = True,
) -> str:
    manifest, paths = init_run(config_path, base=base)
    try:
        cfg = manifest.config
        group = cfg["dataset"]["group"]
        model_name = str(cfg["model"])
        metrics = list(cfg.get("metrics", ["mase"]))
        series = load_m4_group(group, directory=data_dir)
        seasonality = series[0].seasonality
        horizon = series[0].horizon

        fp = forecast_fingerprint(
            model=model_name,
            dataset="m4",
            group=group,
            horizon=horizon,
            seasonality=seasonality,
            extras=fingerprint_extras or {},
        )
        cache = cache_paths(model_name, f"m4_{group.lower()}", fp, base=base)

        if use_cache and cache.forecasts.exists():
            forecasts, _ = read_forecast_cache(cache)
        else:
            preds = predict_batch(series)
            forecasts = predictions_to_frame(series, preds, model_name)
            write_forecast_cache(
                cache,
                forecasts=forecasts,
                meta={
                    "model": model_name,
                    "dataset": "m4",
                    "group": group,
                    "horizon": horizon,
                    "seasonality": seasonality,
                    "fingerprint": fp,
                    "n_series": len(series),
                    "extras": fingerprint_extras or {},
                },
            )

        scores = score_forecasts(series, forecasts, model_name, metrics)
        summary = summarize_scores(scores)
        summary["group"] = group
        summary["fingerprint"] = fp
        write_run_artifacts(paths, forecasts=forecasts, scores=scores, summary=summary)
        mark_completed(manifest, paths)
        return manifest.run_id
    except Exception as exc:
        mark_failed(manifest, paths, str(exc))
        raise


def predictions_to_frame(
    series: list[M4Series],
    preds: dict[str, np.ndarray],
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in series:
        pred = np.asarray(preds[item.series_id], dtype=float).ravel()
        if len(pred) != item.horizon:
            raise ValueError(f"{item.series_id}: expected horizon {item.horizon}, got {len(pred)}")
        for step, (y_true, y_pred) in enumerate(zip(item.test, pred, strict=True), start=1):
            rows.append(
                {
                    "series_id": item.series_id,
                    "model": model_name,
                    "step": step,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                }
            )
    return pd.DataFrame(rows)


def score_forecasts(
    series: list[M4Series],
    forecasts: pd.DataFrame,
    model_name: str,
    metrics: list[str],
) -> pd.DataFrame:
    unknown = [m for m in metrics if m not in _METRICS]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")

    by_id = {item.series_id: item for item in series}
    rows: list[dict[str, object]] = []
    for series_id, frame in forecasts.groupby("series_id", sort=False):
        item = by_id[str(series_id)]
        pred = frame.sort_values("step")["y_pred"].to_numpy(dtype=float)
        for metric in metrics:
            rows.append(
                {
                    "series_id": str(series_id),
                    "model": model_name,
                    "metric": metric,
                    "value": float(_METRICS[metric](item, pred)),
                }
            )
    return pd.DataFrame(rows)

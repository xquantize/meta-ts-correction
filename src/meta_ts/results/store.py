from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from meta_ts.results.paths import CachePaths, RunPaths

FORECAST_COLUMNS = ("series_id", "model", "step", "y_true", "y_pred")
SCORE_COLUMNS = ("series_id", "model", "metric", "value")


def save_forecasts(path: Path, frame: pd.DataFrame) -> None:
    _require_columns(frame, FORECAST_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(FORECAST_COLUMNS)].to_parquet(path, index=False)


def load_forecasts(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    _require_columns(frame, FORECAST_COLUMNS)
    return frame


def save_scores(path: Path, frame: pd.DataFrame) -> None:
    _require_columns(frame, SCORE_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(SCORE_COLUMNS)].to_parquet(path, index=False)


def load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    _require_columns(frame, SCORE_COLUMNS)
    return frame


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_run_artifacts(
    paths: RunPaths,
    *,
    forecasts: pd.DataFrame,
    scores: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    save_forecasts(paths.forecasts, forecasts)
    save_scores(paths.scores, scores)
    save_summary(paths.summary, summary)


def write_forecast_cache(
    paths: CachePaths,
    *,
    forecasts: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    paths.ensure()
    save_forecasts(paths.forecasts, forecasts)
    paths.meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_forecast_cache(paths: CachePaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    return load_forecasts(paths.forecasts), json.loads(paths.meta.read_text())


def summarize_scores(scores: pd.DataFrame) -> dict[str, Any]:
    _require_columns(scores, SCORE_COLUMNS)
    means = (
        scores.groupby(["model", "metric"], sort=True)["value"]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "n_series": int(scores["series_id"].nunique()),
        "n_rows": len(scores),
        "means": means,
    }


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

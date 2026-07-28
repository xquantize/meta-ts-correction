from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from meta_ts.data.m4 import M4Series, load_m4_group
from meta_ts.data.meta_features import series_meta_features
from meta_ts.results.paths import ResidualDatasetPaths, cache_paths, residual_paths
from meta_ts.results.store import read_forecast_cache


def build_residual_tables(
    series: list[M4Series],
    forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_id = {item.series_id: item for item in series}
    meta_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []

    for series_id, frame in forecasts.groupby("series_id", sort=False):
        sid = str(series_id)
        if sid not in by_id:
            raise KeyError(f"forecast series_id not in dataset: {sid}")
        item = by_id[sid]
        meta = series_meta_features(
            item.train,
            seasonality=item.seasonality,
            horizon=item.horizon,
        )
        meta_rows.append({"series_id": sid, **meta})

        ordered = frame.sort_values("step")
        for row in ordered.itertuples(index=False):
            y_true = float(row.y_true)
            y_pred = float(row.y_pred)
            residual_rows.append(
                {
                    "series_id": sid,
                    "model": str(row.model),
                    "step": int(row.step),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "residual": y_true - y_pred,
                    **meta,
                }
            )

    return pd.DataFrame(residual_rows), pd.DataFrame(meta_rows)


def resolve_forecast_cache(
    *,
    model: str,
    dataset_key: str,
    fingerprint: str | None,
    base: str | Path = "outputs",
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    root = Path(base) / "cache" / "forecasts" / model / dataset_key
    if not root.exists():
        raise FileNotFoundError(f"no forecast cache at {root}")

    if fingerprint is None:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"no fingerprints under {root}")
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            raise ValueError(
                f"multiple fingerprints under {root}; set source.fingerprint explicitly: {names}"
            )
        fingerprint = candidates[0].name

    paths = cache_paths(model, dataset_key, fingerprint, base=base)
    if not paths.forecasts.exists():
        raise FileNotFoundError(f"missing forecasts at {paths.forecasts}")
    forecasts, meta = read_forecast_cache(paths)
    return forecasts, meta, fingerprint


def write_residual_dataset(
    paths: ResidualDatasetPaths,
    *,
    residuals: pd.DataFrame,
    series_meta: pd.DataFrame,
    manifest: dict[str, Any],
) -> ResidualDatasetPaths:
    paths.ensure()
    residuals.to_parquet(paths.residuals, index=False)
    series_meta.to_parquet(paths.series_meta, index=False)
    paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return paths


def build_m4_residual_dataset(
    config: dict[str, Any],
    *,
    base: str = "outputs",
    data_dir: str = "data/raw",
) -> ResidualDatasetPaths:
    name = str(config["name"])
    group = str(config["dataset"]["group"])
    source = config.get("source") or {}
    model = str(source.get("model", "chronos"))
    dataset_key = str(source.get("dataset", f"m4_{group.lower()}"))
    fingerprint = source.get("fingerprint")

    forecasts, cache_meta, fingerprint = resolve_forecast_cache(
        model=model,
        dataset_key=dataset_key,
        fingerprint=fingerprint,
        base=base,
    )
    series = load_m4_group(group, directory=data_dir)
    residuals, series_meta = build_residual_tables(series, forecasts)

    paths = residual_paths(name, base=base)
    manifest = {
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "n_rows": len(residuals),
        "n_series": len(series_meta),
        "model": model,
        "dataset": dataset_key,
        "group": group,
        "fingerprint": fingerprint,
        "cache_meta": cache_meta,
        "residual_mean": float(residuals["residual"].mean()),
        "residual_mae": float(np.mean(np.abs(residuals["residual"]))),
    }
    return write_residual_dataset(
        paths,
        residuals=residuals,
        series_meta=series_meta,
        manifest=manifest,
    )


def load_residual_dataset(
    name: str, base: str | Path = "outputs"
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = residual_paths(name, base=base)
    residuals = pd.read_parquet(paths.residuals)
    series_meta = pd.read_parquet(paths.series_meta)
    manifest = json.loads(paths.manifest.read_text())
    return residuals, series_meta, manifest

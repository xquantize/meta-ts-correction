from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from meta_ts.results.fingerprint import forecast_fingerprint
from meta_ts.results.manifest import (
    RunManifest,
    config_hash,
    init_run,
    load_config,
    make_run_id,
)
from meta_ts.results.paths import cache_paths, run_paths
from meta_ts.results.store import (
    load_forecasts,
    load_scores,
    read_forecast_cache,
    summarize_scores,
    write_forecast_cache,
    write_run_artifacts,
)


def test_config_hash_is_stable():
    a = {"name": "x", "seed": 0, "model": "seasonal_naive"}
    b = {"model": "seasonal_naive", "seed": 0, "name": "x"}
    assert config_hash(a) == config_hash(b)


def test_make_run_id_format():
    when = datetime(2026, 7, 27, 12, 0, 0, tzinfo=UTC)
    run_id = make_run_id("seasonal_naive_m4", "abc123", when=when)
    assert run_id == "20260727T120000Z_seasonal_naive_m4_abc123"


def test_init_run_writes_manifest_and_config(tmp_path: Path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(yaml.safe_dump({"name": "demo", "model": "seasonal_naive", "seed": 0}))
    manifest, paths = init_run(cfg, base=tmp_path / "outputs")
    assert paths.manifest.exists()
    assert paths.config.exists()
    loaded = RunManifest.read(paths.manifest)
    assert loaded.name == "demo"
    assert loaded.status == "running"
    assert loaded.run_id == manifest.run_id


def test_store_roundtrip(tmp_path: Path):
    paths = run_paths("demo_run", base=tmp_path).ensure()
    forecasts = pd.DataFrame(
        {
            "series_id": ["H1", "H1"],
            "model": ["seasonal_naive", "seasonal_naive"],
            "step": [1, 2],
            "y_true": [1.0, 2.0],
            "y_pred": [1.1, 2.1],
        }
    )
    scores = pd.DataFrame(
        {
            "series_id": ["H1"],
            "model": ["seasonal_naive"],
            "metric": ["mase"],
            "value": [0.5],
        }
    )
    summary = summarize_scores(scores)
    write_run_artifacts(paths, forecasts=forecasts, scores=scores, summary=summary)

    assert load_forecasts(paths.forecasts).equals(forecasts)
    assert load_scores(paths.scores).equals(scores)
    assert json.loads(paths.summary.read_text())["n_series"] == 1


def test_forecast_cache_roundtrip(tmp_path: Path):
    fp = forecast_fingerprint(model="seasonal_naive", dataset="m4", group="Hourly", horizon=48)
    paths = cache_paths("seasonal_naive", "m4_hourly", fp, base=tmp_path)
    forecasts = pd.DataFrame(
        {
            "series_id": ["H1"],
            "model": ["seasonal_naive"],
            "step": [1],
            "y_true": [1.0],
            "y_pred": [1.0],
        }
    )
    write_forecast_cache(paths, forecasts=forecasts, meta={"fingerprint": fp})
    loaded, meta = read_forecast_cache(paths)
    assert loaded.equals(forecasts)
    assert meta["fingerprint"] == fp


def test_load_config_requires_name(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("model: x\n")
    with pytest.raises(ValueError, match="name"):
        load_config(cfg)

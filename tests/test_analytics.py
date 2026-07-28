from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from meta_ts.analytics.duckdb_store import DuckDBAnalytics
from meta_ts.analytics.factory import get_analytics


def _write_run(base: Path, run_id: str, *, name: str, model: str, mase: float) -> None:
    root = base / "runs" / run_id
    root.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "name": name,
        "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "config_hash": "abc",
        "git_sha": "deadbeef",
        "config": {"name": name, "model": model},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(
        {
            "series_id": ["H1", "H2"],
            "model": [model, model],
            "metric": ["mase", "mase"],
            "value": [mase, mase + 0.2],
        }
    ).to_parquet(root / "scores.parquet", index=False)
    pd.DataFrame(
        {
            "series_id": ["H1", "H1"],
            "model": [model, model],
            "step": [1, 2],
            "y_true": [1.0, 2.0],
            "y_pred": [1.1, 2.1],
        }
    ).to_parquet(root / "forecasts.parquet", index=False)


def test_duckdb_leaderboard(tmp_path: Path):
    _write_run(tmp_path, "run_a", name="naive", model="seasonal_naive", mase=1.0)
    _write_run(tmp_path, "run_b", name="chronos", model="chronos", mase=0.8)

    store = DuckDBAnalytics(base=tmp_path)
    runs = store.list_runs()
    assert set(runs["run_id"]) == {"run_a", "run_b"}

    board = store.leaderboard()
    assert list(board["model"]) == ["chronos", "seasonal_naive"]
    chronos = board.loc[board["model"] == "chronos"].iloc[0]
    assert chronos["mean"] == pytest.approx(0.9)
    assert chronos["n_series"] == 2


def test_factory_duckdb(tmp_path: Path):
    store = get_analytics("duckdb", base=tmp_path)
    assert store.list_runs().empty


def test_factory_lancedb_reserved(tmp_path: Path):
    with pytest.raises(NotImplementedError):
        get_analytics("lancedb", base=tmp_path)

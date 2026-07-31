from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from meta_ts.analytics.seed_sweep import (
    aggregate_seed_rows,
    find_corrector_run,
    list_completed_manifests,
)


def _write_run(
    base: Path,
    *,
    run_id: str,
    model: str,
    seed: int,
    finished_at: str,
    with_artifacts: bool = True,
) -> None:
    root = base / "runs" / run_id
    root.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "name": f"{model}_seed{seed}",
        "created_at": finished_at,
        "finished_at": finished_at,
        "status": "completed",
        "config_hash": "abc",
        "config": {"name": f"{model}_seed{seed}", "model": model, "seed": seed},
        "git_sha": None,
        "git_dirty": False,
        "python_version": "3.12",
        "package_version": "0.1.0",
        "error": None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    if with_artifacts:
        pd.DataFrame(
            {
                "series_id": ["H1"],
                "model": ["chronos"],
                "metric": ["mase"],
                "value": [1.0],
            }
        ).to_parquet(root / "scores.parquet", index=False)
        (root / "model.pt").write_bytes(b"fake")


def test_find_corrector_run_prefers_latest_complete(tmp_path: Path):
    _write_run(
        tmp_path,
        run_id="old",
        model="corrector_v1",
        seed=1,
        finished_at="2026-01-01T00:00:00+00:00",
    )
    _write_run(
        tmp_path,
        run_id="new",
        model="corrector_v1",
        seed=1,
        finished_at="2026-01-02T00:00:00+00:00",
    )
    _write_run(
        tmp_path,
        run_id="other",
        model="corrector_v2",
        seed=1,
        finished_at="2026-01-03T00:00:00+00:00",
    )
    found = find_corrector_run(model="corrector_v1", seed=1, base=tmp_path)
    assert found is not None
    assert found.run_id == "new"


def test_list_completed_skips_incomplete_artifacts(tmp_path: Path):
    _write_run(
        tmp_path,
        run_id="incomplete",
        model="corrector_v1",
        seed=0,
        finished_at="2026-01-01T00:00:00+00:00",
        with_artifacts=False,
    )
    assert list_completed_manifests(tmp_path) == []


def test_aggregate_seed_rows_counts_and_margins():
    rows = [
        {
            "seed": 0,
            "model": "corrector_v2",
            "selected_rule": "abs_diff_mean_high_q0.6",
            "selected_feature": "abs_diff_mean",
            "beats_base_on_test": True,
            "significant_on_test": True,
            "test_margin": 0.02,
            "test_selective_mean": 1.05,
            "test_base_mean": 1.07,
        },
        {
            "seed": 1,
            "model": "corrector_v2",
            "selected_rule": "never",
            "selected_feature": None,
            "beats_base_on_test": False,
            "significant_on_test": False,
            "test_margin": 0.0,
            "test_selective_mean": 1.07,
            "test_base_mean": 1.07,
        },
        {
            "seed": 0,
            "model": "corrector_v1",
            "selected_rule": "seasonal_corr_high_q0.5",
            "selected_feature": "seasonal_corr",
            "beats_base_on_test": True,
            "significant_on_test": False,
            "test_margin": 0.001,
            "test_selective_mean": 1.069,
            "test_base_mean": 1.07,
        },
    ]
    agg = aggregate_seed_rows(rows)
    v2 = agg["models"]["corrector_v2"]
    assert v2["n_seeds"] == 2
    assert v2["n_beats_base"] == 1
    assert v2["n_significant"] == 1
    assert v2["frac_beats_base"] == pytest.approx(0.5)
    assert v2["feature_counts"] == {"abs_diff_mean": 1}
    assert "never" in v2["rule_counts"]
    v1 = agg["models"]["corrector_v1"]
    assert v1["n_seeds"] == 1
    assert v1["feature_counts"] == {"seasonal_corr": 1}

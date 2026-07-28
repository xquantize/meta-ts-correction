from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from meta_ts.analytics.when_it_helps import (
    analyze_run,
    assign_quantile_bins,
    per_series_deltas,
    run_when_it_helps,
    summarize_feature_strata,
)


def test_per_series_deltas_and_helped():
    scores = pd.DataFrame(
        {
            "series_id": ["A", "A", "B", "B"],
            "model": ["chronos", "chronos_corrector_v1", "chronos", "chronos_corrector_v1"],
            "metric": ["mase", "mase", "mase", "mase"],
            "value": [2.0, 1.0, 1.0, 1.5],
        }
    )
    out = per_series_deltas(
        scores,
        base_model="chronos",
        corrected_model="chronos_corrector_v1",
        metric="mase",
    )
    assert list(out["series_id"]) == ["A", "B"]
    assert out.loc[out["series_id"] == "A", "delta"].iloc[0] == pytest.approx(1.0)
    assert bool(out.loc[out["series_id"] == "A", "helped"].iloc[0]) is True
    assert bool(out.loc[out["series_id"] == "B", "helped"].iloc[0]) is False


def test_quantile_bins_and_strata_summary():
    series = pd.DataFrame(
        {
            "series_id": [f"S{i}" for i in range(8)],
            "base_value": [1.0] * 8,
            "corrected_value": [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0],
            "delta": [0.5, 0.5, 0.5, 0.5, -1.0, -1.0, -1.0, -1.0],
            "helped": [True, True, True, True, False, False, False, False],
            "cv": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )
    bins = assign_quantile_bins(series["cv"], n_bins=2)
    assert set(bins["bin_id"]) == {0, 1}
    strata = summarize_feature_strata(series, feature="cv", n_bins=2)
    assert len(strata) == 2
    assert strata.loc[strata["bin_id"] == 0, "frac_helped"].iloc[0] == pytest.approx(1.0)
    assert strata.loc[strata["bin_id"] == 1, "frac_helped"].iloc[0] == pytest.approx(0.0)


def _write_fixture(base: Path) -> tuple[str, str]:
    run_id = "run_corrector_v1_fixture"
    root = base / "runs" / run_id
    root.mkdir(parents=True)
    scores = pd.DataFrame(
        {
            "series_id": ["H1", "H1", "H2", "H2", "H3", "H3", "H4", "H4"],
            "model": ["chronos", "chronos_corrector_v1"] * 4,
            "metric": ["mase"] * 8,
            "value": [2.0, 1.0, 2.0, 1.5, 1.0, 1.2, 3.0, 2.5],
        }
    )
    scores.to_parquet(root / "scores.parquet", index=False)
    (root / "splits.json").write_text(
        json.dumps({"train_ids": [], "val_ids": [], "test_ids": ["H1", "H2", "H3", "H4"], "seed": 0})
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "variant": "corrector_v1",
                "residuals": {"name": "residuals_fixture", "model": "chronos"},
            }
        )
    )
    (root / "config.yaml").write_text("residuals: residuals_fixture\nmodel: corrector_v1\n")

    res = base / "datasets" / "residuals" / "residuals_fixture"
    res.mkdir(parents=True)
    pd.DataFrame(
        {
            "series_id": ["H1", "H2", "H3", "H4"],
            "cv": [0.1, 0.2, 0.8, 0.9],
            "trend_corr": [0.0, 0.1, 0.5, 0.6],
            "seasonal_corr": [0.9, 0.8, 0.2, 0.1],
            "abs_diff_mean": [1.0, 2.0, 10.0, 11.0],
            "n_train": [100, 120, 800, 900],
            "seasonality": [24, 24, 24, 24],
            "horizon": [48, 48, 48, 48],
            "mean": [1.0, 1.0, 1.0, 1.0],
            "std": [0.1, 0.2, 0.8, 0.9],
        }
    ).to_parquet(res / "series_meta.parquet", index=False)
    return run_id, "residuals_fixture"


def test_analyze_and_write(tmp_path: Path):
    run_id, res_name = _write_fixture(tmp_path)
    series, strata, meta = analyze_run(
        run_id=run_id,
        residuals_name=res_name,
        metric="mase",
        strata=[{"feature": "cv", "n_bins": 2}, {"feature": "base_mase", "n_bins": 2}],
        split="test",
        base=tmp_path,
    )
    assert meta["n_series"] == 4
    assert 0.0 < meta["frac_helped"] < 1.0
    assert set(strata["feature"]) == {"cv", "base_mase"}

    out = run_when_it_helps(
        {
            "name": "when_fixture",
            "metric": "mase",
            "residuals": res_name,
            "runs": [{"run_id": run_id}],
            "strata": [{"feature": "cv", "n_bins": 2}],
        },
        base=tmp_path,
    )
    assert (out / "series.parquet").exists()
    assert (out / "strata.parquet").exists()
    assert (out / "summary.json").exists()

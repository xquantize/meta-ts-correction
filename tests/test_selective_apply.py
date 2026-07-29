from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from meta_ts.analytics.selective_apply import (
    analyze_selective_run,
    apply_rule,
    fit_threshold,
    run_selective_apply,
    selective_values,
    summarize_policy,
)


def test_threshold_and_rule():
    meta = pd.DataFrame(
        {
            "series_id": ["a", "b", "c", "d"],
            "abs_diff_mean": [1.0, 2.0, 3.0, 4.0],
        }
    )
    thr = fit_threshold(
        meta, series_ids=["a", "b", "c", "d"], feature="abs_diff_mean", quantile=0.75
    )
    assert thr == pytest.approx(3.25)
    apply = apply_rule(pd.Series([1.0, 3.25, 4.0]), threshold=thr, direction="high")
    assert list(apply) == [False, True, True]


def test_selective_values_and_summary():
    base = pd.Series([2.0, 2.0, 2.0, 2.0])
    corr = pd.Series([1.0, 3.0, 1.0, 3.0])
    apply = pd.Series([True, False, True, False])
    sel = selective_values(base, corr, apply)
    assert list(sel) == [1.0, 2.0, 1.0, 2.0]
    summary = summarize_policy(base, sel, apply=apply)
    assert summary["frac_applied"] == pytest.approx(0.5)
    assert summary["policy_mean"] == pytest.approx(1.5)
    assert summary["delta_mean"] == pytest.approx(0.5)


def _write_fixture(base: Path) -> str:
    run_id = "run_selective_fixture"
    root = base / "runs" / run_id
    root.mkdir(parents=True)
    # train: T1 T2; test: H1..H4
    scores = pd.DataFrame(
        {
            "series_id": ["H1", "H1", "H2", "H2", "H3", "H3", "H4", "H4"],
            "model": ["chronos", "chronos_corrector_v1"] * 4,
            "metric": ["mase"] * 8,
            # H1/H2: correction helps; H3/H4: correction hurts
            "value": [2.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 3.0],
        }
    )
    scores.to_parquet(root / "scores.parquet", index=False)
    (root / "splits.json").write_text(
        json.dumps(
            {
                "train_ids": ["T1", "T2", "T3", "T4"],
                "val_ids": [],
                "test_ids": ["H1", "H2", "H3", "H4"],
                "seed": 0,
            }
        )
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "variant": "corrector_v1",
                "residuals": {"name": "residuals_sel", "model": "chronos"},
            }
        )
    )
    (root / "config.yaml").write_text("residuals: residuals_sel\n")

    res = base / "datasets" / "residuals" / "residuals_sel"
    res.mkdir(parents=True)
    # High abs_diff on H1/H2 (should apply); low on H3/H4 (abstain)
    pd.DataFrame(
        {
            "series_id": ["T1", "T2", "T3", "T4", "H1", "H2", "H3", "H4"],
            "abs_diff_mean": [1.0, 2.0, 3.0, 4.0, 3.5, 4.0, 1.0, 1.5],
        }
    ).to_parquet(res / "series_meta.parquet", index=False)
    return run_id


def test_analyze_and_write(tmp_path: Path):
    run_id = _write_fixture(tmp_path)
    series, summary = analyze_selective_run(
        run_id=run_id,
        feature="abs_diff_mean",
        quantile=0.5,
        direction="high",
        fit_on="train",
        metric="mase",
        residuals_name="residuals_sel",
        base=tmp_path,
    )
    assert summary["n_test"] == 4
    # train median of [1,2,3,4] = 2.5 → apply H1,H2 only
    assert summary["rule"]["threshold"] == pytest.approx(2.5)
    assert summary["selective"]["frac_applied"] == pytest.approx(0.5)
    # selective uses corr on H1/H2 (1.0) and base on H3/H4 (1.0) → mean 1.0
    assert summary["selective"]["policy_mean"] == pytest.approx(1.0)
    assert summary["always_corrected"]["policy_mean"] == pytest.approx(2.0)
    assert summary["worth_gate"] is True
    assert set(series.loc[series["apply"], "series_id"]) == {"H1", "H2"}

    out = run_selective_apply(
        {
            "name": "sel_fixture",
            "residuals": "residuals_sel",
            "metric": "mase",
            "rule": {
                "feature": "abs_diff_mean",
                "quantile": 0.5,
                "direction": "high",
                "fit_on": "train",
            },
            "runs": [{"run_id": run_id}],
        },
        base=tmp_path,
    )
    assert (out / "summary.json").exists()
    assert (out / "comparison.csv").exists()

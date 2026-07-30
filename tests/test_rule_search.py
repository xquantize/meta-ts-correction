from __future__ import annotations

import pandas as pd
import pytest

from meta_ts.analytics.rule_search import (
    candidate_rules,
    evaluate_candidate,
    is_never,
    merge_meta_features,
    rule_label,
    search_rules,
)


def _val_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    # V1/V2 have high abs_diff_mean and the correction helps there;
    # V3/V4 are smooth and the correction hurts badly.
    series = pd.DataFrame(
        {
            "series_id": ["V1", "V2", "V3", "V4"],
            "base_value": [2.0, 2.0, 1.0, 1.0],
            "corrected_value": [1.0, 1.0, 5.0, 5.0],
        }
    )
    meta = pd.DataFrame(
        {
            "series_id": ["T1", "T2", "T3", "T4", "V1", "V2", "V3", "V4"],
            "abs_diff_mean": [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 0.5, 0.7],
            "cv": [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4],
        }
    )
    return series, meta


def test_candidate_rules_grid_includes_never():
    rules = candidate_rules(["cv"], [0.5, 0.9], ["high", "low"])
    assert sum(1 for r in rules if is_never(r)) == 1
    assert len(rules) == 5
    assert rule_label(rules[0]) == "never"
    assert "cv_high_q0.5" in {rule_label(r) for r in rules}


def test_never_rule_scores_as_base():
    series, meta = _val_fixture()
    row = evaluate_candidate(
        series,
        rule={"policy": "never"},
        meta=meta,
        fit_ids=["T1", "T2", "T3", "T4"],
    )
    assert row["frac_applied"] == pytest.approx(0.0)
    assert row["policy_mean"] == pytest.approx(1.5)
    assert row["threshold"] is None


def test_search_picks_rule_that_targets_helpful_series():
    series, meta = _val_fixture()
    series = merge_meta_features(series, meta, ["abs_diff_mean", "cv"])
    ranked, best = search_rules(
        series,
        meta=meta,
        fit_ids=["T1", "T2", "T3", "T4"],
        candidates=candidate_rules(["abs_diff_mean"], [0.5, 0.9], ["high", "low"]),
    )
    assert best["feature"] == "abs_diff_mean"
    assert best["direction"] == "high"
    # applies to V1/V2 only → (1.0 + 1.0 + 1.0 + 1.0) / 4
    assert ranked.loc[0, "policy_mean"] == pytest.approx(1.0)
    assert ranked.loc[0, "frac_applied"] == pytest.approx(0.5)


def test_search_falls_back_to_never_when_correction_only_hurts():
    series = pd.DataFrame(
        {
            "series_id": ["V1", "V2"],
            "base_value": [1.0, 1.0],
            "corrected_value": [3.0, 4.0],
            "cv": [0.1, 0.9],
        }
    )
    meta = pd.DataFrame(
        {
            "series_id": ["T1", "T2", "V1", "V2"],
            "cv": [0.2, 0.8, 0.1, 0.9],
        }
    )
    ranked, best = search_rules(
        series,
        meta=meta,
        fit_ids=["T1", "T2"],
        candidates=candidate_rules(["cv"], [0.5], ["high", "low"]),
    )
    assert is_never(best)
    assert ranked.loc[0, "policy_mean"] == pytest.approx(1.0)


def test_merge_meta_features_rejects_unknown_feature():
    series, meta = _val_fixture()
    with pytest.raises(KeyError):
        merge_meta_features(series, meta, ["not_a_feature"])

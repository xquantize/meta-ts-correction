from __future__ import annotations

import pandas as pd
import pytest

from meta_ts.corrector.gate import (
    DEFAULT_GATE_FEATURE,
    DEFAULT_GATE_QUANTILE,
    HardGate,
    apply_rule,
    fit_threshold,
)


def _meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["T1", "T2", "T3", "T4", "H1", "H2"],
            "abs_diff_mean": [1.0, 2.0, 3.0, 4.0, 3.5, 0.5],
            "cv": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def test_fit_threshold_on_train_ids_only():
    thr = fit_threshold(
        _meta(),
        series_ids=["T1", "T2", "T3", "T4"],
        feature="abs_diff_mean",
        quantile=0.75,
    )
    assert thr == pytest.approx(3.25)


def test_apply_rule_high_and_low():
    values = pd.Series([1.0, 3.25, 4.0])
    assert list(apply_rule(values, threshold=3.25, direction="high")) == [False, True, True]
    assert list(apply_rule(values, threshold=3.25, direction="low")) == [True, True, False]
    assert apply_rule(4.0, threshold=3.25, direction="high") is True
    assert apply_rule(1.0, threshold=3.25, direction="high") is False


def test_hard_gate_fit_defaults_match_v3_story():
    gate = HardGate.fit(_meta(), series_ids=["T1", "T2", "T3", "T4"])
    assert gate.feature == DEFAULT_GATE_FEATURE
    assert gate.quantile == DEFAULT_GATE_QUANTILE
    assert gate.direction == "high"
    # train abs_diff_mean [1,2,3,4], q=0.6 → 2.8
    assert gate.threshold == pytest.approx(2.8)
    assert list(gate.should_apply(_meta().loc[4:, "abs_diff_mean"])) == [True, False]


def test_hard_gate_roundtrip_dict():
    gate = HardGate.fit(
        _meta(),
        series_ids=["T1", "T2", "T3", "T4"],
        feature="cv",
        quantile=0.5,
        direction="low",
    )
    restored = HardGate.from_dict(gate.to_dict())
    assert restored == gate


def test_fit_threshold_rejects_bad_inputs():
    meta = _meta()
    with pytest.raises(ValueError, match="quantile"):
        fit_threshold(meta, series_ids=["T1"], feature="abs_diff_mean", quantile=1.5)
    with pytest.raises(KeyError, match="nope"):
        fit_threshold(meta, series_ids=["T1"], feature="nope", quantile=0.5)
    with pytest.raises(ValueError, match="no series"):
        fit_threshold(meta, series_ids=["Z9"], feature="abs_diff_mean", quantile=0.5)
    with pytest.raises(ValueError, match="direction"):
        apply_rule(1.0, threshold=0.0, direction="sideways")

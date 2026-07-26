from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from meta_ts.stats import (
    compare_pairwise,
    diebold_mariano,
    holm_adjust,
    wilcoxon_signed_rank,
)


def test_dm_identical_losses_are_non_significant():
    loss = np.array([1.0, 2.0, 1.5, 2.5, 1.2])
    result = diebold_mariano(loss, loss, horizon=1)
    assert result.statistic == 0.0
    assert result.pvalue == 1.0
    assert result.mean_diff == 0.0


def test_dm_constant_gap_is_treated_as_degenerate():
    loss_a = np.array([1.0, 2.0, 1.5, 2.5])
    loss_b = loss_a - 0.25
    result = diebold_mariano(loss_a, loss_b, horizon=1)
    assert result.statistic == 0.0
    assert result.pvalue == 1.0
    assert result.mean_diff == pytest.approx(0.25)


def test_dm_hand_calculation_n2():
    loss_a = np.array([2.0, 5.0])
    loss_b = np.array([1.0, 2.0])
    result = diebold_mariano(
        loss_a,
        loss_b,
        horizon=1,
        alternative="two-sided",
        harvey_correction=True,
    )
    assert result.mean_diff == pytest.approx(2.0)
    assert result.statistic == pytest.approx(2.0)
    assert result.pvalue == pytest.approx(2.0 * stats.norm.sf(2.0))


def test_dm_worse_model_has_positive_mean_diff():
    rng = np.random.default_rng(0)
    good = rng.normal(0.0, 1.0, size=200) ** 2
    bad = good + 0.5 + rng.normal(0.0, 0.05, size=200)
    result = diebold_mariano(bad, good, horizon=1, alternative="greater")
    assert result.mean_diff > 0.0
    assert result.pvalue < 0.01


def test_holm_matches_canonical_example():
    p = np.array([0.01, 0.04, 0.03])
    adjusted = holm_adjust(p)
    assert adjusted == pytest.approx(np.array([0.03, 0.06, 0.06]))


def test_holm_rejects_invalid_pvalues():
    with pytest.raises(ValueError):
        holm_adjust(np.array([0.1, 1.2]))


def test_wilcoxon_identical_scores():
    scores = np.array([1.0, 2.0, 1.5, 0.8])
    result = wilcoxon_signed_rank(scores, scores)
    assert result.pvalue == 1.0
    assert result.n == 0


def test_wilcoxon_detects_shift():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = a - 0.5
    result = wilcoxon_signed_rank(a, b, alternative="greater")
    assert result.pvalue < 0.05


def test_compare_pairwise_applies_holm():
    scores = {
        "naive": np.array([1.0, 1.2, 1.1, 0.9, 1.3, 1.0]),
        "arima": np.array([0.8, 0.9, 0.85, 0.7, 1.0, 0.75]),
        "chronos": np.array([0.7, 0.8, 0.75, 0.65, 0.9, 0.7]),
    }
    results = compare_pairwise(scores)
    assert len(results) == 3
    assert all(r.pvalue_holm >= r.pvalue - 1e-12 for r in results)
    assert {(r.model_a, r.model_b) for r in results} == {
        ("naive", "arima"),
        ("naive", "chronos"),
        ("arima", "chronos"),
    }

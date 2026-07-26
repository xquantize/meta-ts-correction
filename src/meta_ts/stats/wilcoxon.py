from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    pvalue: float
    n: int
    alternative: str


def wilcoxon_signed_rank(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be two-sided|less|greater")

    diff = a - b
    n = int(np.sum(diff != 0.0))
    if n == 0:
        return WilcoxonResult(statistic=0.0, pvalue=1.0, n=0, alternative=alternative)

    result = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return WilcoxonResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        n=n,
        alternative=alternative,
    )

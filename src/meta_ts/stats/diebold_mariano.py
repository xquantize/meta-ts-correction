from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    statistic: float
    pvalue: float
    mean_diff: float
    n: int
    horizon: int
    harvey_correction: bool


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    alternative: str = "two-sided",
    harvey_correction: bool = True,
) -> DMResult:
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be two-sided|less|greater")

    d = a - b
    n = len(d)
    if n < 2:
        raise ValueError("need at least 2 observations")

    mean_diff = float(np.mean(d))
    variance = _newey_west_variance(d, lags=horizon - 1)
    if variance <= np.finfo(float).eps * max(1.0, mean_diff * mean_diff):
        return DMResult(
            statistic=0.0,
            pvalue=1.0,
            mean_diff=mean_diff,
            n=n,
            horizon=horizon,
            harvey_correction=harvey_correction,
        )

    statistic = mean_diff / np.sqrt(variance / n)
    if harvey_correction:
        factor = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        if factor <= 0.0:
            raise ValueError("Harvey correction factor is non-positive; reduce horizon")
        statistic = statistic * np.sqrt(factor)

    pvalue = float(_normal_pvalue(statistic, alternative))
    return DMResult(
        statistic=float(statistic),
        pvalue=pvalue,
        mean_diff=mean_diff,
        n=n,
        horizon=horizon,
        harvey_correction=harvey_correction,
    )


def _newey_west_variance(d: np.ndarray, *, lags: int) -> float:
    d = d - np.mean(d)
    n = len(d)
    gamma0 = float(np.dot(d, d) / n)
    variance = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        gamma = float(np.dot(d[lag:], d[:-lag]) / n)
        variance += 2.0 * weight * gamma
    return max(variance, 0.0)


def _normal_pvalue(statistic: float, alternative: str) -> float:
    if alternative == "two-sided":
        return float(2.0 * stats.norm.sf(abs(statistic)))
    if alternative == "greater":
        return float(stats.norm.sf(statistic))
    return float(stats.norm.cdf(statistic))

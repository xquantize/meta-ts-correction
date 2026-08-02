"""Hard abstain gate for residual correctors.

Fit a scalar meta-feature threshold on the train fold; at inference, apply the
residual correction only when the rule fires, otherwise keep the base forecast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

DEFAULT_GATE_FEATURE = "abs_diff_mean"
DEFAULT_GATE_QUANTILE = 0.6
DEFAULT_GATE_DIRECTION = "high"


def fit_threshold(
    meta: pd.DataFrame,
    *,
    series_ids: list[str],
    feature: str,
    quantile: float,
) -> float:
    """Fit a scalar threshold on the fit fold (typically train)."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if feature not in meta.columns:
        raise KeyError(f"feature {feature!r} not in series_meta")
    sub = meta.loc[meta["series_id"].isin(series_ids), feature].astype(float)
    if sub.empty:
        raise ValueError("no series available to fit threshold")
    return float(sub.quantile(quantile))


def apply_rule(
    values: pd.Series | float,
    *,
    threshold: float,
    direction: str,
) -> pd.Series | bool:
    """True = apply correction; False = keep the base forecast."""
    if direction not in {"high", "low"}:
        raise ValueError("direction must be 'high' or 'low'")
    if isinstance(values, pd.Series):
        x = values.astype(float)
        return x >= threshold if direction == "high" else x <= threshold
    value = float(values)
    return value >= threshold if direction == "high" else value <= threshold


@dataclass(frozen=True)
class HardGate:
    """Train-fit hard abstain rule (canonical v3 gate)."""

    feature: str
    quantile: float
    direction: str
    threshold: float

    def should_apply(self, values: pd.Series | float) -> pd.Series | bool:
        return apply_rule(values, threshold=self.threshold, direction=self.direction)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardGate:
        return cls(
            feature=str(payload["feature"]),
            quantile=float(payload["quantile"]),
            direction=str(payload["direction"]),
            threshold=float(payload["threshold"]),
        )

    @classmethod
    def fit(
        cls,
        meta: pd.DataFrame,
        *,
        series_ids: list[str],
        feature: str = DEFAULT_GATE_FEATURE,
        quantile: float = DEFAULT_GATE_QUANTILE,
        direction: str = DEFAULT_GATE_DIRECTION,
    ) -> HardGate:
        threshold = fit_threshold(
            meta,
            series_ids=series_ids,
            feature=feature,
            quantile=quantile,
        )
        # Validate direction early (apply_rule also checks).
        apply_rule(0.0, threshold=threshold, direction=direction)
        return cls(
            feature=feature,
            quantile=float(quantile),
            direction=direction,
            threshold=threshold,
        )

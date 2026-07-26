from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meta_ts.stats.diebold_mariano import DMResult, diebold_mariano
from meta_ts.stats.holm import holm_adjust
from meta_ts.stats.wilcoxon import WilcoxonResult, wilcoxon_signed_rank


@dataclass(frozen=True)
class PairwiseComparison:
    model_a: str
    model_b: str
    dm: DMResult | None
    wilcoxon: WilcoxonResult
    pvalue: float
    pvalue_holm: float


def compare_pairwise(
    scores: dict[str, np.ndarray],
    *,
    losses: dict[str, np.ndarray] | None = None,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> list[PairwiseComparison]:
    names = list(scores)
    if len(names) < 2:
        raise ValueError("need at least two models")

    pairs: list[tuple[str, str, WilcoxonResult, DMResult | None]] = []
    raw_p: list[float] = []

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            w = wilcoxon_signed_rank(scores[name_a], scores[name_b], alternative=alternative)
            dm = None
            if losses is not None:
                dm = diebold_mariano(
                    losses[name_a],
                    losses[name_b],
                    horizon=horizon,
                    alternative=alternative,
                )
            p = dm.pvalue if dm is not None else w.pvalue
            pairs.append((name_a, name_b, w, dm))
            raw_p.append(p)

    adjusted = holm_adjust(np.asarray(raw_p, dtype=float))
    out: list[PairwiseComparison] = []
    for (name_a, name_b, w, dm), p, p_holm in zip(pairs, raw_p, adjusted, strict=True):
        out.append(
            PairwiseComparison(
                model_a=name_a,
                model_b=name_b,
                dm=dm,
                wilcoxon=w,
                pvalue=float(p),
                pvalue_holm=float(p_holm),
            )
        )
    return out

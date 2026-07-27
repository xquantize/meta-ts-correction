from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from meta_ts.baselines.seasonal_naive import seasonal_naive
from meta_ts.data.m4 import M4Series, load_m4_group, load_naive2_forecasts
from meta_ts.metrics.mase import mase


@dataclass(frozen=True)
class GroupResult:
    group: str
    model: str
    mean_mase: float
    n_series: int
    reference_mase: float | None
    abs_error: float | None
    passed: bool | None


def score_group_seasonal_naive(series: list[M4Series]) -> float:
    scores = [
        mase(
            item.test,
            seasonal_naive(item.train, item.horizon, item.seasonality),
            item.train,
            seasonality=item.seasonality,
        )
        for item in series
    ]
    return float(np.mean(scores))


def score_group_naive2(series: list[M4Series], naive2: pd.DataFrame) -> float:
    by_id = naive2.set_index("id")
    scores = []
    for item in series:
        row = by_id.loc[item.series_id]
        pred = row.iloc[: item.horizon].to_numpy(dtype=float)
        scores.append(mase(item.test, pred, item.train, seasonality=item.seasonality))
    return float(np.mean(scores))


def validate_harness(
    config_path: str = "configs/harness_validation.yaml",
    directory: str = "data/raw",
) -> list[GroupResult]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    naive2 = load_naive2_forecasts(directory)
    tol = float(config["tolerance"])
    results: list[GroupResult] = []

    for entry in config["groups"]:
        group = entry["name"]
        series = load_m4_group(group, directory=directory)
        mean_mase = score_group_naive2(series, naive2)
        reference = float(entry["naive2_mase"])
        abs_error = abs(mean_mase - reference)
        results.append(
            GroupResult(
                group=group,
                model="naive2",
                mean_mase=mean_mase,
                n_series=len(series),
                reference_mase=reference,
                abs_error=abs_error,
                passed=abs_error <= tol,
            )
        )
    return results


def main() -> int:
    results = validate_harness()
    failed = False
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"{status}  {r.group:8s}  mean={r.mean_mase:.6f}  "
            f"ref={r.reference_mase:.3f}  |err|={r.abs_error:.6f}  n={r.n_series}"
        )
        failed |= not bool(r.passed)
    if failed:
        print("harness validation failed")
        return 1
    print("harness validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

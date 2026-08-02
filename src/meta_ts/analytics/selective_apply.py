from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from meta_ts.analytics.when_it_helps import (
    infer_base_model,
    infer_corrected_model,
    load_run_bundle,
    per_series_deltas,
)
from meta_ts.corrector.gate import apply_rule, fit_threshold
from meta_ts.results.paths import residual_paths, tables_root
from meta_ts.stats import wilcoxon_signed_rank


def selective_apply_paths(name: str, base: str | Path = "outputs") -> Path:
    return tables_root(base) / "selective_apply" / name


# Threshold helpers live in corrector.gate (shared with hard-gate v3).


def selective_values(
    base_values: pd.Series,
    corrected_values: pd.Series,
    apply: pd.Series,
) -> pd.Series:
    return base_values.where(~apply, corrected_values)


def summarize_policy(
    base_values: pd.Series,
    policy_values: pd.Series,
    *,
    apply: pd.Series | None = None,
) -> dict[str, Any]:
    delta = base_values.astype(float) - policy_values.astype(float)
    out: dict[str, Any] = {
        "n": len(base_values),
        "base_mean": float(base_values.mean()),
        "policy_mean": float(policy_values.mean()),
        "delta_mean": float(delta.mean()),
        "frac_helped": float((delta > 0).mean()),
    }
    if apply is not None:
        out["frac_applied"] = float(apply.mean())
        out["n_applied"] = int(apply.sum())
    if len(base_values) >= 2:
        w = wilcoxon_signed_rank(
            base_values.to_numpy(dtype=float),
            policy_values.to_numpy(dtype=float),
            alternative="greater",
        )
        out["wilcoxon"] = {
            "statistic": w.statistic,
            "pvalue": w.pvalue,
            "n": w.n,
            "alternative": w.alternative,
        }
    else:
        out["wilcoxon"] = None
    return out


def analyze_selective_run(
    *,
    run_id: str,
    feature: str,
    quantile: float,
    direction: str = "high",
    fit_on: str = "train",
    metric: str = "mase",
    residuals_name: str | None = None,
    base_model: str | None = None,
    corrected_model: str | None = None,
    base: str | Path = "outputs",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay correction with a train-fit meta threshold on the held-out test fold.

    No retraining: per-series metric for the selective policy is the corrected
    score when the rule fires, otherwise the base score.
    """
    bundle = load_run_bundle(run_id, base=base)
    summary = bundle["summary"]
    splits = bundle["splits"]
    if not splits:
        raise ValueError(f"run {run_id} is missing splits.json")

    fit_key = f"{fit_on}_ids"
    if fit_key not in splits or "test_ids" not in splits:
        raise KeyError(f"splits.json must contain {fit_key!r} and 'test_ids'")

    corr_model = corrected_model or infer_corrected_model(bundle["paths"].root, summary)
    b_model = base_model or infer_base_model(summary, corr_model)

    res_name = (
        residuals_name
        or bundle["config"].get("residuals")
        or summary.get("residuals", {}).get("name")
    )
    if not res_name:
        raise ValueError("residuals dataset name not found; set residuals in config")

    meta_path = residual_paths(res_name, base=base).series_meta
    if not meta_path.exists():
        raise FileNotFoundError(f"missing series_meta: {meta_path}")
    meta = pd.read_parquet(meta_path)

    threshold = fit_threshold(
        meta,
        series_ids=list(splits[fit_key]),
        feature=feature,
        quantile=quantile,
    )

    series = per_series_deltas(
        bundle["scores"],
        base_model=b_model,
        corrected_model=corr_model,
        metric=metric,
    )
    test_ids = set(splits["test_ids"])
    series = series.loc[series["series_id"].isin(test_ids)].copy()
    series = series.merge(meta[["series_id", feature]], on="series_id", how="left")
    if series[feature].isna().any():
        missing = series.loc[series[feature].isna(), "series_id"].tolist()
        raise ValueError(f"missing {feature} for series: {missing[:5]}")

    apply = apply_rule(series[feature], threshold=threshold, direction=direction)
    series["apply"] = apply.to_numpy()
    series["selective_value"] = selective_values(
        series["base_value"], series["corrected_value"], series["apply"]
    )
    series["selective_delta"] = series["base_value"] - series["selective_value"]
    series["selective_helped"] = series["selective_delta"] > 0
    series.insert(0, "run_id", run_id)
    series.insert(1, "corrected_model", corr_model)

    always_corr = summarize_policy(series["base_value"], series["corrected_value"])
    selective = summarize_policy(
        series["base_value"], series["selective_value"], apply=series["apply"]
    )
    run_summary = {
        "run_id": run_id,
        "residuals": res_name,
        "metric": metric,
        "base_model": b_model,
        "corrected_model": corr_model,
        "rule": {
            "feature": feature,
            "quantile": quantile,
            "direction": direction,
            "fit_on": fit_on,
            "threshold": threshold,
        },
        "n_test": len(series),
        "always_base_mean": float(series["base_value"].mean()),
        "always_corrected": always_corr,
        "selective": selective,
        "worth_gate": bool(
            selective["policy_mean"] < always_corr["policy_mean"]
            and selective["policy_mean"] <= float(series["base_value"].mean())
        ),
    }
    return series.reset_index(drop=True), run_summary


def run_selective_apply(config: dict[str, Any], *, base: str | Path = "outputs") -> Path:
    if "name" not in config:
        raise ValueError("config requires 'name'")
    rule = config.get("rule")
    if not isinstance(rule, dict) or "feature" not in rule:
        raise ValueError("config.rule must include 'feature'")

    feature = rule["feature"]
    quantile = float(rule.get("quantile", 0.75))
    direction = str(rule.get("direction", "high"))
    fit_on = str(rule.get("fit_on", "train"))
    metric = config.get("metric", "mase")
    residuals_name = config.get("residuals")
    base_model = config.get("base_model")

    run_specs = config.get("runs")
    if not run_specs:
        if "run_id" in config:
            run_specs = [{"run_id": config["run_id"]}]
        else:
            raise ValueError("config requires 'runs' or 'run_id'")

    series_parts: list[pd.DataFrame] = []
    run_summaries: list[dict[str, Any]] = []
    for spec in run_specs:
        run_id = spec["run_id"] if isinstance(spec, dict) else str(spec)
        corr = spec.get("corrected_model") if isinstance(spec, dict) else None
        b = spec.get("base_model", base_model) if isinstance(spec, dict) else base_model
        series_df, run_summary = analyze_selective_run(
            run_id=run_id,
            feature=feature,
            quantile=quantile,
            direction=direction,
            fit_on=fit_on,
            metric=metric,
            residuals_name=residuals_name,
            base_model=b,
            corrected_model=corr,
            base=base,
        )
        series_parts.append(series_df)
        run_summaries.append(run_summary)

    out_dir = selective_apply_paths(config["name"], base=base)
    out_dir.mkdir(parents=True, exist_ok=True)
    series_all = pd.concat(series_parts, ignore_index=True)
    series_all.to_parquet(out_dir / "series.parquet", index=False)
    series_all.to_csv(out_dir / "series.csv", index=False)

    summary = {
        "name": config["name"],
        "metric": metric,
        "rule": {
            "feature": feature,
            "quantile": quantile,
            "direction": direction,
            "fit_on": fit_on,
        },
        "residuals": residuals_name or run_summaries[0].get("residuals"),
        "runs": run_summaries,
        "config": config,
        "note": (
            "Threshold is fit on the train fold only; evaluation is on test. "
            "Selective score = corrected if rule fires else base (no retrain)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Compact comparison table for paper paste
    rows = []
    for r in run_summaries:
        rows.append(
            {
                "run_id": r["run_id"],
                "corrected_model": r["corrected_model"],
                "threshold": r["rule"]["threshold"],
                "always_base": r["always_base_mean"],
                "always_corrected": r["always_corrected"]["policy_mean"],
                "selective": r["selective"]["policy_mean"],
                "frac_applied": r["selective"]["frac_applied"],
                "delta_vs_base": r["selective"]["delta_mean"],
                "wilcoxon_p": (r["selective"].get("wilcoxon") or {}).get("pvalue"),
                "worth_gate": r["worth_gate"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    return out_dir

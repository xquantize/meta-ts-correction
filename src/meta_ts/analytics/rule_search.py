from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from meta_ts.analytics.fold_scores import score_fold
from meta_ts.analytics.selective_apply import (
    analyze_selective_run,
    apply_rule,
    fit_threshold,
    selective_values,
    summarize_policy,
)
from meta_ts.analytics.when_it_helps import (
    infer_base_model,
    infer_corrected_model,
    load_run_bundle,
    per_series_deltas,
)
from meta_ts.results.paths import residual_paths, tables_root

DEFAULT_FEATURES = ("abs_diff_mean", "cv", "seasonal_corr", "trend_corr", "n_train")
DEFAULT_QUANTILES = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
DEFAULT_DIRECTIONS = ("high", "low")

NEVER_RULE: dict[str, Any] = {"policy": "never"}


def rule_search_paths(name: str, base: str | Path = "outputs") -> Path:
    return tables_root(base) / "rule_search" / name


def is_never(rule: dict[str, Any]) -> bool:
    return rule.get("policy") == "never"


def rule_label(rule: dict[str, Any]) -> str:
    if is_never(rule):
        return "never"
    return f"{rule['feature']}_{rule['direction']}_q{rule['quantile']:g}"


def candidate_rules(
    features: tuple[str, ...] | list[str] = DEFAULT_FEATURES,
    quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
    directions: tuple[str, ...] | list[str] = DEFAULT_DIRECTIONS,
    *,
    include_never: bool = True,
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [NEVER_RULE] if include_never else []
    for feature in features:
        for quantile in quantiles:
            for direction in directions:
                rules.append(
                    {
                        "feature": str(feature),
                        "quantile": float(quantile),
                        "direction": str(direction),
                    }
                )
    return rules


def evaluate_candidate(
    series: pd.DataFrame,
    *,
    rule: dict[str, Any],
    meta: pd.DataFrame,
    fit_ids: list[str],
) -> dict[str, Any]:
    """Score one candidate rule on the given fold. Threshold comes from fit_ids only."""
    base_values = series["base_value"]
    if is_never(rule):
        policy = base_values.copy()
        apply = pd.Series(False, index=series.index)
        threshold = None
    else:
        feature = rule["feature"]
        if feature not in series.columns:
            raise KeyError(f"feature {feature!r} not merged into series")
        threshold = fit_threshold(
            meta,
            series_ids=fit_ids,
            feature=feature,
            quantile=float(rule["quantile"]),
        )
        apply = apply_rule(series[feature], threshold=threshold, direction=rule["direction"])
        policy = selective_values(base_values, series["corrected_value"], apply)

    summary = summarize_policy(base_values, policy, apply=apply)
    return {
        "rule": rule_label(rule),
        "feature": None if is_never(rule) else rule["feature"],
        "quantile": None if is_never(rule) else float(rule["quantile"]),
        "direction": None if is_never(rule) else rule["direction"],
        "threshold": threshold,
        "n": summary["n"],
        "frac_applied": summary["frac_applied"],
        "policy_mean": summary["policy_mean"],
        "base_mean": summary["base_mean"],
        "delta_mean": summary["delta_mean"],
        "frac_helped": summary["frac_helped"],
    }


def search_rules(
    series: pd.DataFrame,
    *,
    meta: pd.DataFrame,
    fit_ids: list[str],
    candidates: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank candidates on this fold; select the lowest mean metric, preferring abstention."""
    rows = [
        evaluate_candidate(series, rule=rule, meta=meta, fit_ids=fit_ids) for rule in candidates
    ]
    table = pd.DataFrame(rows)
    by_rule = {rule_label(rule): rule for rule in candidates}
    ranked = table.sort_values(
        ["policy_mean", "frac_applied", "rule"], ascending=[True, True, True]
    ).reset_index(drop=True)
    best = by_rule[str(ranked.loc[0, "rule"])]
    return ranked, best


def merge_meta_features(
    series: pd.DataFrame,
    meta: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    missing = [f for f in features if f not in meta.columns]
    if missing:
        raise KeyError(f"features not in series_meta: {missing}")
    cols = ["series_id", *dict.fromkeys(features)]
    return series.merge(meta[cols], on="series_id", how="left", validate="one_to_one")


def _test_summary_for_never(run_id: str, *, metric: str, base: str | Path) -> dict[str, Any]:
    bundle = load_run_bundle(run_id, base=base)
    corrected = infer_corrected_model(bundle["paths"].root, bundle["summary"])
    base_model = infer_base_model(bundle["summary"], corrected)
    series = per_series_deltas(
        bundle["scores"], base_model=base_model, corrected_model=corrected, metric=metric
    )
    test_ids = set(bundle["splits"]["test_ids"])
    series = series.loc[series["series_id"].isin(test_ids)]
    apply = pd.Series(False, index=series.index)
    selective = summarize_policy(series["base_value"], series["base_value"], apply=apply)
    always = summarize_policy(series["base_value"], series["corrected_value"])
    return {
        "run_id": run_id,
        "corrected_model": corrected,
        "rule": {"policy": "never", "threshold": None},
        "n_test": len(series),
        "always_base_mean": float(series["base_value"].mean()),
        "always_corrected": always,
        "selective": selective,
        "worth_gate": False,
    }


def analyze_run(
    *,
    run_id: str,
    metric: str = "mase",
    select_on: str = "val",
    features: list[str] | None = None,
    quantiles: list[float] | None = None,
    directions: list[str] | None = None,
    residuals_name: str | None = None,
    base: str | Path = "outputs",
    data_dir: str = "data/raw",
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame | None]:
    bundle = load_run_bundle(run_id, base=base)
    splits = bundle["splits"]
    if not splits:
        raise ValueError(f"run {run_id} is missing splits.json")

    corrected = infer_corrected_model(bundle["paths"].root, bundle["summary"])
    base_model = infer_base_model(bundle["summary"], corrected)
    res_name = (
        residuals_name
        or bundle["config"].get("residuals")
        or bundle["summary"].get("residuals", {}).get("name")
    )
    if not res_name:
        raise ValueError("residuals dataset name not found; set residuals in config")
    meta = pd.read_parquet(residual_paths(str(res_name), base=base).series_meta)

    select_scores = score_fold(
        run_id,
        fold=select_on,
        base=base,
        data_dir=data_dir,
        device=device,
        metrics=[metric],
    )
    select_series = per_series_deltas(
        select_scores, base_model=base_model, corrected_model=corrected, metric=metric
    )
    feature_list = list(features or DEFAULT_FEATURES)
    select_series = merge_meta_features(select_series, meta, feature_list)

    candidates = candidate_rules(
        feature_list,
        list(quantiles or DEFAULT_QUANTILES),
        list(directions or DEFAULT_DIRECTIONS),
    )
    ranked, best = search_rules(
        select_series,
        meta=meta,
        fit_ids=list(splits["train_ids"]),
        candidates=candidates,
    )
    ranked.insert(0, "run_id", run_id)
    ranked.insert(1, "corrected_model", corrected)

    if is_never(best):
        test_summary = _test_summary_for_never(run_id, metric=metric, base=base)
        test_series = None
    else:
        test_series, test_summary = analyze_selective_run(
            run_id=run_id,
            feature=best["feature"],
            quantile=float(best["quantile"]),
            direction=best["direction"],
            fit_on="train",
            metric=metric,
            residuals_name=str(res_name),
            base=base,
        )

    selected_row = ranked.loc[ranked["rule"] == rule_label(best)].iloc[0].to_dict()
    wilcoxon = test_summary["selective"].get("wilcoxon") or {}
    summary = {
        "run_id": run_id,
        "corrected_model": corrected,
        "metric": metric,
        "select_on": select_on,
        "selected_rule": {"label": rule_label(best), **best},
        "selection": {
            "n": selected_row["n"],
            "policy_mean": selected_row["policy_mean"],
            "base_mean": selected_row["base_mean"],
            "frac_applied": selected_row["frac_applied"],
            "n_candidates": len(ranked),
        },
        "test": {
            "n": test_summary["n_test"],
            "base_mean": test_summary["always_base_mean"],
            "always_corrected_mean": test_summary["always_corrected"]["policy_mean"],
            "selective_mean": test_summary["selective"]["policy_mean"],
            "frac_applied": test_summary["selective"].get("frac_applied"),
            "wilcoxon_pvalue": wilcoxon.get("pvalue"),
        },
        "beats_base_on_test": bool(
            test_summary["selective"]["policy_mean"] < test_summary["always_base_mean"]
        ),
        "significant_on_test": bool(float(wilcoxon.get("pvalue", 1.0)) < 0.05),
    }
    return ranked, summary, test_series


def run_rule_search(
    config: dict[str, Any],
    *,
    base: str | Path = "outputs",
    data_dir: str = "data/raw",
) -> Path:
    if "name" not in config:
        raise ValueError("config requires 'name'")
    metric = config.get("metric", "mase")
    select_on = config.get("select_on", "val")
    device = config.get("device", "cpu")
    residuals_name = config.get("residuals")
    search_cfg = config.get("search") or {}

    run_specs = config.get("runs") or ([{"run_id": config["run_id"]}] if "run_id" in config else [])
    if not run_specs:
        raise ValueError("config requires 'runs' or 'run_id'")

    ranked_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    run_summaries: list[dict[str, Any]] = []
    for spec in run_specs:
        run_id = spec["run_id"] if isinstance(spec, dict) else str(spec)
        ranked, summary, test_series = analyze_run(
            run_id=run_id,
            metric=metric,
            select_on=select_on,
            features=search_cfg.get("features"),
            quantiles=search_cfg.get("quantiles"),
            directions=search_cfg.get("directions"),
            residuals_name=residuals_name,
            base=base,
            data_dir=data_dir,
            device=device,
        )
        ranked_parts.append(ranked)
        if test_series is not None:
            test_parts.append(test_series)
        run_summaries.append(summary)

    out_dir = rule_search_paths(config["name"], base=base)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.concat(ranked_parts, ignore_index=True)
    candidates.to_csv(out_dir / "candidates.csv", index=False)
    candidates.to_parquet(out_dir / "candidates.parquet", index=False)
    if test_parts:
        pd.concat(test_parts, ignore_index=True).to_parquet(
            out_dir / "test_series.parquet", index=False
        )

    summary_payload = {
        "name": config["name"],
        "metric": metric,
        "select_on": select_on,
        "residuals": residuals_name,
        "runs": run_summaries,
        "config": config,
        "note": (
            "Rules are ranked on the selection fold (val) with thresholds fit on train "
            "meta-features; the winning rule is then scored once on test. The candidate "
            "set includes a 'never apply' policy."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )
    return out_dir

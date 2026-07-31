from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from meta_ts.results.paths import residual_paths, run_paths, tables_root

DEFAULT_META_STRATA = (
    "cv",
    "trend_corr",
    "seasonal_corr",
    "abs_diff_mean",
    "n_train",
)


def when_it_helps_paths(name: str, base: str | Path = "outputs") -> Path:
    return tables_root(base) / "when_it_helps" / name


def per_series_deltas(
    scores: pd.DataFrame,
    *,
    base_model: str,
    corrected_model: str,
    metric: str = "mase",
) -> pd.DataFrame:
    """One row per series: base/corrected metric and delta (base - corrected)."""
    sub = scores.loc[scores["metric"] == metric, ["series_id", "model", "value"]]
    if sub.empty:
        raise ValueError(f"no scores for metric={metric!r}")

    base = sub.loc[sub["model"] == base_model, ["series_id", "value"]].rename(
        columns={"value": "base_value"}
    )
    corr = sub.loc[sub["model"] == corrected_model, ["series_id", "value"]].rename(
        columns={"value": "corrected_value"}
    )
    if base.empty:
        raise ValueError(f"no scores for base_model={base_model!r}")
    if corr.empty:
        raise ValueError(f"no scores for corrected_model={corrected_model!r}")

    out = base.merge(corr, on="series_id", how="inner")
    out["delta"] = out["base_value"] - out["corrected_value"]
    out["helped"] = out["delta"] > 0
    out["metric"] = metric
    return out.sort_values("series_id").reset_index(drop=True)


def assign_quantile_bins(values: pd.Series, n_bins: int) -> pd.DataFrame:
    """Descriptive quantile bins on the given values (ties → duplicate edges dropped)."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    x = values.astype(float)
    if x.nunique(dropna=True) < 2:
        return pd.DataFrame(
            {
                "bin_id": np.zeros(len(x), dtype=int),
                "bin_lo": np.full(len(x), float(x.iloc[0]) if len(x) else np.nan),
                "bin_hi": np.full(len(x), float(x.iloc[0]) if len(x) else np.nan),
            },
            index=values.index,
        )

    cats, edges = pd.qcut(x, q=n_bins, retbins=True, duplicates="drop")
    # pandas Categorical codes: -1 for NaN
    codes = cats.cat.codes.to_numpy()
    edges = np.asarray(edges, dtype=float)
    lo = np.full(len(x), np.nan)
    hi = np.full(len(x), np.nan)
    valid = codes >= 0
    lo[valid] = edges[codes[valid]]
    hi[valid] = edges[codes[valid] + 1]
    return pd.DataFrame({"bin_id": codes, "bin_lo": lo, "bin_hi": hi}, index=values.index)


def summarize_feature_strata(
    series: pd.DataFrame,
    *,
    feature: str,
    n_bins: int = 4,
) -> pd.DataFrame:
    if feature not in series.columns:
        raise KeyError(f"feature {feature!r} not in series columns")
    bins = assign_quantile_bins(series[feature], n_bins)
    tagged = pd.concat([series.reset_index(drop=True), bins.reset_index(drop=True)], axis=1)
    rows: list[dict[str, Any]] = []
    for bin_id, g in tagged.groupby("bin_id", sort=True):
        if int(bin_id) < 0:
            continue
        rows.append(
            {
                "feature": feature,
                "bin_id": int(bin_id),
                "bin_lo": float(g["bin_lo"].iloc[0]),
                "bin_hi": float(g["bin_hi"].iloc[0]),
                "n": len(g),
                "mean_delta": float(g["delta"].mean()),
                "median_delta": float(g["delta"].median()),
                "frac_helped": float(g["helped"].mean()),
                "mean_base": float(g["base_value"].mean()),
                "mean_corrected": float(g["corrected_value"].mean()),
            }
        )
    return pd.DataFrame(rows)


def infer_corrected_model(run_dir: Path, summary: dict[str, Any]) -> str:
    variant = summary.get("variant")
    if isinstance(variant, str) and variant.startswith("corrector_"):
        # corrector_v1 -> chronos_corrector_v1 (base model from residuals)
        suffix = variant.removeprefix("corrector_")
        base = summary.get("residuals", {}).get("model") or "chronos"
        return f"{base}_corrector_{suffix}"

    scores_path = run_dir / "scores.parquet"
    if scores_path.exists():
        models = pd.read_parquet(scores_path, columns=["model"])["model"].unique().tolist()
        corrected = [m for m in models if "corrector" in str(m)]
        if len(corrected) == 1:
            return str(corrected[0])
    raise ValueError(f"could not infer corrected_model for {run_dir}")


def infer_base_model(summary: dict[str, Any], corrected_model: str) -> str:
    residual_model = summary.get("residuals", {}).get("model")
    if isinstance(residual_model, str) and residual_model:
        return residual_model
    if "_corrector_" in corrected_model:
        return corrected_model.split("_corrector_", 1)[0]
    raise ValueError("could not infer base_model")


def load_run_bundle(run_id: str, base: str | Path = "outputs") -> dict[str, Any]:
    paths = run_paths(run_id, base=base)
    if not paths.scores.exists():
        raise FileNotFoundError(f"missing scores: {paths.scores}")
    summary: dict[str, Any] = {}
    if paths.summary.exists():
        summary = json.loads(paths.summary.read_text())
    splits: dict[str, Any] = {}
    splits_path = paths.root / "splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
    config: dict[str, Any] = {}
    if paths.config.exists():
        with open(paths.config) as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            config = loaded
    return {
        "run_id": run_id,
        "paths": paths,
        "scores": pd.read_parquet(paths.scores),
        "summary": summary,
        "splits": splits,
        "config": config,
    }


def analyze_run(
    *,
    run_id: str,
    residuals_name: str | None = None,
    metric: str = "mase",
    base_model: str | None = None,
    corrected_model: str | None = None,
    strata: list[dict[str, Any]] | None = None,
    split: str = "test",
    base: str | Path = "outputs",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bundle = load_run_bundle(run_id, base=base)
    summary = bundle["summary"]
    scores = bundle["scores"]
    splits = bundle["splits"]

    corr_model = corrected_model or infer_corrected_model(bundle["paths"].root, summary)
    b_model = base_model or infer_base_model(summary, corr_model)

    series = per_series_deltas(
        scores, base_model=b_model, corrected_model=corr_model, metric=metric
    )

    if split and splits:
        key = f"{split}_ids"
        if key not in splits:
            raise KeyError(f"splits.json missing {key!r}")
        keep = set(splits[key])
        series = series.loc[series["series_id"].isin(keep)].reset_index(drop=True)

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
    series = series.merge(meta, on="series_id", how="left", validate="one_to_one")

    # Convenience: also allow stratifying by the base metric itself.
    series["base_mase" if metric == "mase" else f"base_{metric}"] = series["base_value"]

    if strata is None:
        strata_specs = [{"feature": f, "n_bins": 4} for f in DEFAULT_META_STRATA]
        strata_specs.append(
            {"feature": "base_mase" if metric == "mase" else f"base_{metric}", "n_bins": 4}
        )
    else:
        strata_specs = strata

    strata_frames = []
    skipped: list[str] = []
    for spec in strata_specs:
        feature = spec["feature"]
        n_bins = int(spec.get("n_bins", 4))
        if feature not in series.columns:
            skipped.append(feature)
            continue
        if series[feature].nunique(dropna=True) < 2:
            skipped.append(feature)
            continue
        part = summarize_feature_strata(series, feature=feature, n_bins=n_bins)
        if part.empty or part["bin_id"].nunique() < 2:
            skipped.append(feature)
            continue
        part.insert(0, "run_id", run_id)
        part.insert(1, "corrected_model", corr_model)
        strata_frames.append(part)

    strata_df = pd.concat(strata_frames, ignore_index=True) if strata_frames else pd.DataFrame()

    series_out = series.copy()
    series_out.insert(0, "run_id", run_id)
    series_out.insert(1, "corrected_model", corr_model)

    meta_summary = {
        "run_id": run_id,
        "residuals": res_name,
        "metric": metric,
        "base_model": b_model,
        "corrected_model": corr_model,
        "split": split,
        "n_series": len(series_out),
        "mean_delta": float(series_out["delta"].mean()) if len(series_out) else None,
        "frac_helped": float(series_out["helped"].mean()) if len(series_out) else None,
        "binning": "quantile_on_analysis_series",
        "note": (
            "Quantile edges are fit on the analysis fold (descriptive). "
            "Not a pre-registered policy."
        ),
        "skipped_features": skipped,
    }
    return series_out, strata_df, meta_summary


def run_when_it_helps(
    config: dict[str, Any],
    *,
    base: str | Path = "outputs",
) -> Path:
    if "name" not in config:
        raise ValueError("config requires 'name'")
    name = config["name"]
    metric = config.get("metric", "mase")
    split = config.get("split", "test")
    residuals_name = config.get("residuals")
    strata = config.get("strata")
    base_model = config.get("base_model")

    run_specs = config.get("runs")
    if not run_specs:
        if "run_id" in config:
            run_specs = [{"run_id": config["run_id"]}]
        else:
            raise ValueError("config requires 'runs' or 'run_id'")

    series_parts: list[pd.DataFrame] = []
    strata_parts: list[pd.DataFrame] = []
    run_summaries: list[dict[str, Any]] = []

    for spec in run_specs:
        run_id = spec["run_id"] if isinstance(spec, dict) else str(spec)
        corr = spec.get("corrected_model") if isinstance(spec, dict) else None
        b = spec.get("base_model", base_model) if isinstance(spec, dict) else base_model
        series_df, strata_df, run_summary = analyze_run(
            run_id=run_id,
            residuals_name=residuals_name,
            metric=metric,
            base_model=b,
            corrected_model=corr,
            strata=strata,
            split=split,
            base=base,
        )
        series_parts.append(series_df)
        if not strata_df.empty:
            strata_parts.append(strata_df)
        run_summaries.append(run_summary)

    out_dir = when_it_helps_paths(name, base=base)
    out_dir.mkdir(parents=True, exist_ok=True)

    series_all = pd.concat(series_parts, ignore_index=True)
    strata_all = pd.concat(strata_parts, ignore_index=True) if strata_parts else pd.DataFrame()
    series_all.to_parquet(out_dir / "series.parquet", index=False)
    strata_all.to_parquet(out_dir / "strata.parquet", index=False)
    strata_all.to_csv(out_dir / "strata.csv", index=False)

    summary = {
        "name": name,
        "metric": metric,
        "split": split,
        "residuals": residuals_name or run_summaries[0].get("residuals"),
        "binning": "quantile_on_analysis_series",
        "runs": run_summaries,
        "config": config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return out_dir

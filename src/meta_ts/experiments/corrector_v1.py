from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import torch

from meta_ts.corrector.features import FeatureScaler, resolve_feature_names
from meta_ts.corrector.split import SeriesSplit, split_series_ids
from meta_ts.corrector.train import predict_residuals, train_corrector
from meta_ts.data.m4 import load_m4_group
from meta_ts.data.residuals import load_residual_dataset
from meta_ts.metrics.mase import mase
from meta_ts.metrics.smape import smape
from meta_ts.results.manifest import init_run, mark_completed, mark_failed
from meta_ts.results.paths import RunPaths
from meta_ts.results.store import summarize_scores, write_run_artifacts
from meta_ts.stats import diebold_mariano, wilcoxon_signed_rank


def run_corrector(
    config_path: str,
    *,
    base: str = "outputs",
    data_dir: str = "data/raw",
    overrides: dict[str, Any] | None = None,
) -> str:
    manifest, paths = init_run(config_path, base=base, overrides=overrides)
    try:
        cfg = manifest.config
        variant = str(cfg.get("model", "corrector_v1"))
        residual_name = str(cfg["residuals"])
        group = str(cfg["dataset"]["group"])
        seed = int(cfg.get("seed", 0))
        device = str(cfg.get("device", "auto"))
        metrics = list(cfg.get("metrics", ["mase", "smape"]))
        train_cfg = cfg.get("train") or {}
        split_cfg = cfg.get("split") or {}
        feature_names = resolve_feature_names(cfg)
        corrected_model_name = f"chronos_{variant}"

        residuals, _series_meta, residual_manifest = load_residual_dataset(residual_name, base=base)
        split = split_series_ids(
            residuals["series_id"].astype(str).unique().tolist(),
            seed=seed,
            train_frac=float(split_cfg.get("train", 0.7)),
            val_frac=float(split_cfg.get("val", 0.15)),
            test_frac=float(split_cfg.get("test", 0.15)),
        )

        train_df = residuals[residuals["series_id"].isin(split.train_ids)]
        val_df = residuals[residuals["series_id"].isin(split.val_ids)]
        test_df = residuals[residuals["series_id"].isin(split.test_ids)]

        scaler = FeatureScaler(feature_names).fit(train_df)
        x_train = scaler.transform(train_df)
        x_val = scaler.transform(val_df)
        x_test = scaler.transform(test_df)
        y_train = train_df["residual"].to_numpy(dtype=float)
        y_val = val_df["residual"].to_numpy(dtype=float)

        result = train_corrector(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden=int(train_cfg.get("hidden", 32)),
            dropout=float(train_cfg.get("dropout", 0.0)),
            epochs=int(train_cfg.get("epochs", 40)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            lr=float(train_cfg.get("lr", 1e-3)),
            seed=seed,
            device=device,
        )

        test_resid_hat = predict_residuals(result.model, x_test, device=device)
        test_corrected = test_df.copy()
        test_corrected["residual_hat"] = test_resid_hat
        test_corrected["y_corr"] = test_corrected["y_pred"] + test_corrected["residual_hat"]

        series = {s.series_id: s for s in load_m4_group(group, directory=data_dir)}
        scores = score_base_and_corrected(
            test_corrected, series, metrics, corrected_model_name=corrected_model_name
        )
        forecasts = _forecast_frame(test_corrected, corrected_model_name=corrected_model_name)
        comparisons = _compare_models(
            scores, test_corrected, corrected_model_name=corrected_model_name
        )

        summary = summarize_scores(scores)
        summary.update(
            {
                "variant": variant,
                "features": list(feature_names),
                "n_parameters": result.model.n_parameters(),
                "best_epoch": result.best_epoch,
                "best_val_mae": result.best_val_mae,
                "split": {
                    "n_train": len(split.train_ids),
                    "n_val": len(split.val_ids),
                    "n_test": len(split.test_ids),
                    "seed": split.seed,
                    "fractions": list(split.fractions),
                },
                "residuals": residual_manifest,
                "comparisons": comparisons,
                "go_nogo": _go_nogo(comparisons),
            }
        )

        write_run_artifacts(paths, forecasts=forecasts, scores=scores, summary=summary)
        _write_tracking(
            paths,
            split=split,
            scaler=scaler,
            result=result,
            comparisons=comparisons,
        )
        mark_completed(manifest, paths)
        return manifest.run_id
    except Exception as exc:
        mark_failed(manifest, paths, str(exc))
        raise


def run_corrector_v1(
    config_path: str,
    *,
    base: str = "outputs",
    data_dir: str = "data/raw",
    overrides: dict[str, Any] | None = None,
) -> str:
    return run_corrector(config_path, base=base, data_dir=data_dir, overrides=overrides)


def run_corrector_v2(
    config_path: str,
    *,
    base: str = "outputs",
    data_dir: str = "data/raw",
    overrides: dict[str, Any] | None = None,
) -> str:
    return run_corrector(config_path, base=base, data_dir=data_dir, overrides=overrides)


def score_base_and_corrected(
    frame: pd.DataFrame,
    series: dict,
    metrics: list[str],
    *,
    corrected_model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_id, group in frame.groupby("series_id", sort=False):
        item = series[str(series_id)]
        ordered = group.sort_values("step")
        y_true = ordered["y_true"].to_numpy(dtype=float)
        y_base = ordered["y_pred"].to_numpy(dtype=float)
        y_corr = ordered["y_corr"].to_numpy(dtype=float)
        for model_name, y_pred in (
            ("chronos", y_base),
            (corrected_model_name, y_corr),
        ):
            values = {
                "mase": mase(y_true, y_pred, item.train, seasonality=item.seasonality),
                "smape": smape(y_true, y_pred),
            }
            for metric in metrics:
                rows.append(
                    {
                        "series_id": str(series_id),
                        "model": model_name,
                        "metric": metric,
                        "value": float(values[metric]),
                    }
                )
    return pd.DataFrame(rows)


def _forecast_frame(frame: pd.DataFrame, *, corrected_model_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "series_id": str(row.series_id),
                "model": "chronos",
                "step": int(row.step),
                "y_true": float(row.y_true),
                "y_pred": float(row.y_pred),
            }
        )
        rows.append(
            {
                "series_id": str(row.series_id),
                "model": corrected_model_name,
                "step": int(row.step),
                "y_true": float(row.y_true),
                "y_pred": float(row.y_corr),
            }
        )
    return pd.DataFrame(rows)


def _compare_models(
    scores: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    corrected_model_name: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in sorted(scores["metric"].unique()):
        base = scores[(scores["model"] == "chronos") & (scores["metric"] == metric)].set_index(
            "series_id"
        )["value"]
        corr = scores[
            (scores["model"] == corrected_model_name) & (scores["metric"] == metric)
        ].set_index("series_id")["value"]
        aligned = base.index.intersection(corr.index)
        w = wilcoxon_signed_rank(
            base.loc[aligned].to_numpy(),
            corr.loc[aligned].to_numpy(),
            alternative="greater",
        )
        out[metric] = {
            "base_mean": float(base.loc[aligned].mean()),
            "corrected_mean": float(corr.loc[aligned].mean()),
            "delta_mean": float(base.loc[aligned].mean() - corr.loc[aligned].mean()),
            "wilcoxon": {
                "statistic": w.statistic,
                "pvalue": w.pvalue,
                "n": w.n,
                "alternative": w.alternative,
            },
        }

    abs_base = np.abs(test_frame["y_true"] - test_frame["y_pred"]).to_numpy(dtype=float)
    abs_corr = np.abs(test_frame["y_true"] - test_frame["y_corr"]).to_numpy(dtype=float)
    dm = diebold_mariano(abs_base, abs_corr, horizon=1, alternative="greater")
    out["abs_error_dm"] = {
        "statistic": dm.statistic,
        "pvalue": dm.pvalue,
        "mean_diff": dm.mean_diff,
        "n": dm.n,
        "horizon": dm.horizon,
        "harvey_correction": dm.harvey_correction,
    }
    return out


def _go_nogo(comparisons: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    mase = comparisons.get("mase", {})
    w = mase.get("wilcoxon", {})
    improved = float(mase.get("delta_mean", 0.0)) > 0.0
    significant = float(w.get("pvalue", 1.0)) < alpha
    decision = "go" if improved and significant else "no_go"
    return {
        "decision": decision,
        "alpha": alpha,
        "mase_improved": improved,
        "mase_wilcoxon_pvalue": w.get("pvalue"),
        "rule": "go if corrected mean MASE < base and Wilcoxon(base>corrected) p < alpha",
    }


def _write_tracking(
    paths: RunPaths,
    *,
    split: SeriesSplit,
    scaler: FeatureScaler,
    result,
    comparisons: dict[str, Any],
) -> None:
    (paths.root / "splits.json").write_text(json.dumps(split.to_dict(), indent=2) + "\n")
    (paths.root / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2) + "\n")
    (paths.root / "train_log.json").write_text(
        json.dumps(
            {
                "best_epoch": result.best_epoch,
                "best_val_mae": result.best_val_mae,
                "n_parameters": result.model.n_parameters(),
                "history": result.history,
            },
            indent=2,
        )
        + "\n"
    )
    (paths.root / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    torch.save(result.model.state_dict(), paths.root / "model.pt")

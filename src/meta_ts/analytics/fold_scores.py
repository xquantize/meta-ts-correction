from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from meta_ts.analytics.when_it_helps import (
    infer_corrected_model,
    load_run_bundle,
)
from meta_ts.corrector.features import FeatureScaler, StandardScaler1D, scale_point_features
from meta_ts.corrector.model import ResidualCorrectorV1
from meta_ts.corrector.train import predict_residuals
from meta_ts.data.residuals import load_residual_dataset
from meta_ts.experiments.corrector_v1 import score_base_and_corrected

LEGACY_FEATURE_NAMES = ("y_pred", "step_frac")


@dataclass(frozen=True)
class CorrectorReplay:
    model: ResidualCorrectorV1
    feature_names: tuple[str, ...]
    transform: Callable[[pd.DataFrame], np.ndarray]
    corrected_model: str


def load_feature_transform(
    scaler_payload: dict[str, Any],
) -> tuple[tuple[str, ...], Callable[[pd.DataFrame], np.ndarray]]:
    """Support both FeatureScaler runs and the legacy y_pred-only scaler (v1)."""
    if "columns" in scaler_payload:
        scaler = FeatureScaler.from_dict(scaler_payload)
        return tuple(scaler.columns), scaler.transform

    legacy = StandardScaler1D.from_dict(scaler_payload)
    return LEGACY_FEATURE_NAMES, lambda frame: scale_point_features(frame, legacy)


def load_corrector(run_id: str, *, base: str | Path = "outputs") -> CorrectorReplay:
    bundle = load_run_bundle(run_id, base=base)
    root = bundle["paths"].root
    scaler_path = root / "scaler.json"
    model_path = root / "model.pt"
    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"run {run_id} is missing scaler.json / model.pt")

    feature_names, transform = load_feature_transform(json.loads(scaler_path.read_text()))
    state = torch.load(model_path, map_location="cpu")
    in_dim = int(state["net.0.weight"].shape[1])
    hidden = int(state["net.0.weight"].shape[0])
    if in_dim != len(feature_names):
        raise ValueError(f"checkpoint expects {in_dim} features, scaler has {len(feature_names)}")

    model = ResidualCorrectorV1(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(state)
    model.eval()
    return CorrectorReplay(
        model=model,
        feature_names=feature_names,
        transform=transform,
        corrected_model=infer_corrected_model(root, bundle["summary"]),
    )


def score_fold(
    run_id: str,
    *,
    fold: str = "val",
    base: str | Path = "outputs",
    data_dir: str = "data/raw",
    device: str = "cpu",
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Replay a finished corrector run on another fold (val/train) and score it.

    Runs only persist test-fold scores, so rule selection on val needs this.
    """
    from meta_ts.data.m4 import load_m4_group

    bundle = load_run_bundle(run_id, base=base)
    splits = bundle["splits"]
    key = f"{fold}_ids"
    if key not in splits:
        raise KeyError(f"splits.json missing {key!r}")

    cfg = bundle["config"]
    summary = bundle["summary"]
    residual_name = cfg.get("residuals") or summary.get("residuals", {}).get("name")
    if not residual_name:
        raise ValueError("residuals dataset name not found in run config/summary")
    group = str((cfg.get("dataset") or {}).get("group") or summary["residuals"]["group"])
    metric_names = metrics or list(cfg.get("metrics") or ["mase", "smape"])

    residuals, _meta, _manifest = load_residual_dataset(str(residual_name), base=base)
    fold_df = residuals[residuals["series_id"].isin(set(splits[key]))].copy()
    if fold_df.empty:
        raise ValueError(f"no residual rows for fold {fold!r}")

    replay = load_corrector(run_id, base=base)
    x = replay.transform(fold_df)
    fold_df["residual_hat"] = predict_residuals(replay.model, x, device=device)
    fold_df["y_corr"] = fold_df["y_pred"] + fold_df["residual_hat"]

    series = {s.series_id: s for s in load_m4_group(group, directory=data_dir)}
    return score_base_and_corrected(
        fold_df,
        series,
        metric_names,
        corrected_model_name=replay.corrected_model,
    )

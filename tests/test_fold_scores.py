from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from meta_ts.analytics.fold_scores import (
    LEGACY_FEATURE_NAMES,
    load_corrector,
    load_feature_transform,
)
from meta_ts.corrector.features import FeatureScaler
from meta_ts.corrector.model import ResidualCorrectorV1


def _residual_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["H1", "H1"],
            "step": [1, 2],
            "horizon": [48, 48],
            "y_pred": [10.0, 12.0],
            "n_train": [700, 700],
            "cv": [0.3, 0.3],
        }
    )


def test_legacy_scaler_transform_matches_point_features():
    names, transform = load_feature_transform({"mean": 10.0, "std": 2.0})
    assert names == LEGACY_FEATURE_NAMES
    x = transform(_residual_frame())
    assert x.shape == (2, 2)
    assert x[0, 0] == pytest.approx(0.0)
    assert x[1, 0] == pytest.approx(1.0)
    assert x[0, 1] == pytest.approx(1 / 48)


def test_feature_scaler_payload_roundtrip():
    frame = _residual_frame()
    scaler = FeatureScaler(("y_pred", "step_frac", "cv")).fit(frame)
    names, transform = load_feature_transform(scaler.to_dict())
    assert names == ("y_pred", "step_frac", "cv")
    np.testing.assert_allclose(transform(frame), scaler.transform(frame))


def _write_run(base: Path, *, in_dim: int, scaler_payload: dict) -> str:
    run_id = "run_replay_fixture"
    root = base / "runs" / run_id
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "series_id": ["H1", "H1"],
            "model": ["chronos", "chronos_corrector_v1"],
            "metric": ["mase", "mase"],
            "value": [1.0, 0.9],
        }
    ).to_parquet(root / "scores.parquet", index=False)
    (root / "splits.json").write_text(
        json.dumps({"train_ids": ["T1"], "val_ids": ["V1"], "test_ids": ["H1"], "seed": 0})
    )
    (root / "summary.json").write_text(
        json.dumps({"variant": "corrector_v1", "residuals": {"name": "res", "model": "chronos"}})
    )
    (root / "config.yaml").write_text("residuals: res\n")
    (root / "scaler.json").write_text(json.dumps(scaler_payload))
    torch.save(ResidualCorrectorV1(in_dim=in_dim, hidden=8).state_dict(), root / "model.pt")
    return run_id


def test_load_corrector_infers_shape_and_model_name(tmp_path: Path):
    run_id = _write_run(tmp_path, in_dim=2, scaler_payload={"mean": 1.0, "std": 1.0})
    replay = load_corrector(run_id, base=tmp_path)
    assert replay.feature_names == LEGACY_FEATURE_NAMES
    assert replay.corrected_model == "chronos_corrector_v1"
    x = replay.transform(_residual_frame())
    assert replay.model(torch.tensor(x, dtype=torch.float32)).shape == (2,)


def test_load_corrector_rejects_scaler_checkpoint_mismatch(tmp_path: Path):
    payload = FeatureScaler(("y_pred", "step_frac", "cv")).fit(_residual_frame()).to_dict()
    run_id = _write_run(tmp_path, in_dim=2, scaler_payload=payload)
    with pytest.raises(ValueError, match="checkpoint expects"):
        load_corrector(run_id, base=tmp_path)

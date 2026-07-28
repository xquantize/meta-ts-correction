from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from meta_ts.corrector.features import StandardScaler1D, add_point_features, scale_point_features
from meta_ts.corrector.model import ResidualCorrectorV1
from meta_ts.corrector.split import split_series_ids
from meta_ts.corrector.train import predict_residuals, train_corrector


def test_split_is_disjoint_and_complete():
    ids = [f"H{i}" for i in range(20)]
    split = split_series_ids(ids, seed=0)
    all_ids = set(split.train_ids) | set(split.val_ids) | set(split.test_ids)
    assert all_ids == set(ids)
    assert not (set(split.train_ids) & set(split.val_ids))
    assert not (set(split.train_ids) & set(split.test_ids))
    assert not (set(split.val_ids) & set(split.test_ids))


def test_corrector_forward_and_param_count():
    model = ResidualCorrectorV1(in_dim=2, hidden=8)
    x = torch.randn(4, 2)
    y = model(x)
    assert y.shape == (4,)
    assert model.n_parameters() > 0


def test_train_corrector_overfits_tiny_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    result = train_corrector(
        x[:160],
        y[:160],
        x[160:],
        y[160:],
        hidden=16,
        epochs=20,
        batch_size=32,
        device="cpu",
        seed=0,
    )
    pred = predict_residuals(result.model, x[160:], device="cpu")
    mae = float(np.mean(np.abs(pred - y[160:])))
    assert mae < 0.15


def test_scale_point_features():
    frame = pd.DataFrame(
        {
            "y_pred": [10.0, 20.0, 30.0],
            "step": [1, 2, 3],
            "horizon": [3, 3, 3],
        }
    )
    scaler = StandardScaler1D().fit(frame["y_pred"].to_numpy())
    feats = scale_point_features(frame, scaler)
    assert feats.shape == (3, 2)
    assert add_point_features(frame)["step_frac"].tolist() == [1 / 3, 2 / 3, 1.0]

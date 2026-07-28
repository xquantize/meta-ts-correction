from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch

chronos = pytest.importorskip("chronos")

from meta_ts.baselines.chronos import _median_forecast


class _FakePipeline:
    quantiles: ClassVar[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_median_forecast_uses_half_quantile():
    # shape [batch, quantiles, horizon]
    quantiles = torch.zeros(2, 9, 4)
    quantiles[:, 4, :] = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    out = _median_forecast(quantiles, _FakePipeline())
    assert len(out) == 2
    assert np.allclose(out[0], [1, 2, 3, 4])
    assert np.allclose(out[1], [5, 6, 7, 8])

from __future__ import annotations

import numpy as np


def _require_statsforecast():
    try:
        from statsforecast.models import AutoARIMA, AutoETS
    except ImportError as exc:
        raise ImportError(
            'statsforecast is required; install with: uv pip install -e ".[forecast]"'
        ) from exc
    return AutoARIMA, AutoETS


def auto_arima(
    train: np.ndarray,
    horizon: int,
    seasonality: int = 1,
) -> np.ndarray:
    AutoARIMA, _ = _require_statsforecast()
    y = np.asarray(train, dtype=float).ravel()
    season_length = max(int(seasonality), 1)
    model = AutoARIMA(season_length=season_length)
    out = model.forecast(y=y, h=horizon)
    return np.asarray(out["mean"], dtype=float).ravel()


def auto_ets(
    train: np.ndarray,
    horizon: int,
    seasonality: int = 1,
) -> np.ndarray:
    _, AutoETS = _require_statsforecast()
    y = np.asarray(train, dtype=float).ravel()
    season_length = max(int(seasonality), 1)
    model = AutoETS(season_length=season_length)
    out = model.forecast(y=y, h=horizon)
    return np.asarray(out["mean"], dtype=float).ravel()

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from meta_ts.device import get_device


def _require_chronos():
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise ImportError(
            'chronos-forecasting is required; install with: uv pip install -e ".[tsfm]"'
        ) from exc
    return BaseChronosPipeline


def _torch_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


@lru_cache(maxsize=4)
def load_pipeline(model_id: str, device: str = "auto"):
    BaseChronosPipeline = _require_chronos()
    torch_device = get_device(device)
    return BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=str(torch_device),
        torch_dtype=_torch_dtype(torch_device),
    )


def chronos_point_forecast(
    train: np.ndarray,
    horizon: int,
    seasonality: int = 1,
    *,
    model_id: str = "amazon/chronos-bolt-tiny",
    device: str = "auto",
) -> np.ndarray:
    del seasonality
    pipeline = load_pipeline(model_id, device=device)
    context = torch.tensor(np.asarray(train, dtype=np.float32).ravel())
    quantiles = pipeline.predict(context, prediction_length=horizon)
    return _median_forecast(quantiles, pipeline)[0]


def chronos_point_forecast_batch(
    trains: list[np.ndarray],
    horizon: int,
    *,
    model_id: str = "amazon/chronos-bolt-tiny",
    device: str = "auto",
    batch_size: int = 32,
) -> list[np.ndarray]:
    if not trains:
        return []
    pipeline = load_pipeline(model_id, device=device)
    out: list[np.ndarray] = []
    for start in range(0, len(trains), batch_size):
        chunk = trains[start : start + batch_size]
        contexts = [torch.tensor(np.asarray(y, dtype=np.float32).ravel()) for y in chunk]
        quantiles = pipeline.predict(contexts, prediction_length=horizon)
        out.extend(_median_forecast(quantiles, pipeline))
    return out


def _median_forecast(quantiles: torch.Tensor, pipeline) -> list[np.ndarray]:
    q_levels = list(getattr(pipeline, "quantiles", []))
    if q_levels:
        idx = int(np.argmin(np.abs(np.asarray(q_levels, dtype=float) - 0.5)))
    else:
        idx = quantiles.shape[1] // 2
    point = quantiles[:, idx, :].detach().cpu().numpy().astype(float)
    return [row.copy() for row in point]

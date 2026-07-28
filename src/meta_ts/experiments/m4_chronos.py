from __future__ import annotations

import numpy as np

from meta_ts.baselines.chronos import chronos_point_forecast_batch
from meta_ts.data.m4 import M4Series
from meta_ts.experiments.m4_forecast import run_m4_point_forecast
from meta_ts.results.manifest import load_config


def run_m4_chronos(
    config_path: str,
    *,
    base: str = "outputs",
    data_dir: str = "data/raw",
    use_cache: bool = True,
) -> str:
    cfg = load_config(config_path)
    model_id = str(cfg.get("model_id", "amazon/chronos-bolt-tiny"))
    device = str(cfg.get("device", "auto"))
    batch_size = int(cfg.get("batch_size", 32))

    def predict_batch(series: list[M4Series]) -> dict[str, np.ndarray]:
        preds = chronos_point_forecast_batch(
            [item.train for item in series],
            horizon=series[0].horizon,
            model_id=model_id,
            device=device,
            batch_size=batch_size,
        )
        return {item.series_id: pred for item, pred in zip(series, preds, strict=True)}

    return run_m4_point_forecast(
        config_path,
        predict_batch=predict_batch,
        fingerprint_extras={
            "model_id": model_id,
            "device": device,
            "batch_size": batch_size,
        },
        base=base,
        data_dir=data_dir,
        use_cache=use_cache,
    )

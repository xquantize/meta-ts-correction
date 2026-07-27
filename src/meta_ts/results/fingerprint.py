from __future__ import annotations

import hashlib
import json
from typing import Any


def forecast_fingerprint(
    *,
    model: str,
    dataset: str,
    group: str | None = None,
    horizon: int | None = None,
    seasonality: int | None = None,
    extras: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "dataset": dataset,
        "group": group,
        "horizon": horizon,
        "seasonality": seasonality,
        "extras": extras or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

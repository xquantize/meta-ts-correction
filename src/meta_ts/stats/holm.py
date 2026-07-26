from __future__ import annotations

import numpy as np


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float).ravel()
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")

    m = len(p)
    if m == 0:
        return p.copy()

    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        candidate = (m - i) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    return adjusted

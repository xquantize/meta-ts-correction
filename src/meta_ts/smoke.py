"""Minimal environment smoke check — run before any research code."""

from __future__ import annotations

import sys

import numpy as np
import torch

from meta_ts.device import device_report, get_device
from meta_ts.metrics.mase import mase


def main() -> int:
    print("=== meta-ts-correction smoke check ===")
    print(f"python: {sys.version.split()[0]}")

    report = device_report()
    for k, v in report.items():
        print(f"{k}: {v}")

    device = get_device("auto")
    x = torch.randn(4, 8, device=device)
    y = (x @ x.T).sum()
    # Force sync on MPS so failures surface here, not later
    _ = float(y.detach().cpu())
    print(f"torch matmul on {device}: ok ({float(y):.4f})")

    # Tiny hand-checked MASE sanity (same series → error 0 → MASE 0)
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_pred = y_true.copy()
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    score = mase(y_true, y_pred, y_train, seasonality=1)
    assert score == 0.0, score
    print(f"mase sanity: ok ({score})")

    print("=== all good — harness scaffolding can proceed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

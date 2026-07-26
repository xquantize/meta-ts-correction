"""Device selection for Mac (MPS) / CUDA / CPU."""

from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available torch device.

    prefer:
      - "auto": mps > cuda > cpu
      - "mps" | "cuda" | "cpu": force that backend (falls back with a warning path)
    """
    prefer = prefer.lower()
    if prefer == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if prefer == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS requested but not available on this machine/build")
        return torch.device("mps")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown prefer={prefer!r}; use auto|mps|cuda|cpu")


def device_report() -> dict[str, object]:
    """Collect a small environment report for smoke checks."""
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    return {
        "torch": torch.__version__,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cuda_available": torch.cuda.is_available(),
        "selected": str(get_device("auto")),
    }

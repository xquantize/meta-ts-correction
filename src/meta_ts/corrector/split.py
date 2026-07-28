from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SeriesSplit:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    seed: int
    fractions: tuple[float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> SeriesSplit:
        return cls(
            train_ids=list(payload["train_ids"]),
            val_ids=list(payload["val_ids"]),
            test_ids=list(payload["test_ids"]),
            seed=int(payload["seed"]),
            fractions=tuple(payload["fractions"]),
        )


def split_series_ids(
    series_ids: list[str],
    *,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SeriesSplit:
    ids = sorted({str(s) for s in series_ids})
    if len(ids) < 3:
        raise ValueError("need at least 3 series for train/val/test split")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1")

    rng = np.random.default_rng(seed)
    order = ids.copy()
    rng.shuffle(order)

    n = len(order)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError("split left no test series; adjust fractions or seed")

    train_ids = order[:n_train]
    val_ids = order[n_train : n_train + n_val]
    test_ids = order[n_train + n_val :]
    return SeriesSplit(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        seed=seed,
        fractions=(train_frac, val_frac, test_frac),
    )

from __future__ import annotations

import numpy as np
import pandas as pd

POINT_FEATURE_NAMES = ("y_pred", "step_frac")


def add_point_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "horizon" not in out.columns:
        raise ValueError("residuals frame requires 'horizon'")
    out["step_frac"] = out["step"].astype(float) / out["horizon"].astype(float)
    return out


def feature_matrix(
    frame: pd.DataFrame, columns: tuple[str, ...] = POINT_FEATURE_NAMES
) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return frame.loc[:, list(columns)].to_numpy(dtype=np.float64)


class StandardScaler1D:
    def __init__(self):
        self.mean: float = 0.0
        self.std: float = 1.0

    def fit(self, values: np.ndarray) -> StandardScaler1D:
        x = np.asarray(values, dtype=float).ravel()
        self.mean = float(np.mean(x))
        self.std = float(np.std(x))
        if self.std == 0.0:
            self.std = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        return (x - self.mean) / self.std

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> StandardScaler1D:
        obj = cls()
        obj.mean = float(payload["mean"])
        obj.std = float(payload["std"])
        return obj


def scale_point_features(
    frame: pd.DataFrame,
    scaler: StandardScaler1D,
) -> np.ndarray:
    feats = add_point_features(frame)
    y_pred = scaler.transform(feats["y_pred"].to_numpy(dtype=float))
    step_frac = feats["step_frac"].to_numpy(dtype=float)
    return np.column_stack([y_pred, step_frac])

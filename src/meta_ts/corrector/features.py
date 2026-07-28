from __future__ import annotations

import numpy as np
import pandas as pd

POINT_FEATURE_NAMES = ("y_pred", "step_frac")
META_FEATURE_NAMES = (
    "cv",
    "trend_corr",
    "seasonal_corr",
    "abs_diff_mean",
    "log_n_train",
)
V1_FEATURE_NAMES = POINT_FEATURE_NAMES
V2_FEATURE_NAMES = POINT_FEATURE_NAMES + META_FEATURE_NAMES


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "horizon" not in out.columns:
        raise ValueError("residuals frame requires 'horizon'")
    out["step_frac"] = out["step"].astype(float) / out["horizon"].astype(float)
    if "n_train" in out.columns:
        out["log_n_train"] = np.log1p(out["n_train"].astype(float))
    return out


def add_point_features(frame: pd.DataFrame) -> pd.DataFrame:
    return add_derived_features(frame)


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


class FeatureScaler:
    def __init__(self, columns: tuple[str, ...]):
        self.columns = tuple(columns)
        self.scalers = {c: StandardScaler1D() for c in self.columns}

    def fit(self, frame: pd.DataFrame) -> FeatureScaler:
        feats = add_derived_features(frame)
        missing = [c for c in self.columns if c not in feats.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        for col in self.columns:
            self.scalers[col].fit(feats[col].to_numpy(dtype=float))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        feats = add_derived_features(frame)
        cols = [self.scalers[c].transform(feats[c].to_numpy(dtype=float)) for c in self.columns]
        return np.column_stack(cols)

    def to_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "scalers": {c: s.to_dict() for c, s in self.scalers.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> FeatureScaler:
        columns = tuple(payload["columns"])
        obj = cls(columns)
        for col, stats in payload["scalers"].items():
            obj.scalers[col] = StandardScaler1D.from_dict(stats)
        return obj


def scale_point_features(
    frame: pd.DataFrame,
    scaler: StandardScaler1D,
) -> np.ndarray:
    feats = add_derived_features(frame)
    y_pred = scaler.transform(feats["y_pred"].to_numpy(dtype=float))
    step_frac = feats["step_frac"].to_numpy(dtype=float)
    return np.column_stack([y_pred, step_frac])


def resolve_feature_names(cfg: dict) -> tuple[str, ...]:
    if "features" in cfg:
        return tuple(str(x) for x in cfg["features"])
    model = str(cfg.get("model", "corrector_v1"))
    if model == "corrector_v2":
        return V2_FEATURE_NAMES
    return V1_FEATURE_NAMES

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from meta_ts.results.paths import residual_paths, runs_root


class DuckDBAnalytics:
    def __init__(self, base: str | Path = "outputs"):
        self.base = Path(base)
        self.runs = runs_root(self.base)
        self._con = duckdb.connect(database=":memory:")

    def list_runs(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        if not self.runs.exists():
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "name",
                    "model",
                    "status",
                    "created_at",
                    "finished_at",
                    "config_hash",
                    "git_sha",
                ]
            )
        for path in sorted(self.runs.iterdir()):
            manifest_path = path / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            cfg = manifest.get("config") or {}
            rows.append(
                {
                    "run_id": manifest.get("run_id", path.name),
                    "name": manifest.get("name"),
                    "model": cfg.get("model"),
                    "status": manifest.get("status"),
                    "created_at": manifest.get("created_at"),
                    "finished_at": manifest.get("finished_at"),
                    "config_hash": manifest.get("config_hash"),
                    "git_sha": manifest.get("git_sha"),
                }
            )
        return pd.DataFrame(rows)

    def load_scores(self, run_ids: list[str] | None = None) -> pd.DataFrame:
        return self._load_parquet_join("scores.parquet", run_ids)

    def load_forecasts(self, run_ids: list[str] | None = None) -> pd.DataFrame:
        return self._load_parquet_join("forecasts.parquet", run_ids)

    def leaderboard(self) -> pd.DataFrame:
        scores = self.load_scores()
        if scores.empty:
            return pd.DataFrame(columns=["run_id", "name", "model", "metric", "mean", "n_series"])
        grouped = (
            scores.groupby(["run_id", "name", "model", "metric"], dropna=False)
            .agg(mean=("value", "mean"), n_series=("series_id", "nunique"))
            .reset_index()
            .sort_values(["metric", "mean", "model"])
            .reset_index(drop=True)
        )
        return grouped

    def sql(self, query: str) -> pd.DataFrame:
        return self._con.execute(query).df()

    def load_residuals(self, name: str) -> pd.DataFrame:
        path = residual_paths(name, base=self.base).residuals
        if not path.exists():
            raise FileNotFoundError(f"residual dataset not found: {path}")
        return self._con.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()

    def residual_summary(self, name: str) -> pd.DataFrame:
        residuals = self.load_residuals(name)
        if residuals.empty:
            return pd.DataFrame()
        return (
            residuals.groupby(["model", "series_id"], dropna=False)
            .agg(
                residual_mae=("residual", lambda s: float(s.abs().mean())),
                residual_mean=("residual", "mean"),
                n_steps=("step", "count"),
            )
            .reset_index()
        )

    def _load_parquet_join(
        self,
        filename: str,
        run_ids: list[str] | None,
    ) -> pd.DataFrame:
        runs = self.list_runs()
        if run_ids is not None:
            runs = runs[runs["run_id"].isin(run_ids)]
        if runs.empty:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for run_id in runs["run_id"].tolist():
            parquet_path = self.runs / str(run_id) / filename
            if not parquet_path.exists():
                continue
            frame = self._con.execute(
                "SELECT * FROM read_parquet(?)",
                [str(parquet_path)],
            ).df()
            meta = runs.loc[runs["run_id"] == run_id].iloc[0]
            frame.insert(0, "run_id", meta["run_id"])
            frame.insert(1, "name", meta["name"])
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

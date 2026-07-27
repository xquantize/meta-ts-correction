from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def forecasts(self) -> Path:
        return self.root / "forecasts.parquet"

    @property
    def scores(self) -> Path:
        return self.root / "scores.parquet"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    def ensure(self) -> RunPaths:
        self.root.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class CachePaths:
    root: Path

    @property
    def forecasts(self) -> Path:
        return self.root / "forecasts.parquet"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    def ensure(self) -> CachePaths:
        self.root.mkdir(parents=True, exist_ok=True)
        return self


def runs_root(base: str | Path = "outputs") -> Path:
    return Path(base) / "runs"


def cache_root(base: str | Path = "outputs") -> Path:
    return Path(base) / "cache" / "forecasts"


def tables_root(base: str | Path = "outputs") -> Path:
    return Path(base) / "tables"


def figures_root(base: str | Path = "outputs") -> Path:
    return Path(base) / "figures"


def run_paths(run_id: str, base: str | Path = "outputs") -> RunPaths:
    return RunPaths(root=runs_root(base) / run_id)


def cache_paths(
    model: str,
    dataset: str,
    fingerprint: str,
    base: str | Path = "outputs",
) -> CachePaths:
    return CachePaths(root=cache_root(base) / model / dataset / fingerprint)

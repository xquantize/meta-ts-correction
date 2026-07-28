from __future__ import annotations

from pathlib import Path

from meta_ts.analytics.protocol import AnalyticsBackend


def get_analytics(backend: str = "duckdb", base: str | Path = "outputs") -> AnalyticsBackend:
    name = backend.lower().strip()
    if name == "duckdb":
        from meta_ts.analytics.duckdb_store import DuckDBAnalytics

        return DuckDBAnalytics(base=base)
    if name in {"lancedb", "lance"}:
        from meta_ts.analytics.lancedb_store import LanceAnalyticsBackend

        return LanceAnalyticsBackend(base=base)
    raise ValueError(f"unknown analytics backend: {backend!r} (use duckdb|lancedb)")

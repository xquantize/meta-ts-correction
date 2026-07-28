from __future__ import annotations

from pathlib import Path

import pandas as pd


class LanceAnalyticsBackend:
    """Reserved backend for embedding / vector retrieval (e.g. LanceDB).

    Same surface as DuckDBAnalytics via AnalyticsBackend. Switch later with:
    get_analytics(backend='lancedb').
    """

    def __init__(self, base: str | Path = "outputs"):
        self.base = Path(base)
        raise NotImplementedError(
            "LanceDB backend is reserved for embedding retrieval; "
            "use get_analytics(backend='duckdb') for run/score analytics."
        )

    def list_runs(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_scores(self, run_ids: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def load_forecasts(self, run_ids: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def leaderboard(self) -> pd.DataFrame:
        raise NotImplementedError

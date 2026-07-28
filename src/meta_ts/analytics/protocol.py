from __future__ import annotations

from typing import Protocol

import pandas as pd


class AnalyticsBackend(Protocol):
    def list_runs(self) -> pd.DataFrame:
        """Return one row per completed run with manifest metadata."""

    def load_scores(self, run_ids: list[str] | None = None) -> pd.DataFrame:
        """Return long-form scores joined with run metadata."""

    def load_forecasts(self, run_ids: list[str] | None = None) -> pd.DataFrame:
        """Return long-form forecasts joined with run metadata."""

    def leaderboard(self) -> pd.DataFrame:
        """Return mean metrics per run / model / metric."""

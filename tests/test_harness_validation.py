from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path("data/raw/m4/datasets")


@pytest.mark.skipif(
    not (DATA / "Hourly-train.csv").exists(),
    reason="M4 data not downloaded; run: python -m meta_ts.validation.harness",
)
def test_harness_matches_published_naive2_mase():
    from meta_ts.validation.harness import validate_harness

    results = validate_harness()
    assert results, "expected at least one group"
    assert all(r.passed for r in results), results

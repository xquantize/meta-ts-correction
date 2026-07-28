from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from meta_ts.analytics.when_it_helps import run_when_it_helps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meta-ts-when-it-helps",
        description="Stratify corrector Δ by series meta-features (held-out fold)",
    )
    parser.add_argument("config", type=Path, help="When-it-helps YAML config")
    parser.add_argument("--base", default="outputs", help="Output root directory")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "name" not in config:
        raise SystemExit("config must be a mapping with a 'name' field")

    out = run_when_it_helps(config, base=args.base)
    summary_path = out / "summary.json"
    print(f"wrote {out}")
    print(f"summary: {summary_path}")

    # Compact console view of strata
    strata_csv = out / "strata.csv"
    if strata_csv.exists():
        import pandas as pd

        strata = pd.read_csv(strata_csv)
        cols = [
            c
            for c in [
                "run_id",
                "corrected_model",
                "feature",
                "bin_id",
                "n",
                "mean_delta",
                "frac_helped",
            ]
            if c in strata.columns
        ]
        # Shorten run_id for readability
        view = strata[cols].copy()
        if "run_id" in view.columns:
            view["run_id"] = view["run_id"].astype(str).str.slice(0, 20)
        print(view.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

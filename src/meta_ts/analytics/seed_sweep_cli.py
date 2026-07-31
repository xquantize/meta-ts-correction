from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meta_ts.analytics.seed_sweep import run_seed_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meta-ts-seed-sweep",
        description=(
            "Train/reuse correctors across split seeds, run val-selected rule search "
            "per seed, and aggregate stability"
        ),
    )
    parser.add_argument("config", type=Path, help="Seed-sweep YAML config")
    parser.add_argument("--base", default="outputs", help="Output root directory")
    parser.add_argument("--data-dir", default="data/raw", help="Dataset root directory")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only reuse existing completed corrector runs (fail if a seed is missing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when a completed run already exists for that seed",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "name" not in config:
        raise SystemExit("config must be a mapping with a 'name' field")

    out = run_seed_sweep(
        config,
        base=args.base,
        data_dir=args.data_dir,
        skip_train=args.skip_train,
        force=args.force,
    )
    summary = json.loads((out / "summary.json").read_text())
    print(f"wrote {out}")
    for model, agg in summary["aggregation"]["models"].items():
        print(
            f"{model}: beats_base={agg['n_beats_base']}/{agg['n_seeds']} "
            f"significant={agg['n_significant']}/{agg['n_seeds']} "
            f"mean_margin={agg['mean_test_margin']:.4f}±{agg['std_test_margin']:.4f} "
            f"features={agg['feature_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

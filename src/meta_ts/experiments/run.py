from __future__ import annotations

import argparse
from pathlib import Path

from meta_ts.experiments.m4_chronos import run_m4_chronos
from meta_ts.experiments.m4_naive import run_m4_seasonal_naive
from meta_ts.results.manifest import load_config
from meta_ts.results.paths import run_paths
from meta_ts.results.store import load_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meta-ts-run", description="Run an experiment config")
    parser.add_argument("config", type=Path, help="Path to experiment YAML")
    parser.add_argument("--base", default="outputs", help="Output root directory")
    parser.add_argument("--data-dir", default="data/raw", help="Dataset root directory")
    parser.add_argument("--no-cache", action="store_true", help="Ignore forecast cache")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model = config.get("model")
    dataset = config.get("dataset", {})

    if dataset.get("name") != "m4":
        raise SystemExit(f"unsupported dataset: {dataset!r}")

    if model == "seasonal_naive":
        run_id = run_m4_seasonal_naive(
            str(args.config),
            base=args.base,
            data_dir=args.data_dir,
            use_cache=not args.no_cache,
        )
    elif model == "chronos":
        run_id = run_m4_chronos(
            str(args.config),
            base=args.base,
            data_dir=args.data_dir,
            use_cache=not args.no_cache,
        )
    else:
        raise SystemExit(f"unsupported model: {model!r}")

    summary = load_summary(run_paths(run_id, base=args.base).summary)
    print(f"run_id: {run_id}")
    print("status: completed")
    for row in summary["means"]:
        print(f"{row['model']}  {row['metric']}={row['value']:.6f}  n={summary['n_series']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

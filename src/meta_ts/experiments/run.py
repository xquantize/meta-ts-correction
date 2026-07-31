from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from meta_ts.experiments.corrector_v1 import run_corrector_v1, run_corrector_v2
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed (frozen into the run's config.yaml / manifest)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model = config.get("model")
    overrides: dict[str, Any] | None = None
    if args.seed is not None:
        overrides = {"seed": args.seed}

    if model == "corrector_v1":
        run_id = run_corrector_v1(
            str(args.config),
            base=args.base,
            data_dir=args.data_dir,
            overrides=overrides,
        )
    elif model == "corrector_v2":
        run_id = run_corrector_v2(
            str(args.config),
            base=args.base,
            data_dir=args.data_dir,
            overrides=overrides,
        )
    elif config.get("dataset", {}).get("name") != "m4":
        raise SystemExit(f"unsupported dataset: {config.get('dataset')!r}")
    elif model == "seasonal_naive":
        if overrides:
            raise SystemExit("--seed is only supported for corrector runs")
        run_id = run_m4_seasonal_naive(
            str(args.config),
            base=args.base,
            data_dir=args.data_dir,
            use_cache=not args.no_cache,
        )
    elif model == "chronos":
        if overrides:
            raise SystemExit("--seed is only supported for corrector runs")
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
    for row in summary.get("means", []):
        print(f"{row['model']}  {row['metric']}={row['value']:.6f}  n={summary['n_series']}")
    go = summary.get("go_nogo")
    if go:
        print(
            f"go/no-go: {go['decision']}  "
            f"ΔMASE={summary.get('comparisons', {}).get('mase', {}).get('delta_mean', float('nan')):.6f}  "
            f"wilcoxon_p={go.get('mase_wilcoxon_pvalue')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

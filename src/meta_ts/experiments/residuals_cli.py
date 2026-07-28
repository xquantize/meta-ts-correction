from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from meta_ts.data.residuals import build_m4_residual_dataset, load_residual_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meta-ts-residuals",
        description="Build residual datasets from cached foundation-model forecasts",
    )
    parser.add_argument("config", type=Path, help="Residual dataset YAML config")
    parser.add_argument("--base", default="outputs", help="Output root directory")
    parser.add_argument("--data-dir", default="data/raw", help="Dataset root directory")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "name" not in config:
        raise SystemExit("config must be a mapping with a 'name' field")

    paths = build_m4_residual_dataset(config, base=args.base, data_dir=args.data_dir)
    residuals, series_meta, manifest = load_residual_dataset(config["name"], base=args.base)

    print(f"name: {manifest['name']}")
    print(f"rows: {manifest['n_rows']}  series: {manifest['n_series']}")
    print(f"residual_mae: {manifest['residual_mae']:.6f}")
    print(f"wrote {paths.root}")
    print(f"meta columns: {list(series_meta.columns)}")
    print(f"residual columns: {list(residuals.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

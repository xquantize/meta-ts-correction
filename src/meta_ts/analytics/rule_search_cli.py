from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meta_ts.analytics.rule_search import run_rule_search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meta-ts-rule-search",
        description=(
            "Select a selective-apply rule on the validation fold, then score it "
            "once on test (thresholds fit on train)"
        ),
    )
    parser.add_argument("config", type=Path, help="Rule-search YAML config")
    parser.add_argument("--base", default="outputs", help="Output root directory")
    parser.add_argument("--data-dir", default="data/raw", help="Dataset root directory")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "name" not in config:
        raise SystemExit("config must be a mapping with a 'name' field")

    out = run_rule_search(config, base=args.base, data_dir=args.data_dir)
    summary = json.loads((out / "summary.json").read_text())
    print(f"wrote {out}")
    for run in summary["runs"]:
        sel = run["selection"]
        test = run["test"]
        print(
            f"{run['corrected_model']}: rule={run['selected_rule']['label']} "
            f"({sel['n_candidates']} candidates) "
            f"val={sel['policy_mean']:.4f} (base {sel['base_mean']:.4f}) | "
            f"test base={test['base_mean']:.4f} always_corr={test['always_corrected_mean']:.4f} "
            f"selective={test['selective_mean']:.4f} p={test['wilcoxon_pvalue']} "
            f"beats_base={run['beats_base_on_test']} significant={run['significant_on_test']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

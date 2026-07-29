from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meta_ts.analytics.selective_apply import run_selective_apply


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meta-ts-selective-apply",
        description=(
            "Replay a corrector with a train-fit meta threshold: "
            "apply correction only when the rule fires, else keep the base score"
        ),
    )
    parser.add_argument("config", type=Path, help="Selective-apply YAML config")
    parser.add_argument("--base", default="outputs", help="Output root directory")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "name" not in config:
        raise SystemExit("config must be a mapping with a 'name' field")

    out = run_selective_apply(config, base=args.base)
    summary = json.loads((out / "summary.json").read_text())
    print(f"wrote {out}")
    print(f"rule: {summary['rule']}")
    for run in summary["runs"]:
        sel = run["selective"]
        print(
            f"{run['corrected_model']}: "
            f"base={run['always_base_mean']:.4f} "
            f"always_corr={run['always_corrected']['policy_mean']:.4f} "
            f"selective={sel['policy_mean']:.4f} "
            f"applied={sel['frac_applied']:.2f} "
            f"wilcoxon_p={(sel.get('wilcoxon') or {}).get('pvalue')} "
            f"worth_gate={run['worth_gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

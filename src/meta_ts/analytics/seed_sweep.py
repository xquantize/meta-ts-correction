from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from meta_ts.analytics.rule_search import run_rule_search
from meta_ts.experiments.corrector_v1 import run_corrector
from meta_ts.results.manifest import RunManifest, load_config
from meta_ts.results.paths import runs_root, tables_root


def seed_sweep_paths(name: str, base: str | Path = "outputs") -> Path:
    return tables_root(base) / "seed_sweep" / name


def list_completed_manifests(base: str | Path = "outputs") -> list[RunManifest]:
    root = runs_root(base)
    if not root.exists():
        return []
    out: list[RunManifest] = []
    for path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = RunManifest.read(path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if manifest.status != "completed":
            continue
        # Prefer runs that actually finished scoring.
        run_dir = path.parent
        if not (run_dir / "scores.parquet").exists() or not (run_dir / "model.pt").exists():
            continue
        out.append(manifest)
    return out


def find_corrector_run(
    *,
    model: str,
    seed: int,
    base: str | Path = "outputs",
) -> RunManifest | None:
    """Latest completed corrector run matching model + seed."""
    matches = [
        m
        for m in list_completed_manifests(base)
        if m.config.get("model") == model and int(m.config.get("seed", 0)) == int(seed)
    ]
    if not matches:
        return None
    matches.sort(key=lambda m: m.finished_at or m.created_at)
    return matches[-1]


def ensure_corrector_run(
    config_path: str | Path,
    *,
    seed: int,
    base: str | Path = "outputs",
    data_dir: str = "data/raw",
    force: bool = False,
) -> str:
    cfg = load_config(config_path)
    model = str(cfg["model"])
    if not force:
        existing = find_corrector_run(model=model, seed=seed, base=base)
        if existing is not None:
            return existing.run_id
    return run_corrector(
        str(config_path),
        base=str(base),
        data_dir=data_dir,
        overrides={"seed": seed},
    )


def rule_search_name(sweep_name: str, model: str, seed: int) -> str:
    return f"{sweep_name}__{model}_seed{seed}"


def run_seed_rule_search(
    *,
    sweep_name: str,
    run_id: str,
    model: str,
    seed: int,
    rule_cfg: dict[str, Any],
    base: str | Path = "outputs",
    data_dir: str = "data/raw",
) -> dict[str, Any]:
    config = {
        "name": rule_search_name(sweep_name, model, seed),
        "metric": rule_cfg.get("metric", "mase"),
        "select_on": rule_cfg.get("select_on", "val"),
        "device": rule_cfg.get("device", "cpu"),
        "residuals": rule_cfg.get("residuals"),
        "search": rule_cfg.get("search") or {},
        "runs": [{"run_id": run_id}],
        "seed": seed,
        "model": model,
    }
    out = run_rule_search(config, base=base, data_dir=data_dir)
    summary = json.loads((out / "summary.json").read_text())
    run = summary["runs"][0]
    return {
        "seed": seed,
        "model": model,
        "run_id": run_id,
        "rule_search_dir": str(out),
        "selected_rule": run["selected_rule"]["label"],
        "selected_feature": run["selected_rule"].get("feature"),
        "selected_direction": run["selected_rule"].get("direction"),
        "selected_quantile": run["selected_rule"].get("quantile"),
        "val_policy_mean": run["selection"]["policy_mean"],
        "val_base_mean": run["selection"]["base_mean"],
        "test_base_mean": run["test"]["base_mean"],
        "test_always_corrected_mean": run["test"]["always_corrected_mean"],
        "test_selective_mean": run["test"]["selective_mean"],
        "test_frac_applied": run["test"]["frac_applied"],
        "test_wilcoxon_pvalue": run["test"]["wilcoxon_pvalue"],
        "beats_base_on_test": run["beats_base_on_test"],
        "significant_on_test": run["significant_on_test"],
        "test_margin": float(run["test"]["base_mean"]) - float(run["test"]["selective_mean"]),
    }


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"models": {}}
    frame = pd.DataFrame(rows)
    models: dict[str, Any] = {}
    for model, group in frame.groupby("model", sort=True):
        g = group.reset_index(drop=True)
        rule_counts = Counter(g["selected_rule"].astype(str))
        feature_counts = Counter(
            [f for f in g["selected_feature"].tolist() if f is not None and str(f) != "None"]
        )
        models[str(model)] = {
            "n_seeds": len(g),
            "n_beats_base": int(g["beats_base_on_test"].sum()),
            "n_significant": int(g["significant_on_test"].sum()),
            "frac_beats_base": float(g["beats_base_on_test"].mean()),
            "frac_significant": float(g["significant_on_test"].mean()),
            "mean_test_margin": float(g["test_margin"].mean()),
            "std_test_margin": float(g["test_margin"].std(ddof=0)),
            "mean_test_selective": float(g["test_selective_mean"].mean()),
            "mean_test_base": float(g["test_base_mean"].mean()),
            "rule_counts": dict(rule_counts),
            "feature_counts": dict(feature_counts),
            "seeds": [int(s) for s in g["seed"].tolist()],
        }
    return {"models": models, "n_rows": len(frame)}


def run_seed_sweep(
    config: dict[str, Any],
    *,
    base: str | Path = "outputs",
    data_dir: str = "data/raw",
    skip_train: bool = False,
    force: bool = False,
) -> Path:
    if "name" not in config:
        raise ValueError("config requires 'name'")
    seeds = [int(s) for s in config.get("seeds") or []]
    if not seeds:
        raise ValueError("config.seeds must be a non-empty list")
    correctors = config.get("correctors") or []
    if not correctors:
        raise ValueError("config.correctors must be a non-empty list")
    rule_cfg = config.get("rule_search") or {}

    rows: list[dict[str, Any]] = []
    for spec in correctors:
        config_path = Path(spec["config"] if isinstance(spec, dict) else spec)
        model = str(load_config(config_path)["model"])
        for seed in seeds:
            if skip_train:
                found = find_corrector_run(model=model, seed=seed, base=base)
                if found is None:
                    raise FileNotFoundError(
                        f"no completed {model} run for seed={seed}; "
                        "re-run without --skip-train or train that seed first"
                    )
                run_id = found.run_id
            else:
                run_id = ensure_corrector_run(
                    config_path,
                    seed=seed,
                    base=base,
                    data_dir=data_dir,
                    force=force,
                )
            row = run_seed_rule_search(
                sweep_name=config["name"],
                run_id=run_id,
                model=model,
                seed=seed,
                rule_cfg=rule_cfg,
                base=base,
                data_dir=data_dir,
            )
            rows.append(row)

    out_dir = seed_sweep_paths(config["name"], base=base)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_dir / "per_seed.csv", index=False)
    per_seed.to_parquet(out_dir / "per_seed.parquet", index=False)
    aggregation = aggregate_seed_rows(rows)
    summary = {
        "name": config["name"],
        "seeds": seeds,
        "aggregation": aggregation,
        "rows": rows,
        "config": config,
        "note": (
            "For each (corrector, seed): train/reuse a corrector run, select an abstain "
            "rule on val, score test once. Aggregation reports how often the val-selected "
            "rule beats base / is significant across seeds."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "aggregation.json").write_text(
        json.dumps(aggregation, indent=2, sort_keys=True) + "\n"
    )
    return out_dir

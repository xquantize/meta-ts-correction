from __future__ import annotations

from pathlib import Path

import yaml

from meta_ts.results.manifest import (
    apply_config_overrides,
    dump_config,
    init_run,
    load_config,
)


def test_apply_seed_override_rewrites_name_and_records_override():
    cfg = {"name": "corrector_v1_chronos_m4_hourly", "model": "corrector_v1", "seed": 0}
    out = apply_config_overrides(cfg, {"seed": 3})
    assert out["seed"] == 3
    assert out["name"] == "corrector_v1_chronos_m4_hourly_seed3"
    assert out["overrides"] == {"seed": 3}
    # source untouched
    assert cfg["seed"] == 0
    assert cfg["name"] == "corrector_v1_chronos_m4_hourly"


def test_apply_seed_override_strips_existing_seed_suffix():
    cfg = {"name": "corrector_v1_chronos_m4_hourly_seed1", "seed": 1}
    out = apply_config_overrides(cfg, {"seed": 2})
    assert out["name"] == "corrector_v1_chronos_m4_hourly_seed2"


def test_apply_overrides_none_is_identity_copy():
    cfg = {"name": "x", "seed": 0}
    out = apply_config_overrides(cfg, None)
    assert out == cfg
    assert out is not cfg


def test_init_run_freezes_effective_config(tmp_path: Path):
    src = tmp_path / "cfg.yaml"
    src.write_text(
        dump_config(
            {
                "name": "corrector_v1_toy",
                "model": "corrector_v1",
                "seed": 0,
                "residuals": "r",
            }
        )
    )
    base = tmp_path / "outputs"
    manifest, paths = init_run(src, base=base, overrides={"seed": 4})
    assert manifest.name == "corrector_v1_toy_seed4"
    assert manifest.config["seed"] == 4
    frozen = load_config(paths.config)
    assert frozen["seed"] == 4
    assert frozen["name"] == "corrector_v1_toy_seed4"
    assert frozen["overrides"] == {"seed": 4}
    assert "_seed4_" in manifest.run_id or manifest.run_id.endswith(manifest.config_hash)


def test_dump_config_roundtrips_seed():
    payload = {"name": "x", "seed": 2, "split": {"train": 0.7}}
    loaded = yaml.safe_load(dump_config(payload))
    assert loaded == payload

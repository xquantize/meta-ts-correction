from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from meta_ts import __version__
from meta_ts.results.paths import RunPaths, run_paths

_SEED_SUFFIX = re.compile(r"_seed\d+$")


@dataclass
class RunManifest:
    run_id: str
    name: str
    created_at: str
    status: str
    config_hash: str
    config: dict[str, Any]
    git_sha: str | None = None
    git_dirty: bool = False
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    package_version: str = field(default_factory=lambda: __version__)
    finished_at: str | None = None
    error: str | None = None

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> RunManifest:
        return cls(**json.loads(path.read_text()))


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"config must be a mapping: {path}")
    if "name" not in config:
        raise ValueError("config requires a 'name' field")
    return config


def apply_config_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a deep copy with CLI overrides applied.

    The effective config is what gets frozen into the run folder. Seed overrides
    also rewrite ``name`` to ``{base}_seed{N}`` so multi-seed runs are readable.
    """
    out = copy.deepcopy(config)
    if not overrides:
        return out

    recorded: dict[str, Any] = dict(out.get("overrides") or {})
    if "seed" in overrides:
        seed = int(overrides["seed"])
        out["seed"] = seed
        base_name = _SEED_SUFFIX.sub("", str(out["name"]))
        out["name"] = f"{base_name}_seed{seed}"
        recorded["seed"] = seed
    if recorded:
        out["overrides"] = recorded
    return out


def dump_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def make_run_id(name: str, digest: str, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    return f"{stamp}_{safe}_{digest}"


def git_info(cwd: str | Path | None = None) -> tuple[str | None, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, False


def create_manifest(config: dict[str, Any], *, status: str = "running") -> RunManifest:
    digest = config_hash(config)
    sha, dirty = git_info()
    return RunManifest(
        run_id=make_run_id(str(config["name"]), digest),
        name=str(config["name"]),
        created_at=datetime.now(UTC).isoformat(),
        status=status,
        config_hash=digest,
        config=config,
        git_sha=sha,
        git_dirty=dirty,
    )


def init_run(
    config_path: str | Path,
    base: str | Path = "outputs",
    *,
    overrides: dict[str, Any] | None = None,
) -> tuple[RunManifest, RunPaths]:
    config = apply_config_overrides(load_config(config_path), overrides)
    manifest = create_manifest(config)
    paths = run_paths(manifest.run_id, base=base).ensure()
    manifest.write(paths.manifest)
    # Freeze the *effective* config (overrides included), not the source file.
    paths.config.write_text(dump_config(config))
    return manifest, paths


def mark_completed(manifest: RunManifest, paths: RunPaths) -> RunManifest:
    manifest.status = "completed"
    manifest.finished_at = datetime.now(UTC).isoformat()
    manifest.write(paths.manifest)
    return manifest


def mark_failed(manifest: RunManifest, paths: RunPaths, error: str) -> RunManifest:
    manifest.status = "failed"
    manifest.finished_at = datetime.now(UTC).isoformat()
    manifest.error = error
    manifest.write(paths.manifest)
    return manifest

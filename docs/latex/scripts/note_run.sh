#!/usr/bin/env bash
# Print a LaTeX run-log stub from an outputs/runs/<run_id> folder.
# Usage:
#   ./docs/latex/scripts/note_run.sh outputs/runs/<run_id>
#   ./docs/latex/scripts/note_run.sh outputs/runs/<run_id> > /tmp/entry.tex
set -euo pipefail

ROOT="${1:-}"
if [[ -z "${ROOT}" || ! -d "${ROOT}" ]]; then
  echo "usage: $0 outputs/runs/<run_id>" >&2
  exit 1
fi

RUN_ID="$(basename "${ROOT}")"
SUMMARY="${ROOT}/summary.json"
CONFIG="${ROOT}/config.yaml"

if [[ ! -f "${SUMMARY}" ]]; then
  echo "missing ${SUMMARY}" >&2
  exit 1
fi

python - "${RUN_ID}" "${SUMMARY}" "${CONFIG}" <<'PY'
import json, sys
from pathlib import Path

run_id, summary_path, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
summary = json.loads(Path(summary_path).read_text())
config_name = "unknown"
if Path(config_path).exists():
    for line in Path(config_path).read_text().splitlines():
        if line.startswith("name:"):
            config_name = line.split(":", 1)[1].strip()
            break

means = summary.get("means") or []
mean_bits = []
for row in means:
    mean_bits.append(f"{row['model']} {row['metric']}={row['value']:.6f}")
means_txt = "; ".join(mean_bits) if mean_bits else "n/a"

go = summary.get("go_nogo") or {}
decision = go.get("decision", "n/a")
p = go.get("mase_wilcoxon_pvalue", "n/a")
delta = (summary.get("comparisons") or {}).get("mase", {}).get("delta_mean", "n/a")
n = summary.get("n_series", "n/a")

print(r"\subsection*{R\# --- " + config_name.replace("_", r"\_") + r" (\today)}")
print(r"\begin{itemize}")
print(rf"  \item \textbf{{Config:}} \texttt{{configs/{config_name}.yaml}}")
print(rf"  \item \textbf{{Run id:}} \texttt{{{run_id}}}")
print(rf"  \item \textbf{{Setup:}} $n={n}$ series in summary; see frozen config in run folder.")
print(rf"  \item \textbf{{Numbers:}} {means_txt}.")
if delta != "n/a":
    print(rf"  \item \textbf{{$\Delta$MASE / Wilcoxon $p$:}} {delta} / {p}")
print(rf"  \item \textbf{{Decision:}} \textbf{{{decision}}}")
print(r"  \item \textbf{Paper note:} TODO — one sentence for the write-up.")
print(r"\end{itemize}")
PY

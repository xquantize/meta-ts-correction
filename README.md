# meta-ts-correction

Tiny residual corrector (~10⁴–10⁵ params) on top of a **frozen** time-series foundation model (Chronos-Bolt / TTM / TimesFM), conditioned on cheap series meta-features, with a learned gate that shrinks the correction toward zero where the base model is already good.

**Claim under test:** recover most of a large TSFM’s accuracy at a fraction of the cost, and characterize *when* correction helps.

**One rule:** build and validate the evaluation harness before any novel method code. If we cannot reproduce a published seasonal-naive-normalized MASE within noise on 2–3 datasets, nothing downstream is trustworthy.

## Go / no-go (decide in advance)

After corrector v1 (point residual, no meta, no gate): if it does **not** beat the frozen base model on **non-leaked** datasets under DM / Wilcoxon-Holm, pivot to a study of *when TSFMs need correction* — still publishable. A null is not failure if we decided that upfront.

## Mac setup (M-series + MPS)

Requires [uv](https://github.com/astral-sh/uv) (you already have it via Homebrew).

```bash
# Python 3.12 venv + editable install (core + dev)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Prove torch sees Metal / MPS and metrics import
meta-ts-check
# or: python -m meta_ts.smoke

# Unit tests (hand-checked MASE)
pytest -q
```

GPU note: on Apple Silicon, PyTorch uses **MPS** (Metal), not CUDA. `meta-ts-check` fails loudly if a matmul on the selected device errors.

Later (not yet):

```bash
uv pip install -e ".[forecast]"   # StatsForecast / GluonTS when baselines land
```

## Milestone sequence

| # | Milestone | Exit criterion |
|---|-----------|----------------|
| 0 | **Scaffold + smoke** (you are here) | `meta-ts-check` + `pytest` green on Mac/MPS |
| 1 | Rolling-origin windows | Unit-tested window builder, no leakage across origins |
| 2 | Metrics + stats | MASE / CRPS + DM + Wilcoxon-Holm with hand-checked tests |
| 3 | Seasonal naive + StatsForecast | Trivial + AutoARIMA/ETS baselines wired |
| 4 | **Harness validated** → tag `harness-validated` | Reproduce a published/reference MASE on 2–3 datasets |
| 5 | Frozen TSFM + residual cache | Chronos-Bolt (MPS) zero-shot; leakage table filled |
| 6 | Corrector v1 (point) | Go/no-go vs base on clean sets |
| 7 | Meta-features + γ gate | Ablations = earlier commits’ behavior |
| 8 | Full eval + figures | Multi-seed, Pareto, CD, when-it-helps |
| 9 | Paper | Draft + one-command repro |

## Layout

```
src/meta_ts/     # library code
tests/           # metric / window unit tests first
configs/         # one YAML per experiment run
docs/leakage.md  # pretrained-data overlap audit
```

Method code comes **after** the `harness-validated` tag.

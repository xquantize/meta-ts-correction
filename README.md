# meta-ts-correction

Residual correction for frozen time-series foundation models. Start with Chronos-Bolt;
a small model predicts the foundation model's errors. Later: meta-features + a gate that
shrinks (or skips) the correction when it is unlikely to help.

**Claim:** recover most of a large TSFM's accuracy at a fraction of the cost, and
characterize *when* correction helps.

## Setup (Mac / MPS)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
meta-ts-check
pytest -q
```

Forecasting extras + harness check:

```bash
uv pip install -e ".[forecast]"
meta-ts-validate-harness   # published M4 Naive2 MASE → harness-validated
```

Chronos-Bolt on MPS:

```bash
uv pip install -e ".[tsfm]"
```

## Workflow (what each step is for)

| Step | Command | What it tests / produces |
|------|---------|---------------------------|
| 1. Harness | `meta-ts-validate-harness` | Our MASE matches published M4 Naive2 (tag `harness-validated`) |
| 2. Baselines | `meta-ts-run configs/seasonal_naive_…` / `chronos_…` | Frozen Chronos vs seasonal naive on M4 Hourly |
| 3. Residuals | `meta-ts-residuals configs/residuals_…` | Point residuals + series meta-features for corrector training |
| 4. Corrector v1 | `meta-ts-run configs/corrector_v1_…` | Point residual only (no meta, no gate). Pre-committed go/no-go |
| 5. Corrector v2 | `meta-ts-run configs/corrector_v2_…` | Same + meta-features, still no gate |
| 6. When-it-helps | `meta-ts-when-it-helps configs/when_it_helps_…` | *Descriptive* quartiles of ΔMASE by meta-feature on the test fold |
| 7. Selective apply | `meta-ts-selective-apply configs/selective_apply_…` | *No retrain*: train-fit threshold → apply correction only when rule fires |

Analysis artifacts (steps 6–7) write under `outputs/tables/`, not `outputs/runs/`.

### Selective apply (step 7)

Motivated by when-it-helps: v2 hurt overall but helped on high `abs_diff_mean`.

1. Fit a threshold on the **train** fold only (e.g. 75th percentile of `abs_diff_mean`).
2. On **test**: if `feature >= threshold`, use the corrected series MASE; else keep Chronos.
3. Compare **always base** / **always corrected** / **selective**. `worth_gate=true` if selective beats always-corrected *and* is no worse than base (mean MASE).

This is a score-level replay of an existing corrector run — it does not retrain the network.

```bash
meta-ts-selective-apply configs/selective_apply_chronos_m4_hourly.yaml
# → outputs/tables/selective_apply/<name>/{summary.json,comparison.csv,series.parquet}
```

### When-it-helps (step 6)

Descriptive only (quantile edges on the analysis fold). Use it to *find* candidate rules;
selective-apply is the follow-up that fits the rule on train.

```bash
meta-ts-when-it-helps configs/when_it_helps_chronos_m4_hourly.yaml
# → outputs/tables/when_it_helps/<name>/{summary.json,strata.csv,series.parquet}
```

## Experiments (full command list)

```bash
meta-ts-run configs/seasonal_naive_m4_hourly.yaml
meta-ts-run configs/chronos_bolt_tiny_m4_hourly.yaml
meta-ts-residuals configs/residuals_chronos_m4_hourly.yaml
meta-ts-run configs/corrector_v1_chronos_m4_hourly.yaml
meta-ts-run configs/corrector_v2_chronos_m4_hourly.yaml
meta-ts-when-it-helps configs/when_it_helps_chronos_m4_hourly.yaml
meta-ts-selective-apply configs/selective_apply_chronos_m4_hourly.yaml
meta-ts-tables --list-runs
meta-ts-tables
```

Each training/forecast run writes:

```text
outputs/runs/<run_id>/
  manifest.json      # git sha, config hash, status
  config.yaml        # frozen copy
  forecasts.parquet
  scores.parquet
  summary.json
```

Corrector runs also write `splits.json`, `scaler.json`, `train_log.json`,
`comparisons.json`, and `model.pt`. Forecasts are cached under
`outputs/cache/forecasts/`. Residuals live in `outputs/datasets/residuals/<name>/`.

## Go / no-go

After corrector v1 (point residual only): if it does not beat the frozen base on
held-out series under Wilcoxon on per-series MASE ($p < 0.05$), pivot to a study of
when TSFMs need correction.

**Current:** v1 and v2 are both **no_go** ungated. When-it-helps (R5) showed
heterogeneity; selective apply on high `abs_diff_mean` (R6) recovers accuracy and
flags `worth_gate` — especially for v2. Details in `docs/latex/`. Harness tag:
`harness-validated`.

## Layout

```text
src/meta_ts/   library + experiments + analytics
tests/         unit + harness checks
configs/       one YAML per experiment / analysis
docs/          leakage audit; LaTeX notes in docs/latex/
outputs/       runs, cache, tables, figures (gitignored)
```

Keep `docs/latex/` current: after a useful run,
`./docs/latex/scripts/note_run.sh outputs/runs/<run_id>` and paste into
`sections/run_log.tex`. For table analyses, summarize from `summary.json` by hand.

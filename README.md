# meta-ts-correction

Residual correction for frozen time-series foundation models (Chronos-Bolt, TTM, TimesFM): a small model predicts the foundation model's errors, conditioned on cheap series meta-features, with a gate that shrinks the correction toward zero when the base forecast is already good.

**Claim:** recover most of a large TSFM's accuracy at a fraction of the cost, and characterize when correction helps.

## Setup (Mac / MPS)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
meta-ts-check
pytest -q
```

On Apple Silicon, PyTorch uses MPS. Forecasting extras + harness check:

```bash
uv pip install -e ".[forecast]"
meta-ts-validate-harness   # published M4 Naive2 MASE → harness-validated
```

Foundation models (Chronos-Bolt on MPS):

```bash
uv pip install -e ".[tsfm]"
meta-ts-run configs/chronos_bolt_tiny_m4_hourly.yaml
```

## Experiments

Configs live in `configs/`. Each run writes a traceable folder under `outputs/runs/`:

```text
outputs/runs/<run_id>/
  manifest.json      # git sha, config hash, status
  config.yaml        # frozen copy
  forecasts.parquet
  scores.parquet
  summary.json
```

Forecasts are also cached under `outputs/cache/forecasts/` so models are not recomputed across runs.

```bash
meta-ts-run configs/seasonal_naive_m4_hourly.yaml
meta-ts-run configs/chronos_bolt_tiny_m4_hourly.yaml
meta-ts-residuals configs/residuals_chronos_m4_hourly.yaml
meta-ts-run configs/corrector_v1_chronos_m4_hourly.yaml
meta-ts-run configs/corrector_v2_chronos_m4_hourly.yaml
meta-ts-when-it-helps configs/when_it_helps_chronos_m4_hourly.yaml
meta-ts-tables --list-runs
meta-ts-tables
```

Residual datasets land in `outputs/datasets/residuals/<name>/`. Corrector runs also write `splits.json`, `scaler.json`, `train_log.json`, `comparisons.json`, and `model.pt`. Query runs with `meta-ts-tables` (DuckDB). Stratify held-out $\Delta$MASE by meta-features with `meta-ts-when-it-helps` → `outputs/tables/when_it_helps/`.

## Go / no-go

After corrector v1 (point residual only): if it does not beat the frozen base on held-out series under Wilcoxon on per-series MASE ($p < 0.05$), pivot to a study of when TSFMs need correction.

**Current:** corrector v1 and v2 (with meta-features) are both **no_go** on M4 Hourly held-out series; v2 was worse than the frozen base. Stratified when-it-helps (R5) shows heterogeneity — especially where v2 hurts (see `docs/latex/`). Harness tag: `harness-validated`.

## Layout

```text
src/meta_ts/   library + experiments
tests/         unit + harness checks
configs/       one YAML per experiment
docs/          leakage audit; LaTeX notes in docs/latex/ (run log + paper tables)
outputs/       runs, forecast cache, tables, figures (gitignored)
```

Keep `docs/latex/` current: after a useful run, `./docs/latex/scripts/note_run.sh outputs/runs/<run_id>` and paste into `sections/run_log.tex`.

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
meta-ts-validate-harness
```

That scores official M4 Naive2 forecasts with our MASE against published Hourly / Weekly / Daily numbers (`harness-validated`).

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
```

## Go / no-go

After corrector v1 (point residual only): if it does not beat the frozen base model on non-leaked datasets under DM / Wilcoxon-Holm, pivot to a study of when TSFMs need correction.

## Layout

```text
src/meta_ts/   library + experiments
tests/         unit + harness checks
configs/       one YAML per experiment
docs/          leakage audit and notes
outputs/       runs, forecast cache, tables, figures (gitignored)
```

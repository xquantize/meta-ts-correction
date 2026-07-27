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

That last command scores official M4 Naive2 forecasts with our MASE and checks Hourly / Weekly / Daily against published numbers. Passing that is tagged `harness-validated`.

## Go / no-go

After corrector v1 (point residual only): if it does not beat the frozen base model on non-leaked datasets under DM / Wilcoxon-Holm, pivot to a study of when TSFMs need correction. Decided upfront so a null result is still publishable.

## Layout

```
src/meta_ts/   library code
tests/         unit + harness checks
configs/       one YAML per experiment
docs/          leakage audit and notes
```

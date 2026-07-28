# Leakage audit

Significance claims for the corrector must be reported on the **clean** (non-leaked)
subset, not only the full suite.

| Model | Checkpoint | Known pretrain corpora | Overlap risk with our eval |
|-------|------------|------------------------|----------------------------|
| Chronos-Bolt | `amazon/chronos-bolt-*` | Large mixed corpus (~100B obs); includes many public TS benchmarks (see Chronos paper / model card) | **High** for M4 and other standard public sets — treat M4 results as contaminated unless a held-out clean split is documented |
| TTM | TBD | TBD | TBD |
| TimesFM | TBD | TBD | TBD |

Notes:
- Chronos training data is not a simple public manifest; assume leakage for M4 / Monash-style sets until proven otherwise.
- Use non-leaked or post-cutoff private/domain series for go/no-go claims.
- Keep this table updated when wrappers land for TTM / TimesFM.

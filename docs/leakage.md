# Leakage audit

Significance claims for the corrector must be reported on the **clean**
(non-leaked) subset, not only the full suite.

| Model | Checkpoint | Known pretrain corpora | Overlap risk with our eval |
|-------|------------|------------------------|----------------------------|
| Chronos-Bolt | `amazon/chronos-bolt-tiny` (also `*-mini/small/base`) | Large mixed corpus (~100B obs); includes many public TS benchmarks (Chronos paper / model card) | **High** for M4 — treat M4 results as contaminated for strong claims |

Notes:
- Chronos training data is not a simple public manifest; assume leakage for M4 / Monash-style sets until proven otherwise.
- Current experiments use M4 Hourly for harness + development; go/no-go on M4 is exploratory, not final evidence.
- Add rows here when another foundation model is wired in.

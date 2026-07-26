# Leakage audit (first-class artifact)

Track which evaluation datasets (or close relatives) each foundation model
was pretrained on. Fill this in when Chronos-Bolt / TTM / TimesFM wrappers land.

| Model | Checkpoint | Known pretrain corpora | Overlap risk with our eval |
|-------|------------|------------------------|----------------------------|
| Chronos-Bolt | TBD | TBD | TBD |
| TTM | TBD | TBD | TBD |
| TimesFM | TBD | TBD | TBD |

Rule: significance claims for the corrector must be reported on the **clean**
(non-leaked) subset, not only the full suite.

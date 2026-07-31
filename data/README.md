# Local data (gitignored contents)

```text
data/raw/          downloaded corpora (M4, etc.) — gitignored
data/processed/    optional derived tables — gitignored
data/cache/        optional intermediate caches — gitignored
```

Populate with the usual loaders (e.g. `datasetsforecast` writes under `data/raw/`).
Do not commit raw dumps; configs and code assume this layout.

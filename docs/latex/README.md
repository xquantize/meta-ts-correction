# LaTeX working notes

Living notebook for the ~6-page report. **Sources are the record** (`*.tex`);
PDF is local/build output.

## Build

```bash
export PATH="/usr/local/texlive/2026basic/bin/universal-darwin:$PATH"
# (add that export to ~/.zshrc)

cd docs/latex
pdflatex main.tex
pdflatex main.tex
open main.pdf
```

## Layout

```text
main.tex
sections/
  workflow.tex         how we keep notes current
  claim.tex
  protocol.tex
  results.tex          short paper-facing tables only
  run_log.tex          append-only experiment log (newest on top)
  open_questions.tex
  ENTRY_TEMPLATE.tex   manual template if needed
scripts/
  note_run.sh          builds a run_log stub from summary.json
```

## After each useful run

```bash
# from repo root
./docs/latex/scripts/note_run.sh outputs/runs/<run_id>
```

1. Paste the stub at the **top** of `sections/run_log.tex`
2. Bump `R#` and write one paper-note sentence
3. Update `sections/results.tex` only if it should appear in the report
4. `pdflatex main.tex`
5. Commit the `.tex` changes with related code when you can

## Rules

- Every paper number cites a **full** `run_id`
- `run_log.tex` is append-only (newest first) — don’t rewrite history quietly
- Keep `results.tex` short; detail lives in the log
- Record **no_go** outcomes; they matter for the write-up
- `scripts/note_run.sh` and `ENTRY_TEMPLATE.tex` are tracked sources (PDF is not)

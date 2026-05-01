# Archive Migration Log (2026-04-30)

This archive namespace was created during documentation cleanup and consolidation.

## Goal
- Reduce active-document redundancy.
- Preserve legacy information without permanent deletion.
- Redirect readers to canonical documents.

## Archived / legacy groups
- `docs/report/archive/*` (already archived before this pass; remains legacy context)
- `results/paper_draft.md` (deprecated draft, superseded by canonical report + final bundle + LaTeX paper)
- `results/BENCHMARK_DEFERRALS.md` (historical run-note; not canonical narrative)

## Canonical replacements
- Primary report: `results/REPORT.md`
- Short claims: `results/EXECUTIVE_SUMMARY.md`
- Full meeting/cheat-sheet package: `results/final_bundle/*`
- Submission paper primary: `docs/paper/setconca_neurips.tex`

## Reversibility
No destructive purge performed in this cleanup phase. Legacy files remain readable and can be restored to active status if required.

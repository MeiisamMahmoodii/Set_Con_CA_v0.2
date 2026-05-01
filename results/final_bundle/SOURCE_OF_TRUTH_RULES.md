# Source-of-Truth Rules

## Rule 1: Metrics first
All numeric claims must map to keys in:
- `results/results_v2.json`
- `results/extended_alignment_results.json`
- `results/benchmark_matrix_wmt14_fr_en.json`
- `results/benchmark_matrix_opus100_multi_en.json`

## Rule 2: Report precedence
Canonical textual precedence:
1. `results/REPORT.md`
2. `results/EXECUTIVE_SUMMARY.md`
3. `results/final_bundle/*`
4. `docs/report/*` and `docs/paper/*` mirrors

## Rule 3: Tiered docs must agree
- Short doc summarizes canonical claims only.
- Full report includes all EXP and limitations.
- Cheat sheet includes terms, equations, Q/A, and claim boundaries.

## Rule 4: Deprecated docs are non-authoritative
Anything under:
- `docs/report/archive/*`
- deprecated drafts in `results/`
is historical context only.

## Rule 5: Paper sync contract
Markdown and LaTeX paper primaries must agree on:
- abstract claims,
- core experiment numbers (EXP4, EXP7, EXP12, EXP16),
- multilingual status statement,
- limitations statement.

## Rule 6: Verification gate before release
Before calling docs submission-ready:
- run `scripts/verify_reports_against_artifacts.py`
- run tests (`tests/test_setconca.py`, `tests/test_validation_gates.py`)
- record pass output in submission package index.

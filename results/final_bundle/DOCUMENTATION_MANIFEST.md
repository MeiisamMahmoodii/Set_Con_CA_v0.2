# Documentation Manifest (Canonical vs Archived)

## Canonical (authoritative)

### Core report artifacts
- `results/REPORT.md`
- `results/EXECUTIVE_SUMMARY.md`
- `results/results_v2.json`
- `results/extended_alignment_results.json`
- `results/benchmark_matrix_wmt14_fr_en.json`
- `results/benchmark_matrix_opus100_multi_en.json`
- `results/run_manifest_v2.json`

### Final-bundle deliverables
- `results/final_bundle/ShortVersion_vFinal.md`
- `results/final_bundle/FullReport_vFinal.md`
- `results/final_bundle/CheatSheet_vFinal.md`
- `results/final_bundle/Supervisor_Meeting_Paper.md`
- `results/final_bundle/MasterReport_vFinal.md`
- `results/final_bundle/ConCA_SetConCA_Math_Foundation.md`
- `results/final_bundle/PaperKit_vFinal.md`
- `results/final_bundle/PresentationKit_vFinal.md`
- `results/final_bundle/SOTAReproductionAppendix_vFinal.md`
- `results/final_bundle/ClaimLedger_vFinal.md`
- `results/final_bundle/ClaimLedger_vFinal.json`

### Docs narrative backbone
- `docs/report/README.md`
- `docs/report/narrative/*.md`
- `docs/report/generated/deep_dive/*.md`

### Submission paper backbone
- `docs/paper/setconca_neurips.tex`
- `docs/paper/references.bib`
- `docs/paper/figure_table_map.md`
- `docs/paper/reproduction_report.md`
- `docs/paper/test_audit_report.md`
- `docs/paper/claim_traceability_report.md`

## Archived / Legacy (non-canonical)
- `docs/report/archive/*` (legacy briefing and dense summaries)
- `results/paper_draft.md` (deprecated working draft)
- `results/BENCHMARK_DEFERRALS.md` (historical run note, not canonical narrative)

## Generated outputs and owner scripts
- `docs/report/generated/*` <- `scripts/build_report.py`
- `results/final_bundle/*` <- `scripts/build_final_docs_bundle.py` + manual consolidation
- `results/final_bundle/figures/small_*.png` <- `scripts/generate_supervisor_small_charts.py`

## Conflict resolution rule
If numbers disagree:
1. `results/*.json` is the metric source of truth.
2. `results/REPORT.md` is the canonical narrative report.
3. `results/EXECUTIVE_SUMMARY.md` is the canonical short claims layer.
4. Other docs must be updated to match 1-3.

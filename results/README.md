# Results directory

Canonical outputs live here only:

| Artifact | Role |
|----------|------|
| `results_v2.json` | Main EXP1–16 metrics |
| `extended_alignment_results.json` | Extended diagnostics |
| `benchmark_matrix_*.json` | Multilingual pair matrices |
| `figures/*.png` | Plots referenced by `REPORT.md` |
| `REPORT.md` | Verified numbers narrative |
| `EXECUTIVE_SUMMARY.md` | Short claims checked by `scripts/verify_reports_against_artifacts.py` |
| `paper_draft.md` | Optional prose draft |
| `BENCHMARK_DEFERRALS.md` | Deferred models/baselines log |
| `run_manifest_v2.json` | Run metadata when present |

Older briefing / dense summaries were moved to [`docs/report/archive/`](../docs/report/archive/).

- Full storyline: [`docs/report/README.md`](../docs/report/README.md)

Regenerate (GPU recommended):

```bash
uv run python scripts/run_full_pipeline.py --recompute --skip-tests
```

(Add `--skip-tests` only after you have already run `pytest`.)

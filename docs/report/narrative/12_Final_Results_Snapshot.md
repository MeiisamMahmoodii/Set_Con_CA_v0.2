# Final Results Snapshot

This is the compact final-pass scoreboard.

## Final Run Completion

- Final pass marker: `[DONE] FINAL FULL PASS`
- End time: `2026-04-29T15:03:53+09:30`
- GPU: RTX 3090

## Matrix Coverage

| Dataset | Models | Directed pairs |
|---|---:|---:|
| WMT14 fr-en | 7 | 26 |
| OPUS100 multi-en | 7 | 26 |

## Key Means (selected)

| Method | WMT14 | OPUS100 |
|---|---:|---:|
| Set-ConCA | 0.3802 | 0.3688 |
| ConCA (S=1) | 0.3720 | 0.3725 |
| PCA | 0.4542 | 0.4355 |
| SVCCA | 0.4198 | 0.4425 |
| Contrastive Alignment | 0.4658 | 0.4793 |

## Artifacts

- `results/results_v2.json`
- `results/extended_alignment_results.json`
- `results/benchmark_matrix_wmt14_fr_en.json`
- `results/benchmark_matrix_opus100_multi_en.json`
- `results/final_full_pass.log`

Next:
- [13_Multilingual_WMT14_OPUS100](13_Multilingual_WMT14_OPUS100.md)
- [15_Baseline_Comparisons](15_Baseline_Comparisons.md)


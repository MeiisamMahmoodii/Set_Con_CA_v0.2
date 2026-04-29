# Report Visual Upgrade

This page mirrors the new visual/table upgrades added to `results/REPORT.md`.

## New Visuals

- ![[../figures/fig17_report_dashboard.png]]
- ![[../figures/fig18_pipeline_flow.png]]
- ![[../figures/fig19_multilingual_method_heatmap.png]]
- ![[../figures/fig20_claim_strength_map.png]]
- ![[../figures/fig21_model_coverage.png]]

## At-a-Glance Metrics

| Item | Final-pass value |
|---|---:|
| Tests | 62 passed |
| Cross-family transfer | 69.5% +/- 0.6pp |
| Steering gain at alpha=10 | +9.8pp |
| Pointwise TopK transfer | 78.4% +/- 4.6pp |
| Set-ConCA multilingual mean | 0.3802 (WMT14), 0.3688 (OPUS100) |

## Multilingual Top-5 Ranking

| Rank | WMT14 | Score | OPUS100 | Score |
|---:|---|---:|---|---:|
| 1 | Tuned Lens | 1.0000 | Tuned Lens | 1.0000 |
| 2 | Sparse NMF | 1.0000 | Sparse NMF | 0.8810 |
| 3 | SAE-TopK | 0.8558 | SAE-TopK | 0.8128 |
| 4 | k-Sparse Learned Threshold | 0.8387 | k-Sparse Learned Threshold | 0.8000 |
| 5 | CrossCoder | 0.8122 | CrossCoder | 0.7295 |

## Claim Guardrails

| Safe to claim | Do not claim |
|---|---|
| Set-ConCA shows credible cross-family transfer and steering. | Set-ConCA dominates all baselines on raw overlap. |
| Multilingual benchmark pipeline is operational on WMT14 + OPUS100. | Consistency loss is strictly necessary in current TopK setup. |
| Linear bridge is competitive in final pass. | Corruption proves semantic collapse. |
| Set-ConCA is competitive in sparse concept-transfer framing. | PCA-32 universally improves transfer. |

Links: [[00_Home]] · [[12_Final_Results_Snapshot]] · [[15_Baseline_Comparisons]]


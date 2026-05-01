# ClaimLedger_vFinal

Each claim is tagged with evidence tier and source artifact.

| ID | Tier | Status | Claim | Evidence |
|---|---|---|---|---|
| C001 | core_result | supported | Cross-family transfer is strong and reproducible. | `results/results_v2.json:exp4_cross_family.SetConCA.transfer_g_to_l` |
| C002 | core_result | supported | Set-ConCA steering provides meaningful gains at high alpha. | `results/results_v2.json:exp7_steering.gain_at_alpha10_4B` |
| C003 | core_result | supported | Linear bridge is sufficient or better than nonlinear MLP bridge in current rerun. | `results/results_v2.json:exp12_nonlinear_bridge.summary` |
| C004 | core_result | supported | Pointwise TopK beats Set-ConCA on raw overlap in current setup. | `results/results_v2.json:exp16_topk_pointwise_vs_set` |
| C005 | negative_result | supported | Consistency loss is not a dominant transfer driver in TopK mode. | `results/results_v2.json:exp9_consistency_ablation` |
| C006 | negative_result | supported | Corruption does not collapse transfer to chance in current TopK configuration. | `results/results_v2.json:exp10_corruption_test` |
| C007 | core_result | supported | Multilingual benchmark path is operational across WMT14 and OPUS100. | `results/benchmark_matrix_wmt14_fr_en.json, results/benchmark_matrix_opus100_multi_en.json` |
| C008 | supporting | supported | Extended SOTA-like diagnostics show strong NMF and Procrustes baselines. | `results/extended_alignment_results.json:sota_extensions` |

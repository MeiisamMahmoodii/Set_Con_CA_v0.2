# Findings: Failures and Limits

This file is intentionally explicit about what did **not** hold.

## Where Set-ConCA is not top on raw overlap

| Dataset | Set-ConCA | ConCA(S=1) | SAE-TopK |
|---|---:|---:|---:|
| WMT14 | 0.3802 | 0.3720 | 0.8558 |
| OPUS100 | 0.3688 | 0.3725 | 0.8128 |

## Limits Observed

- Raw-overlap dominance is not supported.
- Some added baselines behave as extreme controls and require careful interpretation.
- NMF frequently emits convergence warnings (`max_iter=400`), though runs still complete.
- `Gromov-Wasserstein` scores are near-zero in current approximation and need deeper methodological work before headline use.

## Figures To Inspect With This Context

![](../../results/figures/fig06_sota_comparison.png)

![](../../results/figures/fig14_capability_matrix.png)

Related:
- [15_Baseline_Comparisons](15_Baseline_Comparisons.md)
- [12_Final_Results_Snapshot](12_Final_Results_Snapshot.md)


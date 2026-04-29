# Baseline Comparisons

This section compares Set-ConCA against the baseline families in final multilingual matrix runs.

## Final Means (selected)

| Method | WMT14 | OPUS100 | Read |
|---|---:|---:|---|
| Set-ConCA | 0.3802 | 0.3688 | competitive |
| ConCA (S=1) | 0.3720 | 0.3725 | close baseline |
| SAE-TopK | 0.8558 | 0.8128 | strongest raw-overlap baseline |
| CrossCoder | 0.8122 | 0.7295 | strong |
| Optimal Transport | 0.6823 | 0.6264 | strong |
| Deep CCA | 0.4377 | 0.4275 | mid-tier |
| Gromov-Wasserstein | 0.0100 | 0.0128 | currently unusable as strong result |
| Tuned Lens | 1.0000 | 1.0000 | extreme control; interpret carefully |

## Figure

![[../figures/fig06_sota_comparison.png]]

## Interpretation Rules for Writing

1. Do not claim universal SOTA win.
2. Use "competitive under set-based framing" language.
3. Separate "raw overlap winners" from "concept method suitability".

Related:
- [[11_Findings_Failures_and_Limits]]
- [[12_Final_Results_Snapshot]]


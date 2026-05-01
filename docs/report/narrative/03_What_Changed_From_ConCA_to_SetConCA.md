# What Changed: ConCA -> Set-ConCA

This is the practical delta from ConCA to Set-ConCA.

## Architecture/Training Changes

| Area | ConCA | Set-ConCA |
|---|---|---|
| Input unit | Point (`S=1`) | Set/neighborhood (`S>=1`) |
| Aggregation | N/A | Mean or attention |
| Regularization emphasis | Sparse coding | Sparse coding + set consistency behavior |
| Transfer focus | Pointwise mapping | Set-derived concept mapping |

## Evaluation Expansion

- Added broader matrix-style benchmarking.
- Added multilingual benchmark path.
- Added causal steering comparisons with controls.
- Added final-pass end-to-end run with unified log.

## Key Figures

![](../../results/figures/fig03_aggregator_ablation.png)

![](../../results/figures/fig08_convergence.png)

Next:
- [10_Findings_Successes](10_Findings_Successes.md)
- [11_Findings_Failures_and_Limits](11_Findings_Failures_and_Limits.md)


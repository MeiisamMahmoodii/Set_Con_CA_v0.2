# EXP6 SOTA Comparison

## Why this test exists

Why: compare Set-ConCA with strong decomposition/alignment baselines.

## What we test

- Stability, reconstruction, and transfer behavior under this condition.
- Whether the claimed effect remains under the current results bundle.

## How to read results

- Treat each experiment as one evidence slice, not a standalone global conclusion.
- Cross-check [Findings and limits](../../../narrative/11_Findings_Failures_and_Limits.md) and [Successes](../../../narrative/10_Findings_Successes.md).

## Fields in results_v2.json

`ConCA (S=1), PCA, PCA-Threshold (75th pct), Random, SAE (L1, pointwise), SAE (TopK, pointwise), Set-ConCA, framing`

## Figure

![EXP6 SOTA Comparison](../../../../results/figures/fig06_sota_comparison.png)

## Links

[Index Tests](Index_Tests.md) | [Report home](../../../README.md)

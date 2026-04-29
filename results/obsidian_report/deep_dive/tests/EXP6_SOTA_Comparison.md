# EXP6 SOTA Comparison

## Why this test exists

Why: compare Set-ConCA with strong decomposition/alignment baselines.

## What we test

- Stability/reconstruction/transfer behavior under this condition.
- Whether claimed effect remains under final-pass reruns.

## Significance and take

- Use this test as one evidence slice, not a standalone global conclusion.
- Final interpretation should cross-check [[../11_Findings_Failures_and_Limits]] and [[../10_Findings_Successes]].

## Available fields in artifact

`ConCA (S=1), PCA, PCA-Threshold (75th pct), Random, SAE (L1, pointwise), SAE (TopK, pointwise), Set-ConCA, framing`

## Figure

![[../figures/fig06_sota_comparison.png]]

## Links

[[Index_Tests]] | [[../00_Home]]

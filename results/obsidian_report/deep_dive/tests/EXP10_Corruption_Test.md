# EXP10 Corruption Test

## Why this test exists

Why: test robustness when inputs are corrupted.

## What we test

- Stability/reconstruction/transfer behavior under this condition.
- Whether claimed effect remains under final-pass reruns.

## Significance and take

- Use this test as one evidence slice, not a standalone global conclusion.
- Final interpretation should cross-check [[../11_Findings_Failures_and_Limits]] and [[../10_Findings_Successes]].

## Available fields in artifact

`corruption_0pct, corruption_100pct, corruption_50pct, framing`

## Figure

![[../figures/fig10_corruption_test.png]]

## Links

[[Index_Tests]] | [[../00_Home]]

# EXP16 TopK Pointwise vs Set

## Why this test exists

Why: direct transfer showdown for TopK pointwise vs set mode.

## What we test

- Stability/reconstruction/transfer behavior under this condition.
- Whether claimed effect remains under final-pass reruns.

## Significance and take

- Use this test as one evidence slice, not a standalone global conclusion.
- Final interpretation should cross-check [[../11_Findings_Failures_and_Limits]] and [[../10_Findings_Successes]].

## Available fields in artifact

`diff, framing, pointwise, set`

## Figure

![[../figures/fig16_topk_transfer.png]]

## Links

[[Index_Tests]] | [[../00_Home]]

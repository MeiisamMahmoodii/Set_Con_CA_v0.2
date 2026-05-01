# EXP16 TopK Pointwise vs Set

## Why this test exists

Why: direct transfer showdown for TopK pointwise vs set mode.

## What we test

- Stability, reconstruction, and transfer behavior under this condition.
- Whether the claimed effect remains under the current results bundle.

## How to read results

- Treat each experiment as one evidence slice, not a standalone global conclusion.
- Cross-check [Findings and limits](../../../narrative/11_Findings_Failures_and_Limits.md) and [Successes](../../../narrative/10_Findings_Successes.md).

## Fields in results_v2.json

`diff, framing, pointwise, set`

## Figure

![EXP16 TopK Pointwise vs Set](../../../../results/figures/fig16_topk_transfer.png)

## Links

[Index Tests](Index_Tests.md) | [Report home](../../../README.md)

# EXP3 Aggregator Ablation

## Why this test exists

Why: validate whether attention aggregation is useful vs simpler pooling.

## What we test

- Stability, reconstruction, and transfer behavior under this condition.
- Whether the claimed effect remains under the current results bundle.

## How to read results

- Treat each experiment as one evidence slice, not a standalone global conclusion.
- Cross-check [Findings and limits](../../../narrative/11_Findings_Failures_and_Limits.md) and [Successes](../../../narrative/10_Findings_Successes.md).

## Fields in results_v2.json

`attention, framing, mean`

## Figure

![EXP3 Aggregator Ablation](../../../../results/figures/fig03_aggregator_ablation.png)

## Links

[Index Tests](Index_Tests.md) | [Report home](../../../README.md)

# EXP7 Steering

## Why this test exists

Why: causal intervention test for concept steering behavior.

## What we test

- Stability, reconstruction, and transfer behavior under this condition.
- Whether the claimed effect remains under the current results bundle.

## How to read results

- Treat each experiment as one evidence slice, not a standalone global conclusion.
- Cross-check [Findings and limits](../../../narrative/11_Findings_Failures_and_Limits.md) and [Successes](../../../narrative/10_Findings_Successes.md).

## Fields in results_v2.json

`Random_avg, SetConCA_4B_avg, SetConCA_4B_ci95, WeakToStrong_1B_avg, WeakToStrong_ci95, alphas, baseline_sim, framing, gain_at_alpha10_4B, gain_at_alpha10_w2s`

## Figure

![EXP7 Steering](../../../../results/figures/fig07_steering.png)

## Links

[Index Tests](Index_Tests.md) | [Report home](../../../README.md)

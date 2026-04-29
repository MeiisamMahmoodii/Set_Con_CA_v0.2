# Set-ConCA Initial Idea

Set-ConCA starts from one claim:

> Concept identity is often more stable in a **local set/neighborhood** than in a single point.

## Initial Design Hypothesis

- Use set input (`S > 1`) instead of single-point input.
- Aggregate neighborhood signal (mean/attention).
- Learn sparse concept code.
- Use concept-space bridges for cross-model transfer.

## Expected Benefits (Initial)

1. Better semantic stability.
2. Better cross-model transfer robustness.
3. Cleaner concept-level causal steering.

## Visual

![[../figures/fig02_s_scaling.png]]

Related:
- [[03_What_Changed_From_ConCA_to_SetConCA]]
- [[10_Findings_Successes]]
- [[11_Findings_Failures_and_Limits]]


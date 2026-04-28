# Set-ConCA Test Audit Report

## Test Re-run Result

- Command: `uv run pytest`
- Outcome: **50 passed**, 0 failed
- Runtime: ~4.5s on current machine

## What Is Covered Well

- Core module contracts:
  - encoder/aggregator/decoder tensor shapes
  - gradient flow
  - loss decomposition integrity
  - NaN safety and basic training behavior
- Sparsity and consistency loss behavior:
  - expected range and numerical properties
  - regression protections for previously observed sparsity bugs
- Data pipeline unit behavior:
  - dataset shape checks
  - deterministic loader behavior without shuffle

## What Is Not Covered (Scientific-Claim Gaps)

These are the important missing tests relative to manuscript headline claims:

1. **Top-K branch correctness**
   - Current tests mostly validate sigmoid-sparsity path.
   - Missing direct assertions for top-k mask cardinality and zeroing behavior.

2. **Threshold claim (`S=8`)**
   - No automated test asserting comparative behavior across `S=3,8,16` (or broader sweep) with explicit criterion.

3. **Faithfulness metric claim (~60%)**
   - No deterministic integration test for `eval_faithfulness.py`.
   - No expected-range checks tied to fixed fixture/model seed.

4. **Cross-model overlap significance**
   - No test for random baseline, significance computation, or p-value pipeline.

5. **Transplantation/steering behavior**
   - No end-to-end smoke test validating a minimal bridge application workflow.

## Reliability Opinion

- **Engineering confidence:** Good for model plumbing and local refactors.
- **Scientific confidence:** Insufficient for publication claims until claim-level tests and reproducibility artifacts are added.
- **Recommendation:** Treat current tests as implementation QA, not evidence for the paper’s headline quantitative claims.

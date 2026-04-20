## Phase 3: Permutation-Invariant Set Aggregation

### Implementation summary
- File(s) created: `setconca/model/aggregator.py`, `tests/test_aggregator.py`
- Key class/function names: `SetAggregator`
- Hyperparameters (with values used): `dropout_p = 0.0` (for deterministic baselines)

### Mathematical correspondence
- Equation from paper: $u_{bar} = mean(u_i)$, $\hat{z} = LayerNorm(u_{bar})$
- How code implements it: `u.mean(dim=1)` followed by `nn.LayerNorm(concept_dim, elementwise_affine=False)`
- Any deviations and why: The code mandates `elementwise_affine=False`. Affine parameters would introduce a learnable shift/scale over the pooled approximation, breaking the log-posterior probabilistic interpretation.

### Test results
- Tests run: AGG-01, AGG-02, AGG-03, AGG-04, AGG-05, AGG-06
- All passed: yes
- Any failures and resolution: None

### Paper note
- Section of paper this supports: Section 4.1 (Set Aggregation)
- Key claim being implemented: Permutation invariance and affine-free normalization.
- Values to report in paper: `dropout_p` (0.0 for baselines, 0.1 for regularization).

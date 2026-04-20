## Phase 5: Probability-Domain Sparsity Loss

### Implementation summary
- File(s) created: `setconca/losses/sparsity.py`, `tests/test_sparsity.py`
- Key class/function names: `sparsity_loss`, `probability_domain_l1`
- Hyperparameters (with values used): `alpha` coefficient applied externally.

### Mathematical correspondence
- Equation from paper: $S(g(\hat{z})) = mean_{batch}(mean_{concept}( |g(\hat{z})| ))$ where $g(\hat{z}) = Sigmoid(\hat{z})$
- How code implements it: `torch.sigmoid(z_hat).mean()`
- Any deviations and why: Sigmoid is used instead of exact exponential map for numerical stability. Maps to (0,1) without overflow.

### Test results
- Tests run: SPAR-01, SPAR-02, SPAR-03, SPAR-04, SPAR-05
- All passed: yes
- Any failures and resolution: None

### Paper note
- Section of paper this supports: Section 4.4 (Sparse Probability-Domain Regularization)
- Key claim being implemented: L1 penalty applied in the probability space.
- Values to report in paper: Chosen alpha value and selection method.

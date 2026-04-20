## Phase 6: Subset Consistency Regularization

### Implementation summary
- File(s) created: `setconca/losses/consistency.py`, `tests/test_consistency.py`
- Key class/function names: `consistency_loss`
- Hyperparameters (with values used): Minimum set size 4, parameter $k = floor(m/2)$, external `beta`.

### Mathematical correspondence
- Equation from paper: $L_{cons} = mean_{batch}( ||\hat{z}_a - \hat{z}_b||_2^2 )$
- How code implements it: Disjoint random subsets with `set_size // 2` passed through the same model encoder and aggregator, resulting in L2 squared loss.
- Any deviations and why: Guard clause `S < 4` avoids empty splits or splits that are too small to be meaningful.

### Test results
- Tests run: CONS-01, CONS-02, CONS-03, CONS-04, CONS-05, CONS-06
- All passed: yes
- Any failures and resolution: None

### Paper note
- Section of paper this supports: Section 4.5 (Subset Consistency Regularization)
- Key claim being implemented: Gradients flow to the shared encoder from both disjoint subsets, enforcing stability under subsampling noise.
- Values to report in paper: Minimum set size guard (S < 4) and beta value selected.

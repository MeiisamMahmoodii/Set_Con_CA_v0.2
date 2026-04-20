## Phase 4: Shared + Residual Dual Decoder

### Implementation summary
- File(s) created: `setconca/model/decoder.py`, `tests/test_decoder.py`
- Key class/function names: `DualDecoder`
- Hyperparameters (with values used): `concept_dim`, `hidden_dim`

### Mathematical correspondence
- Equation from paper: $f_{hat}(x_i) = W_{shared} * \hat{z} + W_{residual} * u_i + b_d$
- How code implements it: Two `nn.Linear` layers (shared and residual) without individual biases, plus a single `nn.Parameter` bias vector `b_d`.
- Any deviations and why: Single shared `b_d` across both streams is explicitly enforced to avoid parameter bloat, keeping the separation of concerns clean.

### Test results
- Tests run: DEC-01, DEC-02, DEC-03, DEC-04, DEC-05, DEC-06
- All passed: yes
- Any failures and resolution: None

### Paper note
- Section of paper this supports: Section 4.2 (Decoder)
- Key claim being implemented: Shared stream captures concept-level commonality, residual stream captures instance-specific variation. Also uses single bias `b_d`.
- Values to report in paper: W_shared and W_residual initialized via Xavier uniform.

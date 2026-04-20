## Phase 2: Element Encoder

### Implementation summary
- File(s) created: `setconca/model/encoder.py`, `tests/test_encoder.py`
- Key class/function names: `ElementEncoder`
- Hyperparameters (with values used): `hidden_dim`, `concept_dim`

### Mathematical correspondence
- Equation from paper: $u_i = W_e * f(x_i) + b_e$
- How code implements it: `nn.Linear(hidden_dim, concept_dim, bias=True)` applied to `(B, S, D)` independently.
- Any deviations and why: None. The implementation explicitly avoids any non-linear activations (like ReLU or GELU) to preserve the log-posterior domain interpretation.

### Test results
- Tests run: ENC-01, ENC-02, ENC-03, ENC-04, ENC-05
- All passed: yes
- Any failures and resolution: None

### Paper note
- Section of paper this supports: Section 4.0.1 (Element Encoding)
- Key claim being implemented: No nonlinearity applied
- Values to report in paper: The initialization scheme used (Xavier uniform) and whether `concept_dim > hidden_dim` representing an overcomplete basis.

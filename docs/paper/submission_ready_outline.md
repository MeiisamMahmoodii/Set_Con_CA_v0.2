# Submission-Ready Set-ConCA Manuscript Shell

This is a drafting shell you can fill directly, section by section, with the exact order, paragraph purpose, and artifact linkage needed for a publishable NeurIPS submission.

## Global Writing Contract

- Every numeric claim must cite one script and one artifact file.
- Theory claims and empirical claims must be linguistically separated.
- Main paper uses only artifact-backed numbers.
- All deeper details move to appendix sections referenced from main text.

---

## 1. Title

`Set-ConCA: Distributional Concept Component Analysis for Cross-Model Transplantation and Steering`

---

## 2. Abstract (6 sentences, fixed roles)

1. State the pointwise-transfer limitation in current interpretability methods.
2. Introduce Set-ConCA as a set-level extension of ConCA.
3. Summarize architecture and objective terms.
4. Report only artifact-backed topline findings.
5. Explain why this matters for transplantation and steering.
6. Calibrate claim scope (confirmed vs provisional).

---

## 3. Introduction

### Paragraph 1: Context and motivation
- Mechanistic interpretability goals.
- Why cross-model portability matters.

### Paragraph 2: Problem statement
- Define pointwise residual noise and transfer instability.
- State precise research objective.

### Paragraph 3: Proposed approach
- Set-level latent extraction.
- Consistency regularization.
- Bridge for cross-model transfer and steering.

### Paragraph 4: Contributions
- Bullet list with four lines:
  - theory,
  - method,
  - empirical,
  - reproducibility.

### Paragraph 5: Roadmap
- One sentence per major section.

---

## 4. Related Work

### 4.1 ConCA lineage
- What ConCA establishes.
- What remains unresolved for cross-model transfer.

### 4.2 SAE and sparse-feature work
- Classic SAEs, scaling SAEs, top-k SAEs.
- Distinguish sparse decomposition from transfer-ready coordinates.

### 4.3 Distributional representation learning
- Position against GDE and clarify complementarity.

### 4.4 Causal interpretability + steering
- Causal metrics and model editing/steering baseline framing.

### Closing paragraph
- 2-3 sentence novelty gap statement.

---

## 5. Problem Setup and Notation

### Paragraph 1: Data objects
- Define representation sets and anchors.

### Paragraph 2: Latent variables
- Define set-level concept latent and reconstruction targets.

### Paragraph 3: Transfer object
- Define bridge mapping and target steering variable.

### Paragraph 4: Evaluation axes
- Define all reported axes before results.

---

## 6. Method: Set-ConCA

### 6.1 From pointwise ConCA to setwise inference
- Recap pointwise formulation.
- Motivate set-lifting.

### 6.2 Architecture
- Element encoder.
- Permutation-invariant aggregator.
- Sparse latent operator (sigmoid path and top-k path).
- Shared + residual decoder.

### 6.3 Objective
- Write full objective with notation-consistent terms.
- Clarify branch-specific objective in top-k mode.

### 6.4 Bridge + steering interface
- Bridge training target.
- Inference-time transplant and steering usage.

### 6.5 Implementation summary
- Keep minimal; reference appendix for details.

---

## 7. Theory

### 7.1 Assumptions
- Latent-variable assumptions.
- Conditional-independence assumptions.
- Bounded residual/linearization assumptions.

### 7.2 Proposition 1
- Set aggregation consistency; variance trend with set size.

### 7.3 Proposition 2
- Posterior aggregation interpretation.

### 7.4 Proposition 3
- Transferability/bridge condition with null-relative statement.

### 7.5 Proof placement
- Main text: proof sketches.
- Appendix A: full proofs/lemmas.

---

## 8. Experimental Setup

### 8.1 Model and dataset inventory
- Source/target models and data pipeline summary.

### 8.2 Baselines
- ConCA baseline(s), SAE baseline(s), pointwise baseline, null baselines.

### 8.3 Protocol
- Splits, seeds, stopping criteria, compute platform.

### 8.4 Metrics and statistics
- Mathematical definitions.
- Bootstrap and permutation procedures.

---

## 9. Results

### 9.1 Main comparative results
- Set-ConCA vs all baselines on primary metrics.

### 9.2 Set-size scaling
- Report full sweep; interpret knee empirically.

### 9.3 Top-k ablation
- k-grid and sparsity-mode tradeoffs.

### 9.4 Bridge significance
- Overlap/Jaccard/retrieval vs nulls + significance.

### 9.5 Causal faithfulness
- Layer-wise faithfulness, with and without norm alignment.

### 9.6 Steering transfer case study
- Transplanted direction vs control directions.

---

## 10. Robustness, Failure Cases, Limitations

### Paragraph 1
- Robustness across seeds/models/hyperparameters.

### Paragraph 2
- Explicit failure cases and degradation conditions.

### Paragraph 3
- Scope and statistical limitations.

---

## 11. Broader Impact and Ethics

### Paragraph 1
- Potential positive uses.

### Paragraph 2
- Misuse risks.

### Paragraph 3
- Mitigation and governance.

---

## 12. Reproducibility Statement

### Paragraph 1
- Artifact traceability map.

### Paragraph 2
- Environment and seed guarantees.

### Paragraph 3
- Runtime expectations and reproducible subset in <1 week.

---

## 13. Conclusion

### Paragraph 1
- What is now validated.

### Paragraph 2
- What remains provisional/future.

### Paragraph 3
- Practical impact and next steps.

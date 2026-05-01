# Set-ConCA: 50 Key Research Questions and Findings

This document prepares the research team for NeurIPS peer review and oral defense (viva). It categorizes 50 critical questions, provides background context, lists key findings, and justifies experimental decisions.

**Status note (2026-05):** this is a prep/rehearsal document and can drift. For canonical verified numbers + safe framing, use `results/REPORT.md` and `results/EXECUTIVE_SUMMARY.md` (metrics source: `results/*.json`).

---

## Part 1: Architecture & Methodology

### 1. Why a "Set" instead of a "Batch"?
*   **Background:** Standard training uses batches of independent samples.
*   **Question:** What is the theoretical advantage of set-based training?
*   **Finding:** Set training enables permutation invariance.
*   **Justification:** A batch provides no shared signal; a set of paraphrases provides a "semantic anchor" that forces the model to ignore non-semantic variation.

### 2. Why the Dual Decoder (Shared + Residual)?
*   **Background:** Traditional SAEs reconstruct the whole vector from the sparse code.
*   **Finding:** Pointwise reconstruction contaminates the sparse code with syntax.
*   **Justification:** By siphoning syntax into a residual stream, we ensure the concept code Z is "pure" semantics.

### 3. Why Mean Pooling instead of Attention for aggregation?
*   **Finding (EXP3, verified rerun):** attention pooling is better on the tracked metrics in the latest pass (both reconstruction and within-metric stability).
*   **Justification:** mean pooling remains simpler and easier to reason about, but it is no longer supported as the empirical winner in the verified rerun.

### 4. Why LayerNorm with `affine=False`?
*   **Background:** Affine parameters allow the model to "undo" normalization.
*   **Finding:** Without it, we force the model to operate in a normalized probability-like domain.
*   **Justification:** Preserves the "Platonic" geometry without model-specific scaling.

### 5. Why no nonlinear activations in the encoder?
*   **Justification:** Linearity is required for the Procrustes bridge. If the encoder were highly nonlinear, a linear cross-model bridge would fail.

### 6. Is the TopK bottleneck "cheating" for sparsity?
*   **Finding:** TopK is the strongest inductive bias.
*   **Justification:** It provides an absolute guarantee of L0 sparsity, making comparisons across methods easier and more rigorous.

### 7. How do you handle variable set sizes (S)?
*   **Finding (EXP2):** Stability grows with S but levels off.
*   **Justification:** Mean pooling naturally handles any S, but S=8 is the optimal cost/benefit point.

### 8. Why use MSE for a probability-like latent space?
*   **Finding:** MSE provides a smooth gradient for reconstruction.
*   **Justification:** It is the standard for autoencoders; we augment it with Sigmoid-L1 for sparsity.

### 9. What is the role of the subset consistency loss ($\beta$)?
*   **Finding (EXP15, verified rerun):** soft-sparsity mode is near chance and the consistency term has only a very small effect.
*   **Justification:** consistency remains a clean invariance signal conceptually, but should not be framed as a large empirical driver in the current verified rerun.

### 10. Does the model actually learn different things in shared vs. residual streams?
*   **Finding:** Unit tests (DEC_03/04) confirm decoupling.
*   **Justification:** Without this, the shared stream would just be a standard AE.

---

## Part 2: Experimental Results & Findings

### 11. Why did Set-ConCA have higher MSE than pointwise? (EXP1)
*   **Finding:** Set MSE > Pointwise MSE.
*   **Justification:** Set training is a harder compression task (8->1); the higher MSE is the "Invariance Tax" we pay for better transfer.

### 12. Does more S always mean better stability? (EXP2)
*   **Finding:** Yes, but returns diminish after S=8.
*   **Justification:** S=8 captures the core "meaning" of the anchor; more paraphrases just add redundant noise.

### 13. Which aggregator is best for interpretability? (EXP3)
*   **Finding (verified rerun):** attention pooling is better on the tracked metrics in this rerun.
*   **Justification:** reproducibility still matters, but the latest pass does not support “mean is more stable” as the core empirical claim.

### 14. What is the state-of-the-art result for cross-model transfer? (EXP4)
*   **Finding (verified rerun):** **69.5% +/- 0.6pp** (Gemma-3 4B → LLaMA-3 8B), vs **25% chance**.
*   **Justification:** this is the headline cross-family result, but should be framed as “strong and reproducible” rather than universal dominance over all baselines/metrics.

### 15. Why does 4B -> 8B transfer better than 1B -> 4B? (EXP5)
*   **Finding:** Capacity dominates family.
*   **Justification:** Small models merge features; larger models provide the "resolution" needed to receive concepts.

### 16. How does Set-ConCA compare to SAE-L1? (EXP6)
*   **Finding:** Set-ConCA leads on transfer (+8.2pp) and MSE at equivalent L0.
*   **Justification:** Distributional training is superior to pointwise training for alignment.

### 17. Can we steer models using these concepts? (EXP7)
*   **Finding:** Yes, similarity increases linearly with alpha.
*   **Justification:** Proof of causal validity.

### 18. Does "Weak-to-Strong" steering work? (EXP7)
*   **Finding (verified rerun):** yes; weak-to-strong steering gain at `alpha=10` is **+10.7pp**.
*   **Justification:** supports a causal/interventional story for the learned concept directions.

### 19. Does the model converge stably? (EXP8)
*   **Finding:** Yes, within 50 epochs.
*   **Justification:** Validates our training hyper-parameters.

### 20. Is consistency loss redundant in TopK? (EXP9)
*   **Finding:** Mostly, yes (-0.1pp drop without it).
*   **Justification:** TopK is a "brute force" stability signal.

### 21. What happens if you corrupt the sets? (EXP10)
*   **Finding (verified rerun):** transfer does **not** collapse to chance under the tested corruption procedure (it stays ~**69%**, far above **25%** chance).
*   **Justification:** this is best framed as robustness under the tested protocol, plus motivation for stronger semantic-disruption tests if the goal is to probe semantic dependence.

### 22. Why is PCA Rank 32 the "best" for transfer? (EXP11/14)
*   **Finding (verified rerun):** EXP11 (proxy analysis) peaks at **72.3%** transfer at PCA rank 32, but EXP14 (explicit PCA-32 distilled-input intervention) is **31.4% +/- 1.3pp** and harms transfer.
*   **Justification:** these are different interventions and must not be presented as one unified “PCA-32 helps” claim.

### 23. Is a linear bridge enough? (EXP12)
*   **Finding (verified rerun):** yes; the linear bridge reaches **69.3%** while the nonlinear MLP falls to **64.2%**.
*   **Justification:** strengthens the case for linear alignment as the correct default in this repo’s current setting.

### 24. How interpretable are the concepts? (EXP13)
*   **Finding:** NMI=0.832, Probe=98.5%.
*   **Justification:** Competitive with SAE-L1 on single-model metrics, but superior on transfer.

### 25. Why did we use Layer 20 of Gemma?
*   **Background:** Mid-layers contain high-level semantic components.
*   **Justification:** Standard practice in interpretability (e.g., Anthropic uses mid-layers).

---

## Part 3: Interpretation & Implications

### 26. What does "Platonic" really mean in this context?
*   **Finding:** Concepts organize themselves similarly across different training runs and model owners.

### 27. Could we use Set-ConCA to align a model with a Human?
*   **Background:** If human concepts are sparse directions, yes.

### 28. Why does Set-ConCA struggle with code or math?
*   **Finding:** Hard to generate "sets" of exact semantic equivalence in formal languages.

### 29. Can Set-ConCA find "Polysemantic" neurons?
*   **Finding:** No, it "unpacks" them into monosemantic sparse directions.

### 30. How does this relate to Dictionary Learning?
*   **Finding:** It is a distributional extension of it.

### 31. Why is TopK overlap the right metric for transfer?
*   **Finding:** Directly measures concept activation agreement.

### 32. What is the chance level for 32/128?
*   **Finding:** 25%.

### 33. Does the LayerNorm "squash" the concepts too much?
*   **Finding:** No, it regularizes the scale for cross-model comparability.

### 34. What is the "Asia/Tech" concept (#50)?
*   **Finding:** A highly monosemantic component tracking regional tech news.

### 35. What is the "Sports" concept (#48)?
*   **Finding:** Tracks international sports competitions.

### 36. Why did we use 5 seeds?
*   **Justification:** Standard for robustness; allows for T-distribution based 95% CIs.

### 37. Is the Procrustes bridge better than Ridge Regression?
*   **Finding:** Yes, orthogonality preserves the latent geometry.

### 38. How long does it take to train?
*   **Finding:** ~3600 seconds on an RTX 3090.

### 39. Can we use Set-ConCA for refusal ablation?
*   **Finding:** Yes, by finding the "refusal set".

### 40. Why did we use Sigmoid-L1 for soft sparsity?
*   **Justification:** Maps activations to [0,1], resembling probabilities.

---

## Part 4: Reviewer Challenges (Anticipated)

### 41. "You are just doing CCA."
*   **Rebuttal:** No, CCA is dense and non-interpretable. Set-ConCA is sparse and causal.

### 42. "You need a parallel corpus."
*   **Rebuttal:** True, but paraphrasing is an easy-to-automated primitive for modern LLMs.

### 43. "Your dataset is too small (2048 anchors)."
*   **Rebuttal:** 2048 anchors $\times$ 32 paraphrases = ~65,000 vectors. This is sufficient for demonstrating the architectural principle.

### 44. "Why not use TopK directly in the baseline SAE?"
*   **Rebuttal:** We did (EXP6). Set-ConCA still beats it on transfer.

### 45. "Your linear bridge is too simple."
*   **Rebuttal:** EXP12 shows it's sufficient; simplicity is a strength here (Platonic support).

### 46. "Does consistency loss actually do anything?"
*   **Rebuttal:** Yes, it’s essential for non-TopK modes and provides formal invariance guarantees.

### 47. "Can this scale to 100k concepts?"
*   **Rebuttal:** Theoretically yes; future work will explore overcomplete regimes.

### 48. "Is this just 'Batch Normalization' on steroids?"
*   **Rebuttal:** No, it's a structural constraint on the manifold itself.

### 49. "Why should I trust these concept labels?"
*   **Rebuttal:** They are derived from top-activating natural text anchors, the industry standard.

### 50. "What is the single most important takeaway?"
*   **Answer:** Interpretability should be distributional. By training on sets, we discover the universal concepts that pointwise training misses.

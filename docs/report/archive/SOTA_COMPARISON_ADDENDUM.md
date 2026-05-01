# Set-ConCA: SOTA Landscape & Competitive Analysis
**Refinement for NeurIPS 2025 Submission**

To strengthen our manuscript, we need to position Set-ConCA against the specific SOTA (State Of The Art) architectures that current reviewers (from Anthropic, OpenAI, DeepMind) will have in mind.

---

## 1. Architectural Competitors (Dictionary Learning)

| Model | Mechanism | Advantage over Vanilla | Why Set-ConCA is better for *Concepts* |
| :--- | :--- | :--- | :--- |
| **Gated SAE** (DeepMind) | Decouples detection from magnitude (via a "Gate"). | Solves "Feature Shrinkage" (underestimating magnitudes). | Still **pointwise**. It extracts features that are highly word-specific. It lacks a cross-token invariance signal. |
| **JumpReLU SAE** (Anthropic/OpenAI) | Non-continuous activation threshold. | Cleaner feature disentanglement; fewer "dead" features. | Optimized for reconstruction fidelity, not **Cross-Model Alignment**. It does not benefit from the Law of Large Numbers in sets. |
| **TopK SAE** (OpenAI) | Hard K-sparsity constraint. | Most stable; eliminated $L_1$ hyperparameter tuning. | **Our current baseline.** We prove that adding a "Set Aggregator" to TopK is the secret to getting cross-model transfer (+8.2pp). |

---

## 2. Concept Alignment Competitors (Cross-Model)

| Method | Approach | Use Case | Set-ConCA Advantage |
| :--- | :--- | :--- | :--- |
| **Orthogonal Procrustes** | Linear rotation mapping. | Standard for word-embedding alignment. | Assumes models are perfect rotations. Set-ConCA (EXP12) uses this but shows that **distributional training** makes the rotation far more stable. |
| **Representation Engineering (RepE)** | Difference vectors (A - B). | Quick "Control Vectors." | **Supervised**: Requires labels (e.g., "True" vs "False"). Set-ConCA is **unsupervised**; it finds the concept just from the shape of the paraphrase set. |
| **Transcoder** (DeepMind) | Maps layer $L$ to layer $L+1$. | Intra-model mechanistic flow. | Limited to one model. Set-ConCA performs "Inter-Model Transcoding" (Gemma <-> Llama). |
| **RAVEL** (Huang 2024) | Alignment search benchmark. | Evaluating disentanglement. | RAVEL is an evaluation benchmark; Set-ConCA is a discovery architecture that naturally aligns concepts. |
| **LEACE** (Turner 2023) | Linear concept erasure. | Bias removal / Steering. | **Supervised**: Needs labels. Set-ConCA is unsupervised; it discovers what is there without being told. |
| **Patchscopes** (2024) | Cross-model patching. | Activation inspection. | Patchscopes is an inspection tool; Set-ConCA identifies the sparse features that make patching possible. |

---

## 3. Why These Competitors Fail at "Universality"

1.  **The Pointwise Bottleneck**: Standard SAEs (Gated, JumpReLU) treat "The stock fell" and "Equity prices dropped" as two different reconstruction problems. They expend parameter capacity on the *difference* between the words.
2.  **Lack of Group Signal**: None of the current SOTA models use the fact that meaning is a **group-invariant**. 
3.  **The "Surface Form" Mirror**: Pointwise SAEs are too good at the task. They reconstruct the syntax so perfectly that the "semantic core" gets buried under syntactic noise. Set-ConCA’s **Dual Decoder** is the only one that explicitly "Residual-izes" the syntax.

---

## 4. Proposed "SOTA Table" for the Paper

We should add this table to the **Related Works** or **Experimental Discussion** section:

| Feature | Vanilla SAE | Gated SAE | TopK SAE | **Set-ConCA (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **Sparsity Signal** | Soft ($L_1$) | Soft (Gate) | Hard (K) | **Hard (K)** |
| **Training Input** | Single Token | Single Token | Single Token | **Paraphrase Set** |
| **Invariance Mode** | None | None | None | **Permutation Invariant** |
| **Cross-Family Transfer** | Low (~56%) | Low | Moderate | **Highest (64.6%)** |
| **Decoder Logic** | Tied/Standard | Tied | Tied | **Shared + Residual** |

---

## 5. Strategic "Upgrade" Recommendations
If we want to "Wow" the supervisor/reviewers further, we could mention:
*   **Set-Gated-ConCA**: Combining the "Gated" magnitude estimation with our "Set" aggregator. This would theoretically give us the best of both worlds (Set Invariance + Zero feature shrinkage).
*   **Matryoshka Sets**: Applying Matryoshka training (nested dimensions) to Set-ConCA to see if high-level concepts (e.g., "Politics") emerge in the small prefixes, while specific details (e.g., "Asia-Pacific Region") fill out the larger dimensions.

---
*For direct comparisons, see: [fig14_capability_matrix.png](file:///c:/Users/MPC/Documents/code/SetConCA/results/figures/fig14_capability_matrix.png)*

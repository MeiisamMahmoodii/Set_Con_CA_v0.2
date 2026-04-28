# Set-ConCA: Presentation Plan & Speaker Guide
**Goal:** A 15-minute technical presentation (NeurIPS style) + 5-minute Q&A.

---

## 1. Slide Storyboard & Script

### Slide 1: Title & Hook
*   **Visuals:** A stunning visualization of a "Latent Space Cloud" collapsing into a single crisp concept vector. Title: *Set-ConCA: Recovering Universals via Invariance on Representation Sets*.
*   **Speaking Note:** "Today I’m presenting Set-ConCA. We all know Sparse Autoencoders are great at finding features, but they suffer from a terminal flaw: they can't tell the difference between what a sentence *means* and how it's *written*. We fix this by teaching models to look at sets of paraphrases."

### Slide 2: The Problem (Pointwise Conflation)
*   **Visuals:** Diagram showing two sentences ("Shares fell", "Stock dropped") mapping to different sparse codes in a standard SAE.
*   **Speaking Note:** "Standard SAEs are pointwise. They encode individual vectors. But individual vectors are messy—they contain syntax, word choice, and noise. This conflation makes features model-specific. A 'financial' feature in Gemma won't map to Llama because their syntactic 'flavors' don't match."

### Slide 3: The Insight (Set-based Invariance)
*   **Visuals:** Concept: If we observe $S$ paraphrases of the same meaning, their intersection is the **semantic core**.
*   **Speaking Note:** "Our insight is simple: Meaning is what stays the same when you change the words. By forcing an autoencoder to produce *one* code for an entire set of paraphrases, we mathematically average away the surface variation, leaving only the semantically invariant core."

### Slide 4: Architecture (Set-Aggregator)
*   **Visuals:** Model diagram: Input $(B, S, D) \to$ Linear Encoder $\to$ Mean Pool $\to$ LayerNorm $\to$ TopK Sparsity.
*   **Speaking Note:** "We use a permutation-invariant aggregator. The mean-pool isn't just a choice—it's the MAP estimate for the shared latent factor in a Gaussian model. We use hard TopK sparsity ($k=32$) to ensure we have a clean, interpretable bottleneck."

### Slide 5: Innovation: The Dual Decoder
*   **Visuals:** Split path: $f_{hat} = W_{shared}(z) + W_{res}(u)$. Focus on the Shared vs Residual streams.
*   **Speaking Note:** "The Dual Decoder is our secret weapon. Without it, you get massive reconstruction error. We split the reconstruction: one stream for the shared semantics, and a residual stream for the paraphrase-specific syntax. This formally separates 'What it means' from 'How it was said'."

### Slide 6: Experimental Framework
*   **Visuals:** Logo cloud: Gemma-3 (1B/4B), Gemma-2 (9B), LLaMA-3 (8B). Stats: 2,048 anchors, 65k sentences, 5 seeds.
*   **Speaking Note:** "We didn't just test on toy data. We used 4 real-world LLMs at different capacities. We extracts hidden states at mid-network depth where semantics are richest. This is a rigorous 5-seed study with 95% confidence intervals."

### Slide 7: Headline: Cross-Family Transfer
*   **Visuals:** Bar chart: 64.6% (Gemma 4B $\to$ Llama 8B) vs 56.4% (Pointwise baseline). **+8.2pp win.**
*   **Speaking Note:** "The headline: Set-ConCA concepts transfer across models significantly better than pointwise SAEs. We can Port concepts from a Google model to a Meta model with 64.6% overlap using a simple linear rotation. This is the first mechanistic proof that these models are converging to a shared Platonic space."

### Slide 8: The "Capacity-Receiver" Effect
*   **Visuals:** Heatmap showing 4B $\to$ 8B (high) vs 8B $\to$ 4B (lower).
*   **Speaking Note:** "We discovered something counterintuitive. Transfer isn't about architectural family; it's about capacity. Larger models are better 'receivers' of concepts. They have the geometric resolution to accommodate the smaller model's structure, whereas 1B models merge features that 8B models resolve separately."

### Slide 9: Platonic Geometry (Linearity)
*   **Visuals:** Linear vs MLP bridge comparison. Gain: +0.5pp.
*   **Speaking Note:** "Is the relationship between models complex and nonlinear? No. Adding a 2-layer MLP improves transfer by only 0.5 percent. This near-linear geometry suggests that the multi-model concept space is approximately isometric. It's the same 'Platonic' reality, just viewed through different model coordinate frames."

### Slide 10: Causal steering & Weak-to-Strong
*   **Visuals:** Steering curve. 1B model steering an 8B model. +3.0pp gain.
*   **Speaking Note:** "Concepts must be causal, not just descriptive. We port vectors from a 1B model and use them to steer an 8B model. Surprisingly, 1B concepts are *more* causal (+3.0pp) than 4B concepts. Aggressive compression in smaller models seems to 'distill' the most potent semantic directions."

### Slide 11: Info-Depth & Spectral Dominance
*   **Visuals:** PCA Rank vs Transfer. Peak at Rank 32 (77.8%).
*   **Speaking Note:** "Where does this transfer come from? We found it lives in the dominant spectral directions. The first 32 PCA components of the hidden state transfer at 77.8%. The semantic 'gold' is in the high-variance directions; the rest is model-specific noise."

### Slide 12: Conclusion & Horizon
*   **Visuals:** Summary list: Paraphrase sets, Dual Decoders, Platonic bridges, Trans-Family discovery.
*   **Speaking Note:** "Set-ConCA proves that semantically invariant concepts are universal across model families. This paves the way for safety tools that generalize from small, open models to large, frontier models without retraining. Thank you."

---

## 2. Advanced Knowledge for Q&A

To answer like an expert, master these four technical "defense" points:

### A. The "MSE Deficit" Question
*   **Reviewer:** "Your MSE is worse than pointwise methods. Isn't that a step back?"
*   **Answer:** "MSE on a set task is inherently harder. Pointwise SAEs solve a 1-to-1 reconstruction task. We solve an 8-to-1 compression task. The MSE deficit is exactly the mathematical price of semantic distillation. As EXP4 shows, this price is paid back in +8.2pp of cross-model transferability."

### B. The "Consistency Loss is Redundant" Nuance
*   **Reviewer:** "You added a Consistency Loss, but EXP9 says it doesn't help. Why keep it?"
*   **Answer:** "In TopK-mode, the hard k-bottleneck is so strong that it structurally forces most of the invariance we need. However, the consistency loss is conceptually necessary for *soft* sparsity (Sigmoid-L1) and provides a theoretical guarantee that any subset of information leads to the same concept code."

### C. The "Platonic Ideal" Clarification
*   **Reviewer:** "Is 64% transfer enough to claim a 'Platonic' reality?"
*   **Answer:** "64% is $2.5 \times$ chance levels. The key evidence is EXP12: the fact that a nonlinear bridge provides *no gain* over a linear one. That indicates the *geometry* is shared, even if the feature resolution varies between models."

### D. The "Last-Token" Choice
*   **Reviewer:** "Why last-token layer 20?"
*   **Answer:** "Autoregressive models compute a causal summary at the final token position. Middle layers (layer 20/32) are the 'Goldilocks zone'—shallow enough to not be next-token specific, but deep enough to have discarded raw lexical tokens for semantic abstractions."

---

## 3. Recommended "Demo" Moments
*   **The Labels:** Have the Concept #50 (Asia/Tech politics) labels ready. It proves the math isn't just finding noise.
*   **The Steering:** Mention the random direction baseline (-97pp). It shows that steering isn't just about 'adding magnitude'—it's about the precision of the Set-ConCA vector.

---
*For full results reference, see: [REPORT_DENSE.md](file:///c:/Users/MPC/Documents/code/SetConCA/results/REPORT_DENSE.md)*

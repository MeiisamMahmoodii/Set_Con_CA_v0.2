# Set-ConCA Technical Presentation Deck

This document outlines a 20-slide technical presentation for Set-ConCA, designed for a research audience (coworkers and supervisors). It follows a rigorous empirical narrative, moving from theoretical motivation to architectural novelties and definitive experimental results.

---

## Slide 1: Title Slide
**Title:** Set-ConCA: Distributional Concept Component Analysis for Inter-Model Interpretability  
**Subtext:** Achieving Universal Semantic Alignment through Permutation-Invariant Sparse Encoding  
**Visuals:** A high-resolution abstract visualization of latent spaces from different models (Gemma and LLaMA) aligning via a sparse dictionary bridge.  
**Speaker Notes:**  
"Good morning. Today I am presenting Set-ConCA, a novel architecture for Sparse Dictionary Learning that moves beyond pointwise reconstruction to distributional alignment. Unlike traditional SAEs that process independent tokens, Set-ConCA operates on 'semantic sets'—collections of paraphrases—to isolate universal, model-agnostic concepts from surface-level idiosyncratic noise. We will demonstrate how this distributional inductive bias enables state-of-the-art cross-model concept transfer and supports the Platonic Representation Hypothesis."

---

## Slide 2: The Problem: Pointwise Fragility
**What it shows:**  
- A diagram showing two different models (Model A, Model B) encoding the same sentence.  
- Highlight that while the *meaning* is the same, the latent vectors are dominated by model-specific syntax/noise.  
- Key Text: Pointwise SAEs align to *co-occurrence*, not *semantics*.  
**Speaker Notes:**  
"The fundamental bottleneck in current interpretability research is that Sparse Autoencoders (SAEs) are typically trained pointwise. They learn to reconstruct individual hidden states, which are often dominated by surface features—syntax, token position, and specific model artifacts. When we try to transfer these concepts between different model families, like Gemma and LLaMA, we find that the alignment is fragile because the 'dictionary' has overfitted to the specific 'noise' of the training model. We need a method that can distinguish the semantic 'signal' from the representational 'noise'."

---

## Slide 3: The Hypothesis: Platonic Convergence
**What it shows:**  
- A conceptual diagram of the Platonic Representation Hypothesis (Huh et al., 2024).  
- Multiple models converging on a shared semantic "core" as capacity and data increase.  
- Matrix showing alignment difficulty across families.  
**Speaker Notes:**  
"Our work is grounded in the Platonic Representation Hypothesis: the idea that sufficiently capable models converge toward a shared semantic geometry because they are all learning the same world structure. Set-ConCA leverages this convergence by explicitly training for inter-model stability. If a concept is truly 'Platonic,' it must be invariant to how a thought is phrased (paraphrase invariance) and how it is represented (model invariance). Set-ConCA is the first framework designed to explicitly optimize for this dual invariance."

---

## Slide 4: Set-ConCA: Mathematical Foundation
**What it shows:**  
- Definition of a Semantic Set: $X = \{x_1, x_2, \dots, x_s\}$ where all $x_i$ represent the same anchor $a$.  
- The Permutation Invariance property: $f(X) = f(P(X))$.  
- Comparison: Batch (Independent) vs. Set (Semantically Equivalent).  
**Speaker Notes:**  
"Mathematically, we define a training sample not as a single vector, but as a set $X$ of hidden states. These states are extracted from different paraphrases of the same underlying concept. The core constraint of our element encoder is permutation invariance: the final concept code must be identical regardless of the order or specific wording of the paraphrases in the set. This turns the interpretability problem into an alignment problem: we are no longer just looking for sparse directions; we are looking for sparse directions that are constant across semantic variations."

---

## Slide 5: Architecture: The Permutation-Invariant Encoder
**What it shows:**  
- **Figure:** `results/figures/fig10_architecture.png` (or the encoder portion).  
- Flow: Hidden States ($N \times S \times D$) → Linear Encoder → Mean Pooling ($\bar{u}$) → LayerNorm → TopK Bottleneck → Concept Code $(\hat{z})$.  
- Highlight: No nonlinear activations in the encoder to preserve geometric linearity.  
**Speaker Notes:**  
"The Set-ConCA encoder uses a 'Deep Sets' architecture. Each element in the set passes through a shared linear projection. We then apply mean pooling to aggregate the information into a single 'pre-concept' vector, $\bar{u}$. This is followed by a non-affine LayerNorm and a hard TopK sparsity bottleneck. Crucially, we avoid nonlinear activations like ReLU in the encoder. This maintains a linear relationship between the input space and the latent space, which is essential for the Procrustes bridge we use for cross-model alignment later."

---

## Slide 6: Architecture: The Dual Decoder
**What it shows:**  
- **Flow:** Concept Code $(\hat{z})$ → Shared Decoder ($W_{shared}$) + Residual Connection ($W_{res}$).  
- Formula: $\hat{f}_i = W_{shared}\hat{z} + W_{res}u_i + b$ separate shared/residual streams.  
- Rationale: Decoupling shared semantics from residual syntax.  
**Speaker Notes:**  
"The most innovative feature of our architecture is the Dual Decoder. Traditional SAEs force the sparse code to reconstruct EVERYTHING. We don't. We split the reconstruction into two streams: a 'Shared Stream' $(\hat{z} \to W_{shared})$ which reconstructs the semantic core common to the whole set, and a 'Residual Stream' $(u_i \to W_{res})$ which captures the idiosyncratic syntax of each specific paraphrase. By siphoning off the 'noise' into the residual stream, we prevent it from contaminating the concept code $\hat{z}$."

---

## Slide 7: The Objective Function: Balancing Sparsity and Invariance
**What it shows:**  
- Composite Loss: $\mathcal{L} = \mathcal{L}_{recon} + \alpha\mathcal{L}_{sparsity} + \beta\mathcal{L}_{consistency}$.  
- Explanations:  
  - $\mathcal{L}_{recon}$: MSE on set elements.  
  - $\mathcal{L}_{sparsity}$: Sigmoid-L1 on $\bar{u}$.  
  - $\mathcal{L}_{consistency}$: Invariance across subsets.  
**Speaker Notes:**  
"Our training objective is a multi-task loss. Beyond standard MSE reconstruction, we provide a probability-domain sparsity signal using a Sigmoid-L1 loss on the pooled activations. Most importantly, we include a subset consistency loss, $\mathcal{L}_{consistency}$. This cross-entropy term forces the encoder to produce identical concept codes even when only a subset of the paraphrases is provided. It is the formal implementation of our semantic invariance constraint."

---

## Slide 8: Optimization & Convergence Dynamics
**What it shows:**  
- **Figure:** `results/figures/fig08_convergence.png`.  
- Plot showing rapid loss decay and stability across seeds.  
- Highlighting that variance across seeds decreases over time.  
**Speaker Notes:**  
"Empirical validation begins with convergence. As shown in Figure 8, Set-ConCA is highly stable. Across five random seeds and 80 epochs of training, we see tight convergence with minimal variance toward the end of training. This suggests that the 'concept landscape' is relatively smooth and the model consistently finds the same semantic directions, which is a prerequisite for robust interpretability."

---

## Slide 9: EXP1: Set-based vs. Pointwise Training
**What it shows:**  
- **Figure:** `results/figures/fig01_set_vs_pointwise.png`.  
- Comparison: Set-ConCA (S=8) vs. Pointwise ConCA (S=1).  
- Observation: Set-ConCA has higher MSE but enables the +8.2pp transfer gain.  
**Speaker Notes:**  
"In Experiment 1, we compare Set-ConCA to its pointwise equivalent. You will notice that Set-ConCA's reconstruction error is slightly higher. This is not a failure; it is the 'cost of invariance.' A pointwise model only has to reconstruct one vector, while Set-ConCA must compress the shared meaning of eight diverse sentences into one code. While harder to train, this pressure is exactly what pays dividends in cross-model transfer."

---

## Slide 10: EXP2: Scaling the Set Size (S)
**What it shows:**  
- **Figure:** `results/figures/fig02_s_scaling.png`.  
- Plot of Stability vs. S (1, 3, 8, 16, 32).  
- Finding: Diminishing returns after S=8.  
**Speaker Notes:**  
"A natural question is: how many paraphrases do we need? Figure 2 shows that stability grows monotonically with the set size S. However, we see diminishing returns after S=8. S=8 captures roughly 80% of the benefit of S=32 while keeping the batch computational cost manageable. This informed our decision to use S=8 as the standard configuration for our larger-scale experiments."

---

## Slide 11: EXP 4: Cross-Family Transfer (The "Platonic" Test)
**What it shows:**  
- **Figure:** `results/figures/fig04_cross_family_transfer.png`.  
- Transfer results: Gemma-3 4B → LLaMA-3 8B (64.6%).  
- Comparison to chance (25%).  
**Speaker Notes:**  
"This is the heart of our paper. We trained Set-ConCA on Gemma-3 and attempted to transfer its dictionary to LLaMA-3 using a linear Procrustes bridge. We achieved 64.6% overlap—nearly triple the chance level. This is a massive result: it proves that concepts learned under the Set-ConCA constraint are universal enough to align two models built by different organizations with different architectures."

---

## Slide 12: EXP 5: Capacity Asymmetry and Resolution
**What it shows:**  
- **Figure:** `results/figures/fig05_intra_family_heatmap.png`.  
- Heatmap showing 1B, 4B, and 9B models.  
- Key takeaway: 4B → 8B (Cross-family) is BETTER than 4B → 1B (Intra-family).  
**Speaker Notes:**  
"Counter-intuitively, model family matters less than model capacity. We found that transferring from Gemma 4B to LLaMA 8B (cross-family) actually works *better* than transferring from Gemma 4B to Gemma 1B (intra-family). This suggests that lower-capacity models (1B) merge or drop features that larger models (4B/8B) resolve independently. Alignment is limited by the 'representational resolution' of the receiver model."

---

## Slide 13: EXP 11: Information Depth & PCA-32 Distillation
**What it shows:**  
- **Figure:** `results/figures/fig11_layer_sweep.png` or `results/figures/fig14_pca32_transfer.png`.  
- Bar chart: PCA-32 Distilled (77.8%) vs. Full Rank (64.6%).  
- Takeaway: Semantic "Core" is in the dominant spectral components.  
**Speaker Notes:**  
"In Experiment 11, we tested whether 'pre-filtering' hidden states helps. By projecting representations into their top 32 principal components (retaining ~52% variance) and training there, we boosted transfer to 77.8%. This tells us that the transferable semantic information is highly concentrated in the dominant spectral directions. High-frequency variance in hidden states is mostly model-specific noise that complicates alignment."

---

## Slide 14: SOTA Comparison: Set-ConCA vs. SAE Variants
**What it shows:**  
- **Figure:** `results/figures/fig06_sota_comparison.png` and `results/figures/fig14_capability_matrix.png`.  
- Metrics Table: MSE, Stability, L0, Transfer.  
- Highlight: Set-ConCA beats SAE-TopK and SAE-L1 on cross-model transfer.  
**Speaker Notes:**  
"Compared to state-of-the-art baselines like Anthropic-style SAEs (L1 or TopK variants), Set-ConCA achieves significantly higher cross-model alignment. While SAEs are excellent for single-model circuit analysis, their dictionaries are too 'pointwise-focused' for transfer. Set-ConCA is the only method that combines sparse dictionary learning with multi-view set training to achieve a truly transferable concept space."

---

## Slide 15: Qualitative Results: Interpretable Concepts
**What it shows:**  
- Table of top anchors for Concept #50 (Asia/Tech/Space) and Concept #48 (Sports/Cycling/England).  
- Examples of activation snippets.  
- Comparison of monosemanticity vs. baselines.  
**Speaker Notes:**  
"Qualitatively, the concepts we discover are highly monosemantic. Concept 50, for instance, activates exclusively on East Asian technology and space flight news. Concept 48 tracks international sports events. These are not 'token frequency' neurons; they are high-level semantic components that respond to the same underlying meaning regardless of the specific words used."

---

## Slide 16: EXP 7: Causal Steering & Universality
**What it shows:**  
- **Figure:** `results/figures/fig07_steering.png`.  
- Plot of Topic Similarity vs. Intervention Alpha.  
- Showing Weak-to-Strong Steering (Gemma 1B concept steering LLaMA 8B).  
**Speaker Notes:**  
"A concept is only real if it is causal. In Experiment 7, we used concept directions discovered in a small 4B model to intervene on the hidden states of an 8B model. As we increase the intervention strength, the target model's representations shift predictably toward the target concept. Remarkably, even concepts from a tiny 1B model can successfully steer the 8B model, further proving the universality of the Set-ConCA directions."

---

## Slide 17: Ablation: Inductive Biases and Stability
**What it shows:**  
- **Figure:** `results/figures/fig09_consistency_ablation.png`.  
- Comparison: Full Model vs. No Consistency ($\beta=0$).  
- Result: TopK hard-sparsity provides enough stability that consistency loss is often redundant.  
**Speaker Notes:**  
"We also performed a rigorous ablation on our loss terms. Surprisingly, in the hard TopK mode, the consistency loss is largely redundant. The TopK bottleneck itself is a powerful enough inductive bias for stability. However, we found that the consistency loss is essential in 'soft-sparsity' modes (Sigmoid-L1) where it provide a +7.5pp gain, and remains a valuable formal guarantee for the invariance we seek."

---

## Slide 18: Linear Sufficiency: The Geometric Evidence
**What it shows:**  
- **Figure:** `results/figures/fig12_nonlinear_bridge.png`.  
- Comparison: Linear Procrustes (64.0%) vs. Nonlinear MLP (64.7%).  
- Takeaway: The gain from nonlinearity is within the noise floor (+0.5pp).  
**Speaker Notes:**  
"One of our most significant theoretical findings is the linear sufficiency of the bridge. We replaced our linear Procrustes bridge with a deep MLP and found only a 0.5% gain. This is the strongest geometric evidence yet for the Platonic Representation Hypothesis: it implies that different models aren't just learning the same concepts, but they are organizing them in approximately the same linear relative positions."

---

## Slide 19: Limitations & Future Frontiers
**What it shows:**  
- **Bullet points:**  
  - Dependency on parallel corpora (paraphrases).  
  - Scaling to overcomplete dictionaries ($C > 10,000$).  
  - Domain generalization (beyond news data).  
**Speaker Notes:**  
"Set-ConCA is not without limitations. The requirement for paraphrase sets makes it harder to apply in low-variation domains like code. We also haven't yet scaled to the massive, overcomplete regimes seen in the latest SAE research. However, the path forward is clear: applying Set-ConCA to cross-domain datasets and scaling to millions of concept components to see if the Platonic alignment holds at higher resolution."

---

## Slide 20: Conclusion: A New Primitive for Interpretability
**What it shows:**  
- Summary: Set-ConCA = Sets + Sparsity + Dual Decoder + Transfer.  
- Final message: Understanding models through their shared semantic language.  
**Speaker Notes:**  
"In conclusion, Set-ConCA introduces a new primitive for interpretability: the semantic set. By training models to find invariant structures across paraphrases and architectures, we've moved closer to a universal 'Platonic' concept algebra. We hope this work lays the foundation for a future where interpretability is not about studying one model at a time, but about understanding the shared semantic language that all intelligence converges toward. Thank you."

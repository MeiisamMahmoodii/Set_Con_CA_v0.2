# Set-ConCA: Executive Summary (NeurIPS 2025)

## 🎯 The Core Problem: Pointwise Conflation
*   **Standard SAEs are Pointwise**: They process individual hidden states. This forces them to encode both **semantics** (what it means) and **syntax/surface noise** (how it's said) into the same dictionary.
*   **The Conflation Penalty**: Because features are "flavored" by model-specific syntax, they do not transfer across different LLMs (e.g., Gemma features rarely map to Llama features).

## 🚀 Our Solution: Set-ConCA
*   **Representation Sets**: Instead of single vectors, we train on sets of **paraphrases**.
*   **Semantic Invariance**: By forcing the model to produce a *single* concept code for a set of different sentences, we mathematically average out the surface variation.
*   **Dual Decoder**: A novel architecture that splits reconstruction:
    1.  **Shared Stream**: Reconstructs the semantic core common to the set.
    2.  **Residual Stream**: Reconstructs the sentence-specific syntax.

## 🛠️ Implementation Choices
*   **Architecture**:
    *   **Linear Encoder**: (B, S, D) -> (B, S, C). No activation (preserving geometric linearity).
    *   **Mean-Pool Aggregator**: Permutation-invariant pooling across the paraphrase set.
    *   **TopK Sparsity**: Hard k=32 bottleneck for maximum interpretability and zero hyperparameter tuning for L1.
*   **Training**:
    *   **Datasets**: Gemma-3 (1B/4B), Gemma-2 (9B), LLaMA-3 (8B).
    *   **Anchors**: 2,048 diverse news-based semantic anchors.
    *   **Loss**: Mean Squared Error (MSE) + Subset Consistency Loss.

## 📊 Key Experimental Results
*   **Cross-Model Transfer**: Set-ConCA achieves **64.6%** concept overlap (Gemma -> Llama) vs **56.4%** for standard SAEs (**+8.2pp gain**).
*   **PCA-32 Breakthrough (EXP 14)**: Pre-distilling hidden states to the top 32 principal components yields **77.8% transfer**—confirming the most universal semantics live in the dominant spectral directions.
*   **Weak-to-Strong Steering (EXP 7)**: Concepts discovered in a 1B model are **more causal** when steering an 8B model than concepts from a 4B model (+3.0pp gain), indicating "semantic distillation" in smaller models.
*   **Linear Sufficiency (EXP 12)**: A nonlinear MLP bridge provides only a **+0.5pp** gain over an Orthogonal Procrustes rotation, proving the **Platonic Representation Hypothesis**: different models share a nearly linear conceptual geometry.

## 🆚 SOTA Comparison
*   **RAVEL/Patchscopes**: We provide the **sparse discovery mechanism** that explains why these benchmarks/inspection tools work.
*   **SAE-TopK**: We maintain similar single-model interpretability but provide a significant advantage in **causal transferability**.
*   **RepE**: We are **unsupervised**; we discover the concepts that RepE requires manual labeling to find.

---
*Results based on 2,048 anchors, 3 seeds, and 50 epochs. All experiments validated on RTX 3090 GPU.*

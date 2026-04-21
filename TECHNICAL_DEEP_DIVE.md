# Set-ConCA Technical Deep Dive (v0.2)

This document provides a component-level breakdown of the Set-ConCA architecture, specifically detailing the breakthroughs and optimizations implemented in the v0.2 scaling phase.

---

## 1. The Element Encoder (`encoder.py`)
- **Mathematics:** $u_i = W_e f(x_i) + b_e$.
- **Design:** We use a **Linear Encoder** without non-linear clipping (ReLU). This allows the model to preserve the magnitude of the log-posterior estimates, which is critical for the distributional aggregation phase.
- **Scaling:** In v0.2, the layer dimension scales dynamically (e.g., 2304 for Gemma-2-2B to 4096 for Llama-3-8B).

---

## 2. Advanced Aggregation (`aggregator.py`)
This is where the "Set" logic happens. We have two modes:
- **Mean Pooling (Baseline):** $\hat{z}_X = \text{LayerNorm}(\frac{1}{S} \sum u_i)$.
- **Attention Aggregation (v0.2):** Uses a learnable query to dynamically weight elements. This allows the model to "filter" out elements in a set that are semantically weak or noisy, focusing the concept on the strongest semantic anchors.

---

## 3. The Fidelity Breakthrough: Top-K Activation (`setconca.py`)
The most significant discovery of v0.2 was moving from Sigmoid/L1 sparsity to **Top-K activations**.
- **Motivation:** Sigmoid/L1 causes "activation shrinkage," where the decoder receives a weakened signal, leading to high MSE.
- **Implementation:** We take the top 50 concept activations and zero out the rest.
- **Impact:** This improved reconstruction fidelity by **63%** on the Gemma baseline while preserving the semantic stability of the extracted concepts.

---

## 4. The Dual Decoder (`decoder.py`)
The decoder splits the reconstruction into two paths:
1.  **Shared Path (Concept):** $\hat{y}_{shared} = W_d^{(s)} \hat{z}_X$. This reconstructs the "Thematic core" of the neighborhood.
2.  **Residual Path (Instance):** $\hat{y}_{res} = W_d^{(r)} u_i$. This reconstructs the specific linguistic details of each element.
- **Goal:** By separating these, we force $z_X$ to ignore the "Residual" noise (grammatical variations, specific word choices) and capture only the "Shared" knowledge.

---

## 5. Composite Learning Objective (`compute_loss`)
We optimize three factors simultaneously:
1.  **MSE:** Reconstruction accuracy (boosted by Top-K).
2.  **Sparsity:** Ensuring only a few "True Concepts" fire.
3.  **Subset Consistency ($\beta$):** Enforcing that any random subset of the neighborhood identifies the *same* Top-K concepts. This makes our concepts 4.8x more stable than point-wise analysis.

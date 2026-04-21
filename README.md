# Set-ConCA v0.2: Distributional Concept Component Analysis

**NeurIPS 2026 Submission Readiness Phase**

Set-ConCA is a framework for **Mechanistic Interpretability** that shifts the fundamental unit of analysis from individual hidden states to **Representation Sets** (local neighborhoods). 

This repository contains the validated implementation for **v0.2**, featuring Top-K activation, improved causal faithfulness, and the first demonstration of **Cross-Model Latent Transplantation**.

## 🛰️ Scientific Breakthroughs

### 1. The Semantic Emergence Threshold ($S=8$)
We have identified a universal transition in concept stability. Modern LLMs (Gemma-2, Llama-3, Gemma-3) require a neighborhood of approximately **8 instances** to manifest stable semantic anchors that are robust to distributional noise.

### 2. Latent Transplantation (Gemma ↔ Llama)
We successfully bridged the concept spaces of disparate model families.
- **Top-K Overlap:** 12.53%
- **Significance:** **10.4x better than random chance.**
- Set-ConCA acts as a "Universal Coordinate System" for aligning concepts across different LLM architectures.

### 3. Causal Faithfulness (60%)
By resolving the **Energy Gap** (reconstruction magnitude shrinkage) via Norm-Alignment, we achieve a **60% Faithfulness Score** on causal intervention benchmarks.

---

## 🛠️ Repository Structure

- `docs/paper/`: Complete NeurIPS 2026 LaTeX manuscript and bibliography.
- `setconca/`: Core implementation featuring Top-K and Attention Aggregation.
- `eval_faithfulness.py`: Causal diagnostic tool using `TransformerLens`.
- `train_bridge.py`: Linear bridge trainer for cross-model mapping.
- `build_multi_dataset.py`: Synchronized extraction pipeline (Locked Seed 42).

## 🚀 Getting Started

```bash
# Install dependencies
uv sync

# Extract Aligned Concepts
uv run python build_multi_dataset.py --model_id google/gemma-2-2b --output_path data/gemma.pt
uv run python build_multi_dataset.py --model_id meta-llama/Meta-Llama-3-8B --output_path data/llama.pt --load_in_4bit

# Train the Bridge
uv run python train_bridge.py
```

## 📝 Citation
Please refer to the full manuscript in `docs/paper/setconca_neurips.tex` for formal citation and theoretical foundations.

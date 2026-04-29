# Set-ConCA: Concept Component Analysis on Representation Sets

**MECHANISTIC INTERPRETABILITY | CROSS-MODEL ALIGNMENT | NEURIPS 2025**

Set-ConCA discovers concept components that are stable across **sets** of representations (paraphrases, trajectory steps, local neighborhoods) rather than processing a single vector at a time. In the latest verified rerun it shows strong cross-family transfer and causal steering, while some earlier stronger claims have been narrowed to match current evidence.

![SOTA Comparison](./results/figures/fig06_sota_comparison.png)
*Set-ConCA lead over SAE variants in reconstruction-transfer Pareto frontier.*

---

## 🚀 Headline Results
*   **Cross-Model Transfer**: Set-ConCA achieves **69.5% +/- 0.6pp** concept overlap from Gemma-3 4B to LLaMA-3 8B.
*   **Causal Steering**: Set-ConCA gains **+9.8pp** at `alpha=10`, and weak-to-strong steering from 1B to 8B gains **+10.7pp**.
*   **Linear Bridge Wins**: Linear bridge reaches **69.3%**, while nonlinear MLP falls to **64.2%** in the latest rerun.
*   **PCA-32 Does Not Help Here**: direct PCA-32 distilled-input transfer falls to **31.4% +/- 1.3pp**, below the full-rank baseline.
*   **Real EN/FR Benchmark Exists**: WMT14 `fr-en` tensors and matrix results now exist for `Qwen2.5-3B`, `Qwen2.5-7B`, `Mistral-7B`, and `Gemma-2-2B`; Set-ConCA averages **0.3187** raw overlap on the completed multilingual matrix and should be framed as competitive, not dominant, there.

---

## 🏛️ Architecture

```
x (B, S, D)  # Batch of S paraphrases
  └─ ElementEncoder   → u  (B, S, C)   linear, shared across elements
  └─ SetAggregator    → z  (B, C)      mean-pool [distributional summary]
  └─ LayerNorm        → ẑ  (B, C)      concept normalization
  └─ DualDecoder      → f̂  (B, S, D)   W_shared(ẑ) + W_residual(u)
```

**Innovation: The Dual Decoder**
Standard autoencoders conflate syntax and semantics. Our dual streams formally separate the **invariant semantic core** (shared stream) from the **paraphrase-specific syntax** (residual stream).

---

## 📊 SOTA Comparison

Set-ConCA remains competitive on sparse reconstruction and strong on cross-family transfer/steering, but raw transfer overlap is not universally better than pointwise TopK baselines.

| Feature | SAE-TopK | Gated SAE | **Set-ConCA (Ours)** |
| :--- | :--- | :--- | :--- |
| **Input Signal** | Pointwise | Pointwise | **Distributional Set** |
| **Sparsity** | Hard TopK | Soft (Gated) | **Hard TopK** |
| **Invariance** | None | None | **Permutation-Invariant** |
| **Discovery** | Surface-heavy | Surface-heavy | **Deep Semantic** |
| **Cross-Model Transfer** | 78.4% raw overlap | Low | **69.5% cross-family transfer** |
| **RAVEL Disentanglement** | Baseline | Moderate | **Optimal** |

---

## 🧪 Experimental Suite (EXPs 1-16)

### Key Experiments & Findings
*   **EXP 1-3 (Architecture)**: Verified the current reconstruction/stability trade-offs across set size and aggregator variants.
*   **EXP 4-5 (Alignment)**: Established a verified baseline of 69.5% transfer between Gemma-3 4B and LLaMA-3 8B.
*   **EXP 7 (Steering)**: Demonstrated "Weak-to-Strong" steering where concepts from a 1B model successfully manipulate the behavior of an 8B model.
*   **EXP 11 (Information Sweep)**: PCA-rank proxy peaks near low/intermediate rank in the proxy experiment.
*   **EXP 12 (Linear Bridge)**: Confirmed that linear alignment outperforms the nonlinear bridge in the current rerun.
*   **EXP 16 (TopK Pointwise vs Set)**: Pointwise SAE-TopK achieves higher raw overlap (78.4%) than Set-ConCA (69.5%) in the current setup.

---

## 🛠️ Installation & Usage

```bash
# Clone and install dependencies
git clone https://github.com/MPC/SetConCA.git
cd SetConCA
uv sync
```

### Run Evaluation Suite (GPU recommended)
```bash
# High-fidelity evaluation (2,048 anchors, 5 seeds, GPU)
uv run python experiments/neurips/run_evaluation_v2.py

# Build multilingual EN/FR benchmark tensors
uv run python experiments/neurips/build_multilingual_benchmarks.py

# Evaluate the multilingual model matrix
uv run python experiments/neurips/run_benchmark_matrix.py wmt14_fr_en
```

### Quick Training
```bash
uv run python train.py --use_topk --k 32 --epochs 50
```

---

## 📂 Project Structure
*   `setconca/`: Core model implementation (Encoder, Aggregator, Dual-Decoder).
*   `experiments/neurips/`: Activation extraction pipeline and all 16 experiments.
*   `results/`: Detailed `REPORT.md`, `EXECUTIVE_SUMMARY.md`, and all visualization charts.
*   `tests/`: 60 tests including claim-level validation gates for result/report consistency.

---

## 📜 Citation
*Manuscript currently under review for NeurIPS 2025. Preprint coming soon.*

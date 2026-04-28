# Set-ConCA: Full Research Evaluation Plan

## What we have (real data)

| File | Shape | Meaning |
|---|---|---|
| `gemma3_1b_dataset.pt` | (2048, 32, 1152) | 2048 news anchors × 32 paraphrase variations, last-layer hidden states from Gemma-3 1B |
| `gemma3_4b_dataset.pt` | (2048, 32, 2560) | Same anchors, Gemma-3 4B |
| `gemma_9b_dataset.pt` | (2048, 32, 3584) | Same anchors, Gemma-2 9B |
| `llama_8b_dataset.pt` | (2048, 32, 4096) | Same anchors, LLaMA-3 8B |
| `gemma_s{1,3,8,16,32}_subset.pt` | (2048, S, 2304) | Gemma-2 2B at varying set sizes (S-scaling sweep) |
| `hf_real_dataset.pt` | (2048, 8, 2560) | Set-format (S=8) for direct comparison |
| `hf_point_dataset.pt` | (16384, 1, 2560) | Pointwise (S=1) for ConCA baseline |

**The texts**: 2,048 news article topics, each with 32 semantic paraphrases = "representation sets" (what makes Set-ConCA work).

---

## Baselines to compare against

| Baseline | What it is | Why we compare |
|---|---|---|
| **ConCA / Pointwise** | Standard concept analysis on single vectors (S=1) | The direct predecessor — same idea without sets |
| **PCA** | Classical principal components | Linear, unsupervised, gold standard for linear structure |
| **ICA** | Independent Component Analysis | Non-Gaussian, finds statistically independent components |
| **SAE** (Sparse Autoencoder) | Anthropic/EleutherAI style single-vector sparse AE | Current SOTA for concept discovery in LLMs |
| **Random** | Random projection to concept space | Lower bound |

---

## Experiments (5 core + 3 new)

### Core 5 (from the paper)
1. **Set vs Pointwise** — MSE reconstruction + concept stability (Top-K overlap)
2. **S-Scaling** — How quality changes as set size S grows (S=1,3,8,16,32)
3. **Aggregator Ablation** — mean vs attention vs random pooling
4. **Cross-Model Transfer** (Cross-family alignment) — Gemma-3 → LLaMA-3 concept bridge
5. **Interventional Steering** — Apply concept vectors to steer representations

### New experiments
6. **Intra-family alignment** — Gemma-3 1B ↔ 4B ↔ 9B (same family, different sizes)
7. **SOTA comparison** — Set-ConCA vs PCA/ICA/SAE on reconstruction + stability + disentanglement
8. **Concept interpretability probe** — Do Set-ConCA concepts align with semantic categories?

---

## Metrics

| Metric | Symbol | What it measures |
|---|---|---|
| Reconstruction MSE | MSE | Does the model faithfully recover the original hidden states? |
| Top-K Overlap | J@K | Concept stability across seeds (same data → same active concepts?) |
| Centered Kernel Alignment | CKA | How similar are two concept spaces geometrically? |
| L0 sparsity | L0 | Average number of active concepts per sample |
| Explained Variance | R² | How much variance is captured vs discarded |
| Transfer Overlap | TO | After bridging two models' concept spaces, how well do concepts transfer? |
| Cosine Effect | Δcos | Direction of shift after interventional steering |

---

## Implementation plan

### Step 1: Write `experiments/neurips/run_evaluation.py`
Single script with `--exp` flag covering all experiments. Sections:
- Loads appropriate real `.pt` files
- Trains Set-ConCA with reproducible seeds
- Implements all baselines (PCA/ICA inline with sklearn, SAE as a simple baseline)
- Computes all metrics
- Saves results to `results/` as JSON

### Step 2: Write `experiments/neurips/plot_results.py`  
Reads `results/` JSON, produces all figures:
- Figure 1: MSE comparison (bar chart, all methods)
- Figure 2: Stability (Top-K overlap, bar)
- Figure 3: S-scaling curve (line plot, MSE + stability vs S)
- Figure 4: Aggregator ablation (grouped bar)
- Figure 5: Cross-family transfer (sankey/heatmap style)
- Figure 6: Intra-family CKA matrix (heatmap)
- Figure 7: Interventional steering (before/after cosine similarity)
- Figure 8: SOTA comparison radar chart

### Step 3: Write the full report  
`REPORT.md` — structured as a research paper with:
- Abstract
- Background (what is a set, latent space, concept, alignment, steering)
- Architecture + every design decision with rationale
- Experiments + results
- Reviewer Q&A (20+ questions answered)
- Glossary

---

## Runtime estimate
All experiments on CPU (RTX 3090 optional), subsampled to 512 anchors for speed:
- Training Set-ConCA (per model): ~2 min
- PCA/ICA baselines: instant
- SAE baseline: ~1 min
- Total: ~20 min

---

## Verification
- All 52 unit tests still pass after adding new code
- Plots render and save correctly
- Report is self-contained and reviewable

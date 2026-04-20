# Set-ConCA

### Set-based Concept Component Analysis on Representation Sets

Set-ConCA is a principled extension of Sparse Concept Component Analysis (ConCA) that estimates concept representations from **local representation sets** (paraphrases, local neighbourhoods, or reasoning trajectories) rather than isolated point representations.

By leveraging distributional information, Set-ConCA isolates robust semantic concepts while separating them from instance-specific linguistic noise.

---

## 🚀 Key Innovations

- **Log-Posterior Preservation**: Purely linear encoder architecture that preserves the mathematical interpretation of latent log-posteriors.
- **Set-Level Inference**: Aggregates information across related representations using permutation-invariant mean pooling.
- **Shared + Residual Decoding**: A dual-headed decoder that decomposes hidden states into a shared "Concept" component and an instance-specific "Residual" component.
- **Probability-Domain Sparsity**: Imposes sparsity in the probability space (Sigmoid domain) for superior numerical stability and interpretability.
- **Subset Consistency Regularization**: Enforces that any random subset of a representation set must yield the same core concept, significantly improving concept stability.

---

## 🏗 Architecture

The Set-ConCA architecture follows the mathematical framework introduced in Liu (March 2026):

1.  **Element Encoder**: $u_i = W_e f(x_i) + b_e$ (No non-linear clipping).
2.  **Aggregation**: $\hat{z}_X = \text{LayerNorm}(\text{mean}(u_i))$ (Affine-free).
3.  **Dual Decoder**: $\hat{f}(x_i) = W_d^{(s)} \hat{z}_X + W_d^{(r)} u_i + b_d$.

---

## 📂 Project Structure

```text
setconca/
├── model/            # Core architecture (Encoder, Aggregator, Decoder)
├── losses/           # Specialized losses (Sparsity, Subset Consistency)
├── data/             # Dataset and Loader utilities
├── train.py          # Main training loop with W&B logging
├── build_hf_dataset.py # Large-scale latent extraction from HuggingFace models
├── docs/             # Mathematical documentation and implementation logs
└── tests/            # Automated test suite (99%+ coverage)
```

---

## 🛠 Installation

This project uses `uv` for lightning-fast dependency management and reproducibility.

```bash
# Initialize and install dependencies
uv sync

# Run the test suite
uv run pytest tests/ -v
```

---

## 🏃 Usage

### 1. Extract Latents
Use a local LLM (like Phi-2 or Gemma-2b) to extract hidden states from a text dataset:
```bash
uv run python build_hf_dataset.py
```

### 2. Train the Model
Train the Set-ConCA autoencoder on the extracted latents:
```bash
uv run python train.py --data_path data/hf_real_dataset.pt --hidden_dim 2560 --concept_dim 4096
```

---

## 🔬 Reproducibility & Documentation

Detailed implementation logs for every phase of the project can be found in the `docs/` directory:
- [Phase 2: Element Encoder](docs/phase_2.md)
- [Phase 4: Shared/Residual Decoder](docs/phase_4.md)
- [Phase 6: Subset Consistency](docs/phase_6.md)

---

> [!NOTE]
> Based on the Set-ConCA implementation framework (April 2026). Verified with 99.1% code coverage and stable convergence on real LLM latent sets.

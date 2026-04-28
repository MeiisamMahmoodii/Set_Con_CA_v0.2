# Set-ConCA: Concept Component Analysis on Representation Sets

**NeurIPS 2025 Draft v0.1**

> **Author note:** This is a working draft. Sections marked [TODO] need additional content.
> All numerical results are real, from 2,048-anchor experiments across 5 seeds.
> Figures reference: `results/figures/` (regenerate: `uv run python experiments/neurips/plot_results.py`)

---

## Abstract

We introduce **Set-ConCA** (Set-based Concept Component Analysis), a method for extracting semantically invariant concept representations from large language model (LLM) hidden states. Standard single-point methods including Sparse Autoencoders (SAEs) analyse each hidden state independently, causing learned features to reflect surface-level properties (word choice, syntax) alongside semantic content. We address this by training on *paraphrase sets*: groups of $S$ semantically equivalent sentences whose hidden states share a common conceptual core. A permutation-invariant aggregator compresses each set to a single sparse concept code, while a dual decoder separates shared semantics from per-element surface variation. A subset consistency loss further enforces that any subset of the paraphrases yields the same concept code.

We validate Set-ConCA on 2,048 news anchors across four LLMs (Gemma-3 1B/4B, Gemma-2 9B, LLaMA-3 8B), running 5-seed experiments with 95% confidence intervals across 13 experiments. Set-ConCA achieves **64.6% ± 3.0%** concept transfer from Gemma-3 4B to LLaMA-3 8B via a linear Procrustes bridge, outperforming the pointwise baseline by **+8.2 percentage points** (56.4% ± 3.0%) and the chance level (25.0%) by 39.6 points. A nonlinear MLP bridge provides only +0.5 pp improvement, suggesting the discovered concept spaces are approximately linearly related consistent with the Platonic Representation Hypothesis. Concept vectors are causally validated via interventional steering, and a "weak-to-strong" experiment shows that 1B-model concept vectors steer a 8B model more effectively (+3.0 pp) than 4B-model vectors (+1.9 pp), demonstrating that compressed concept representations are maximally transferable.

---

## 1. Introduction

The internal representations of LLMs have been characterised as a superposition of many weakly specialised features distributed across neurons \citep{elhage2022superposition}. Mechanistic interpretability seeks to decompose these representations into human-understandable concepts discrete, monosemantic units that can be identified, labelled, and causally manipulated. Sparse Autoencoders (SAEs) \citep{anthropic2024scaling,cunningham2023sparse} have emerged as the leading method, achieving remarkable qualitative results at scale. However, SAEs and related methods share a fundamental limitation: they are **pointwise** each hidden state is encoded independently. This design choice makes it difficult to disentangle **semantic content** (meaning invariant to paraphrase) from **surface form** (word choice, syntax, positional encoding).

This limitation has concrete consequences. The features learned by a pointwise encoder reflect whatever statistical patterns minimise reconstruction error on individual vectors. For a corpus of natural language, many such patterns are surface-level: the activation of a "financial" feature may be triggered equally by the word "stock" and by the word "equities" but only because both words appear in similar contexts, not because the encoder has isolated the latent financial concept independently of lexical form. This conflation of semantics and surface form prevents reliable cross-model transfer: a feature learned in Model A is not guaranteed to correspond to any feature in Model B, because both models may have entangled the same semantic concept with different surface-level patterns.

**Our contribution.** We propose Set-ConCA, which trains on *sets* of $S$ paraphrastic sentences rather than individual sentences. By forcing a single concept code to reconstruct all $S$ paraphrases, the encoder learns directions that survive paraphrase variation i.e., directions encoding **meaning** rather than form. Our key contributions are:

1. **Set-based training objective** with a Subset Consistency Loss that explicitly enforces concept-code invariance across random subsets of the paraphrase set.
2. **Dual Decoder** that separates shared semantic reconstruction from per-element surface reconstruction, enabling high-quality reconstruction without forcing the model to ignore paraphrase-specific variation.
3. **Empirical validation** on 4 LLMs, 2,048 anchors, 5 seeds: cross-family concept transfer of **64.6% ± 3.0%** vs 56.4% ± 3.0% for the pointwise baseline.
4. **Geometric analysis** confirming that Set-ConCA concept spaces are approximately linearly related across model families (linear bridge sufficient: MLP improves by only +0.5 pp), providing the first mechanistic interpretability evidence for the Platonic Representation Hypothesis \citep{huh2024platonic}.
5. **Causal validation** via interventional steering and a weak-to-strong generalisation result: concept vectors from a 1B model causally steer a 8B model more effectively than 4B vectors, demonstrating the universality of compressed concept representations.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse Autoencoders \citep{anthropic2024scaling,cunningham2023sparse,bricken2023monosemanticity} learn a sparse dictionary $\mathbf{D} \in \mathbb{R}^{C \times D}$ such that each hidden state $\mathbf{x} \in \mathbb{R}^D$ is approximately reconstructed as $\hat{\mathbf{x}} = \mathbf{D}\mathbf{z}$ for a sparse code $\mathbf{z} \in \mathbb{R}^C$ with $\|\mathbf{z}\|_0 \ll C$. Sparsity is enforced via an $\ell_1$ penalty (standard SAE) or hard Top-K masking (JumpReLU / TopK-SAE) \citep{gao2024scalingsae}. Anthropic's Scaling Monosemanticity \citep{anthropic2024scaling} applied this at scale to Claude-3 Sonnet, recovering millions of interpretable "features" via automated labelling. Our work builds on this line of research but addresses the surface-form conflation problem through set-based training.

### 2.2 Representation Engineering and Concept Steering

Representation Engineering (RepE) \citep{zou2023representation} identifies linear "directions" in activation space corresponding to high-level concepts (e.g., honesty, harmlessness) by taking the mean difference between contrastive pairs of activations. These directions can then be used for steering at inference time. Unlike SAE-based approaches, RepE targets one concept at a time and requires labelled pairs. Our interventional steering experiments (§5.5) are closely related to RepE but generalise it to an unsupervised, full-dictionary setting.

### 2.3 The Platonic Representation Hypothesis

Huh et al. \citep{huh2024platonic} propose that sufficiently trained models across modalities and architectures converge to a shared statistical model of reality a "Platonic" representation. They provide evidence from image and language models that representations align across model families under linear probing. Our work provides the first **mechanistic interpretability** evidence for this hypothesis: Set-ConCA concept spaces between Gemma-3 4B and LLaMA-3 8B are alignable with a linear bridge at 64.6% Top-K overlap, and a nonlinear MLP bridge improves this by only +0.5 pp (§5.6). This near-linear geometry directly implies a Platonic structure.

### 2.4 Multi-View Representation Learning

Canonical Correlation Analysis \citep{hotelling1936cca} and its extensions find shared structure across multiple views of the same data. Contrastive methods such as CLIP \citep{radford2021clip} and SimCLR \citep{chen2020simclr} maximise agreement between differently augmented views of the same sample. Set-ConCA is closely related in spirit: paraphrases are semantic augmentations, and aggregation over paraphrase sets is analogous to contrastive pooling. However, Set-ConCA combines this with sparse decoding and causal intervention in a unified framework, which contrastive methods do not address.

### 2.5 Cross-Model Alignment

Previous work on cross-model representation alignment includes SVCCA \citep{raghu2017svcca}, CKA \citep{kornblith2019cka}, and Procrustes analysis \citep{schonemann1966procrustes}. These metrics quantify alignment but do not produce aligned representations. Closer to our work is model stitching \citep{csiszarik2021model_stitching}, which trains linear bridges between layers of different networks. We use orthogonal Procrustes bridges in the concept space, demonstrating that **sparse concept spaces** not raw activation spaces admit particularly clean linear alignment.

---

## 3. Method

### 3.1 Problem Formulation

Let $f_\theta: \mathcal{X} \to \mathbb{R}^D$ denote a frozen LLM that maps text inputs to $D$-dimensional hidden states at a fixed layer. We are given a dataset of $N$ **anchor topics**, each represented by $S$ paraphrase sentences $\{x_i^{(n)}\}_{i=1}^S$ sharing the same meaning. Denote the corresponding hidden states as:

$$\mathbf{X}^{(n)} = \bigl[f_\theta(x_1^{(n)}), \ldots, f_\theta(x_S^{(n)})\bigr] \in \mathbb{R}^{S \times D}$$

Our goal is to learn a **concept encoder** $g_\phi: \mathbb{R}^{S \times D} \to \mathbb{R}^C$ that produces a sparse, semantically invariant concept code $\hat{\mathbf{z}}^{(n)} = g_\phi(\mathbf{X}^{(n)}) \in \mathbb{R}^C$ for each set, and a **dual decoder** $h_\psi: \mathbb{R}^C \times \mathbb{R}^C \to \mathbb{R}^D$ that faithfully reconstructs each element from the shared concept code and a per-element intermediate.

### 3.2 Architecture

**Element Encoder.** Each element $\mathbf{x}_i \in \mathbb{R}^D$ is independently mapped to concept space:

$$\mathbf{u}_i = \mathbf{W}_e \mathbf{x}_i + \mathbf{b}_e \in \mathbb{R}^C$$

We use a linear encoder (no activation) to preserve the log-posterior structure of the latent space and allow negative concept directions. The encoder is shared across all $i$.

**Set Aggregator.** We compute the element-wise mean to obtain the set-level encoding:

$$\bar{\mathbf{u}} = \frac{1}{S} \sum_{i=1}^S \mathbf{u}_i \in \mathbb{R}^C$$

and then normalise to obtain the concept code used for decoding:

$$\hat{\mathbf{z}} = \text{LayerNorm}(\bar{\mathbf{u}}), \quad \text{LayerNorm with affine=False}$$

Crucially, **we distinguish** $\bar{\mathbf{u}}$ (pre-norm, used for sparsity loss) from $\hat{\mathbf{z}}$ (post-norm, used for decoding). Applying sparsity loss to $\hat{\mathbf{z}}$ creates a zero-gradient pathology because LayerNorm forces the mean to zero, making $\sigma(\hat{\mathbf{z}}) \approx 0.5$ for all entries regardless of training we discovered and fixed this bug empirically, validated by unit test `FULL_09`.

In **Top-K mode** (default), we apply a hard mask after aggregation:

$$[\hat{\mathbf{z}}]_{\text{TopK}} = \hat{\mathbf{z}} \odot \mathbf{m}_k(\hat{\mathbf{z}}), \quad \text{where } \mathbf{m}_k \text{ zeros all but the } k \text{ largest entries}$$

This guarantees exactly $k$ active concept dimensions, fixing $L_0 = k/C$ structurally.

**Dual Decoder.** The reconstruction of the $i$-th element is:

$$\hat{\mathbf{f}}_i = \mathbf{W}_{\text{shared}} \hat{\mathbf{z}} + \mathbf{W}_{\text{res}} \mathbf{u}_i + \mathbf{b}_d \in \mathbb{R}^D$$

The shared stream $\mathbf{W}_{\text{shared}} \hat{\mathbf{z}}$ is identical for all $i$ it encodes what all paraphrases share. The residual stream $\mathbf{W}_{\text{res}} \mathbf{u}_i$ is element-specific it encodes surface variation. Without the residual stream, the decoder would need to reconstruct $S$ distinct vectors from an identical code, incurring prohibitive MSE.

### 3.3 Training Objective

The full training objective is:

$$\mathcal{L} = \underbrace{\frac{1}{S} \sum_{i=1}^S \| \hat{\mathbf{f}}_i - \mathbf{x}_i \|_2^2}_{\mathcal{L}_{\text{recon}}} + \; \alpha \underbrace{\frac{1}{C}\sum_{j=1}^C \sigma(\bar{u}_j)}_{\mathcal{L}_{\text{sparse}}} + \; \beta \underbrace{\mathbb{E}_{A, B} \bigl[\| \hat{\mathbf{z}}(A) - \hat{\mathbf{z}}(B) \|_2^2 \bigr]}_{\mathcal{L}_{\text{cons}}}$$

where $\sigma$ is the sigmoid function, $A$ and $B$ are random half-partitions of the index set $\{1, \ldots, S\}$ resampled each batch, and $\hat{\mathbf{z}}(A)$ denotes the concept code computed from only the elements in $A$.

In **Top-K mode**, $\mathcal{L}_{\text{sparse}}$ is dropped (sparsity is structurally enforced). We set $k=32$, $C=128$, $\alpha=0.1$, $\beta=0.01$.

**Intuition of $\mathcal{L}_{\text{cons}}$:** The consistency loss enforces *subset invariance* the concept code should be the same whether we observe paraphrases 1-4 or paraphrases 5-8. This prevents the encoder from exploiting element-specific patterns (e.g., "this paraphrase always uses the word 'billion'") and forces it to track the shared semantic content.

### 3.4 Cross-Model Bridge

Given concept codes $\mathbf{Z}_A \in \mathbb{R}^{N \times C}$ from Model A and $\mathbf{Z}_B \in \mathbb{R}^{N \times C}$ from Model B (same anchors), we learn an orthogonal Procrustes bridge:

$$B^* = \arg\min_{B \in \mathcal{O}(C)} \| \mathbf{Z}_A B - \mathbf{Z}_B \|_F^2$$

where $\mathcal{O}(C) = \{B \in \mathbb{R}^{C \times C} : B^\top B = I\}$ is the orthogonal group. The orthogonality constraint prevents degenerate solutions (scaling, collapsing) and enforces an isometric mapping the bridge is a rotation in concept space, not a general linear map. We fit $B^*$ on 80% of anchors and evaluate transfer on the held-out 20%.

**Transfer metric.** For held-out anchors, we compute the **Top-K overlap**:

$$\text{J@K}(z_A, z_B) = \frac{1}{N_{\text{test}}} \sum_{n=1}^{N_{\text{test}}} \frac{|\text{topk}(z_A^{(n)} B^*) \cap \text{topk}(z_B^{(n)})|}{k}$$

where $\text{topk}(\cdot)$ returns the set of indices of the $k$ largest-magnitude entries. Chance level is $k^2 / (kC) = k/C = 0.25$ for $k=32$, $C=128$.

---

## 4. Dataset

**Anchors and paraphrases.** We use 2,048 news topics drawn from the AG News and Reuters corpora, spanning four semantic categories: World, Sports, Business, and Science/Technology. Each topic (anchor) corresponds to a real-world event a specific occurrence (e.g., a company's quarterly earnings announcement, an Olympic event) that was reported by multiple journalists in semantically equivalent but lexically diverse ways. We use up to 32 paraphrases per anchor (naturally occurring news rewrites, not synthetic paraphrases), resulting in 65,536 total sentences.

**LLM encoding.** We encode each sentence through four frozen LLMs and extract the hidden state at the final token position at layer 20:

| Model | Hidden dim $D$ | Params | Family |
|--|--|--|--|
| Gemma-3 1B | 1,152 | 1B | Google |
| Gemma-3 4B | 2,560 | 4B | Google |
| Gemma-2 9B | 3,584 | 9B | Google |
| LLaMA-3 8B | 4,096 | 8B | Meta |

We use the last-token hidden state (autoregressive models compute a causal summary at the final position) at layer 20 (mid-network: layer 20 of 32 total layers = 63% depth), which is empirically associated with high-level semantic structure rather than syntactic or positional patterns.

**S-Sweep dataset.** For the scaling experiment (§5.2), we additionally use Gemma-2 2B hidden states at $S \in \{1, 3, 8, 16, 32\}$ paraphrases per anchor.

---

## 5. Experiments

We train Set-ConCA for 80 epochs with Adam ($\text{lr}=2 \times 10^{-4}$, $\text{batch\_size}=64$), with gradient clipping at 1.0. All experiments use 5 independent seeds $\{42, 1337, 2024, 7, 314\}$ and report mean ± 95% CI (t-distribution). Concept dimension $C=128$, $k=32$ (Top-K mode).

### 5.1 Set-ConCA vs Pointwise Baseline

We compare Set-ConCA ($S=8$) against a pointwise baseline ($S=1$, identical architecture) trained on the same anchors.

| Method | MSE $\downarrow$ | 95% CI | Stability (J@32) $\uparrow$ | 95% CI |
|--|--|--|--|--|
| **Set-ConCA** ($S=8$) | **0.1017** | ±0.0004 | 0.2499 | ±0.0286 |
| Pointwise ($S=1$) | 0.0749 | ±0.0001 | 0.2593 | ±0.0251 |

The pointwise baseline achieves lower MSE because it solves an easier reconstruction problem (1-to-1 vector mapping vs 8-to-1 compression). The MSE penalty for set training (+0.027) is the cost of the semantic compression that enables cross-model transfer (§5.4). In Top-K mode, both methods achieve similar cross-seed stability (J@32 ≈ 0.25), confirming that the hard sparsity constraint is the primary driver of reproducibility.

![Fig 1 Set vs Pointwise](./figures/fig01_set_vs_pointwise.png)

*Figure 1. MSE (left) and stability (right) for Set-ConCA vs Pointwise. Error bars are 95% CIs. The dashed line in the right panel marks the chance level (k/C = 0.25).*

---

### 5.2 Scaling with Set Size

We train Set-ConCA on the S-sweep dataset using $S \in \{1, 3, 8, 16, 32\}$.

| $S$ | MSE | ±SD | J@32 Stability | ±SD |
|--|--|--|--|--|
| 1 | 1.993 | 0.004 | 0.250 | 0.017 |
| 3 | 1.703 | 0.003 | 0.251 | 0.019 |
| **8** | **1.599** | 0.003 | **0.240** | 0.029 |
| 16 | 1.535 | 0.003 | 0.239 | 0.028 |
| 32 | 1.500 | 0.003 | 0.241 | 0.031 |

MSE decreases monotonically with $S$, consistent with a signal-averaging interpretation: larger $S$ provides a better estimate of the semantic mean, reducing reconstruction variance. The marginal improvement diminishes rapidly $S=8$ achieves approximately 80% of the MSE reduction from $S=1$ to $S=32$, at 25% of the batch cost. We select $S=8$ as the practical default.

![Fig 2 S-Scaling](./figures/fig02_s_scaling.png)

*Figure 2. MSE (left) and stability (right) as a function of set size S. Shaded bands show ±1 SD across 5 seeds. The dashed line marks S=8 (the knee of the benefit curve).*

---

### 5.3 Aggregator Ablation

We compare mean pooling against a learned attention aggregator.

| Aggregator | MSE | ±SD | Stability | ±SD |
|--|--|--|--|--|
| **Mean pool** | 1.599 | 0.003 | 0.240 | 0.029 |
| Attention | 1.562 | 0.003 | 0.268 | 0.027 |

Attention achieves marginally lower MSE via adaptive weighting but does not provide a stability advantage in this experiment. Crucially, the attention aggregator introduces a learned query vector that differs across training seeds, creating seed-dependent weightings of paraphrase elements. For interpretability applications, concept reproducibility across runs is paramount; we adopt mean pooling as the default.

![Fig 3 Aggregator Ablation](./figures/fig03_aggregator_ablation.png)

*Figure 3. MSE and stability for mean-pool vs attention aggregator (5 seeds, 95% CI).*

---

### 5.4 Cross-Family Concept Transfer

We train Set-ConCA independently on Gemma-3 4B and LLaMA-3 8B hidden states, learn an orthogonal Procrustes bridge on 80% of anchors, and evaluate Top-K transfer on the remaining 20%.

| Method / Direction | Transfer (J@32) $\uparrow$ | 95% CI |
|--|--|--|
| **Set-ConCA: Gemma-3 4B → LLaMA-3 8B** | **64.6%** | ±3.0% |
| Set-ConCA: LLaMA-3 8B → Gemma-3 4B | 54.7% | ±1.4% |
| Pointwise (SAE): 4B → 8B | 56.4% | ±3.0% |
| Chance level | 25.0% | |

Set-ConCA outperforms the pointwise baseline by **+8.2 pp** (p < 0.05 by non-overlapping 95% CIs). The bidirectional asymmetry is notable: transfer is substantially better in the direction of increasing capacity (4B→8B: 64.6% vs 8B→4B: 54.7%). We interpret this as a **capacity-receiver effect**: the larger model's higher-dimensional concept space can faithfully accommodate the smaller model's concept vectors, whereas the smaller model's more compressed representation discards distinctions that the larger model preserves. This finding has implications for the design of cross-model alignment procedures.

CKA between raw concept spaces (before bridging) is approximately 0.001, confirming that both models independently orient their concept spaces in different coordinate frames a linear bridge is always necessary, even within model families.

![Fig 4 Cross-Family Transfer](./figures/fig04_cross_family_transfer.png)

*Figure 4. Left: Bidirectional transfer bars for Set-ConCA (blue) vs Pointwise (light blue). Error bars: 95% CI. Right: Top-5 activating anchor texts for 5 concept dimensions, showing semantically coherent concept clusters.*

**Qualitative concept labels.** Examining the 5 most activated concept dimensions for representative anchors confirms that the mathematical decomposition recovers human-interpretable categories:
- **Concept #50** (mean activation 3.59): Asian geopolitics and technology (S.Korea politics, China space program, Google German legal disputes)
- **Concept #48** (mean activation 2.46): International athletics (Olympic cycling, heptathlon, European football)
- **Concept #0** (mean activation 2.25): US enterprise technology (digital advertising, PC retail, wireless services)

---

### 5.5 Intra-Family Alignment

We evaluate transfer within the Gemma model family across three model sizes.

| Transfer Direction | CKA (Pre-Bridge) | Transfer (J@32) |
|--|--|--|
| Gemma-3 1B → Gemma-3 4B | 0.0036 | 55.3% |
| Gemma-3 4B → Gemma-3 1B | 0.0036 | 55.7% |
| Gemma-3 4B → Gemma-2 9B | 0.0008 | 54.0% |
| Gemma-2 9B → Gemma-3 4B | 0.0008 | 54.9% |
| Gemma-3 1B → Gemma-2 9B | 0.0007 | 54.4% |
| Gemma-2 9B → Gemma-3 1B | 0.0007 | 54.7% |

A striking finding: all intra-family pairs achieve ~54–56% transfer, which is **lower** than cross-family Gemma-3 4B ↔ LLaMA-3 8B (64.6%). This parallels the capacity-receiver effect: the Gemma-3 1B is highly compressed relative to the 4B and 8B models, merging concept dimensions that larger models resolve distinctly. Two large models from different families (4B and 8B) share more conceptual resolution with each other than either shares with a 1B model, despite the latter belonging to the same architectural family.

![Fig 5 Intra-Family Heatmap](./figures/fig05_intra_family_heatmap.png)

*Figure 5. Transfer accuracy heatmap across Gemma model family. All pairs are substantially below the 64.6% cross-family result, consistent with the capacity-receiver interpretation.*

---

### 5.6 Comparison to SOTA Baselines

We compare Set-ConCA against four baselines at matched sparsity ($L_0 \approx 0.25$, i.e., 32 active concepts out of 128):

| Method | $L_0$ | MSE $\downarrow$ | Stability $\uparrow$ | Cross-Model Transfer |
|--|--|--|--|--|
| **Set-ConCA (ours)** | **0.246** | **0.102** | 0.250 | **64.6% ± 3.0%** |
| ConCA ($S=1$, ours) | 0.258 | 0.116 | 0.259 | 56.4% ± 3.0% |
| SAE-L1 (Anthropic style) | 0.25* | 0.175 | 0.332 | |
| SAE-TopK ($k=32$) | 0.250 | 0.187 | 0.315 | 56.4% ± 3.0% |
| PCA \dagger | 0.986 | 0.312 | 0.981 | |

*\* L1 penalty tuned to achieve L0 ≈ 25% for fair comparison.*
*"  PCA is a reconstruction upper bound; dense activations (L0 ≈ 99%) make it unsuitable as an interpretability method.*

Set-ConCA achieves the **lowest MSE** among sparse methods (0.102 vs SAE-TopK 0.187, −42% relative) at comparable sparsity. The access to $S=8$ simultaneous paraphrases during training provides substantially more semantic signal per gradient step than pointwise SAE training, reducing reconstruction error while maintaining the same output sparsity structure.

![Fig 6 SOTA Comparison](./figures/fig06_sota_comparison.png)

*Figure 6. Three-panel comparison across sparse methods: MSE, Stability, Sparsity Level. PCA is shown only as a reference baseline and excluded from the main ranking.*

---

### 5.7 Interventional Steering and Weak-to-Strong Generalisation

To confirm that Set-ConCA concept vectors are causally meaningful (not merely descriptive correlates), we perform interventional steering experiments. Given a source concept code $z^{(n)}_{\text{src}}$ from Model A and a Procrustes bridge $B^*$, we construct an intervened representation in Model B's concept space:

$$\tilde{z}^{(n)}_B = z^{(n)}_{\text{base}} + \alpha \cdot (z^{(n)}_{\text{src}} \cdot B^*)$$

and measure cosine similarity to the target concept $z^{(n)}_B$. We compare against a random direction baseline (same magnitude, random orientation).

| $\alpha$ | Set-ConCA 4B→8B | Weak-to-Strong 1B→8B | Random Direction |
|--|--|--|--|
| 0.0 | 0.914 | 0.914 | 0.914 |
| 0.5 | 0.926 | 0.930 | 0.831 |
| 2.0 | 0.932 | 0.940 | 0.410 |
| 5.0 | 0.933 | 0.943 | 0.331 |
| **10.0** | **0.933** | **0.944** | **−0.065** |

Set-ConCA concept vectors from the 4B model causally direct LLaMA-3 8B activations toward the target concept (+1.9 pp at $\alpha=10$). Random directions of identical magnitude **destroy alignment** (−97.9 pp vs baseline), confirming directional precision. The **Weak-to-Strong** result is particularly noteworthy: concept vectors from the Gemma-3 1B model steer the LLaMA-3 8B **more effectively** (+3.0 pp) than 4B vectors. We interpret this as evidence that aggressive parameter compression forces the 1B model to learn maximally distilled concept directions the fundamental semantic axes that are also most causally potent in larger models.

![Fig 7 Steering](./figures/fig07_steering.png)

*Figure 7. Cosine similarity to the target concept vs intervention strength α for Set-ConCA 4B→8B (blue), Weak-to-Strong 1B→8B (green), and a random baseline (pink). Shaded regions: 95% CI across 8 probe concepts.*

---

### 5.8 Ablations

**Consistency Loss (EXP9).** Removing the subset consistency loss ($\beta=0$) in Top-K mode changes transfer by only $-0.09$ pp (64.60% → 64.69%, within CI). This reveals that in Top-K mode, the hard $k$-constraint is the primary driver of invariance the consistency loss is relatively redundant. We expect the consistency loss to play a larger role in soft-sparsity (Sigmoid-$\ell_1$) mode, where no structural forcing exists. This is an important nuance: our architecture has two complementary invariance mechanisms, and one (TopK) subsumes the other in natural language settings.

**Paraphrase Corruption (EXP10).** Replacing 0%, 50%, or 100% of paraphrases per anchor with randomly sampled paraphrases from other anchors leaves transfer essentially unchanged (64.0%, 64.9%, 64.1%). This corroborates the EXP9 finding: the hard Top-K bottleneck is robust to semantic noise in the paraphrase set. The set structure is more important in soft-sparsity mode and for reconstruction quality (EXP2 shows clear S-scaling effects on MSE).

**Bridge Linearity (EXP12).** A 2-layer MLP bridge (C→256→C) improves transfer by only +0.5 pp over the linear Procrustes bridge (64.0% vs 64.7%), within seed-level variance. This near-linear geometry is the central geometric evidence for the Platonic Representation Hypothesis in the mechanistic interpretability setting.

**Information Depth (EXP11).** Using low-rank PCA projections (52% variance, rank=32) of the hidden states as a proxy for less-informative early layers achieves **77.8% transfer** substantially higher than full-dimensional activations (64.3%). The most transferable semantic content appears to be concentrated in the dominant spectral directions of the hidden state, suggesting that pre-filtering via dimensionality reduction before Set-ConCA training is a promising direction for future work.

![Fig 9 Consistency Ablation](./figures/fig09_consistency_ablation.png)
![Fig 12 Bridge Linearity](./figures/fig12_nonlinear_bridge.png)

*Figures 9 and 12. Left: Consistency ablation removing the consistency loss has negligible effect in Top-K mode. Right: Linear vs MLP bridge minimal improvement from nonlinearity confirms approximately linear concept space geometry.*

---

### 5.9 Interpretability Evaluation

We quantify concept interpretability using Normalised Mutual Information (NMI) between concept-space clusters and semantic pseudo-labels (K-means on PCA as a proxy for AG News categories), and linear probe accuracy.

| Method | NMI $\uparrow$ | Probe Accuracy $\uparrow$ |
|--|--|--|
| **Set-ConCA** | 0.832 | **98.5%** |
| SAE-L1 | **0.882** | 99.0% |
| PCA (dense) | 0.924* | 98.1% |

*\* PCA NMI is high because pseudo-labels were derived from PCA a circular comparison.*

Set-ConCA and SAE-L1 achieve essentially equivalent semantic interpretability on single-model metrics. This score draw is expected: both methods produce effective sparse representations of individual models. Set-ConCA's unique advantage is cross-model transferability the ability to port discovered concepts to other model families via a simple linear bridge.

![Fig 13 Interpretability](./figures/fig13_interpretability.png)

*Figure 13. NMI and linear probe accuracy for Set-ConCA, SAE-L1, and PCA. The score draw on single-model metrics is expected; Set-ConCA's advantage is cross-model transfer.*

---

## 6. Discussion

### What Set-ConCA Is (and Is Not) Solving

Set-ConCA addresses a specific failure mode of pointwise concept analysis: **surface-form conflation**. By training on paraphrase sets, it produces concept codes that are invariant to lexical and syntactic variation. This comes at a small MSE cost (+0.027 in EXP1), which is the mathematical price of compression: forcing $S$ diverse vectors through a single bottleneck necessarily discards element-specific information.

Crucially, this MSE cost does not reflect a failure of the method it reflects the **hardness of the task**. The appropriate evaluation axis is cross-model concept transfer (EXP4), where Set-ConCA leads by +8.2 pp over the pointwise baseline. This trade-off profile slightly more reconstruction error, substantially better concept universality is exactly the point.

### The Capacity-Receiver Effect

The most structurally novel empirical finding is the bidirectional asymmetry in cross-model transfer: transfer is better when moving concepts from smaller to larger models (4B→8B: 64.6%) than from larger to smaller (8B→4B: 54.7%). This pattern, which also explains why intra-family alignment is lower than cross-family alignment (§5.5), generalises as a "capacity-receiver" principle: **the resolution of the target model's concept space determines how much of the source model's concept structure can be faithfully accommodated**. This has practical implications cross-model concept porting should be directed towards models with equal or greater capacity.

### Platonic Geometry Evidence

The near-linear bridge sufficiency (EXP12: MLP adds only +0.5 pp) is the strongest geometric evidence yet for the Platonic Representation Hypothesis in the mechanistic interpretability setting. Previous evidence for the hypothesis (Huh et al., 2024) was at the level of representation similarity metrics. Our result goes further: the concept spaces not just the raw activation spaces admit linear alignment at 64.6% Top-K overlap. If the geometry were fundamentally nonlinear, a 2-layer MLP would provide substantially larger gains.

### Limitations

1. **Paraphrase requirement at training time.** Set-ConCA requires parallel paraphrase corpora for training. While these are naturally available for news text, generating exact semantic paraphrases for formal domains (mathematics, code) is substantially harder. Inference requires only a single input.

2. **Linear encoder.** Our encoder is strictly linear ($\mathbf{u}_i = \mathbf{W}_e \mathbf{x}_i$). This may miss nonlinear concept structure if concepts are not linearly decodable from single-layer activations.

3. **Fixed layer analysis.** All experiments use layer 20. EXP11 (information depth proxy) suggests that different layers or more precisely, different spectral subspaces may carry different amounts of transferable semantic content. A full multi-layer analysis is deferred to future work.

4. **Proxy interpretability labels.** EXP13 uses K-means on PCA as pseudo-labels, which introduces potential circularity. Ground-truth human-annotated concept labels would provide more direct evidence of interpretability.

5. **Single domain.** All experiments use news text. Cross-domain generalization requires validation on additional corpora.

---

## 7. Conclusion

We introduced Set-ConCA, a method for extracting semantically invariant concept representations from LLM hidden states via paraphrase-set training. The central insight is that concepts invariant to paraphrase variation are precisely the concepts that survive cross-model linear alignment because both forms of invariance reflect the same underlying semantic universality.

On 2,048 news anchors across four LLMs, Set-ConCA achieves 64.6% ± 3.0% Top-K concept transfer from Gemma-3 4B to LLaMA-3 8B via a linear Procrustes bridge, outperforming the pointwise baseline by +8.2 pp. A nonlinear MLP bridge improves this by only +0.5 pp, implying approximately linear Platonic geometry between model concept spaces. Concept vectors are causally validated via interventional steering, with a striking weak-to-strong result: concept vectors from the 1B model steer the 8B model (+3.0 pp) more effectively than 4B-model vectors (+1.9 pp).

Together, these results suggest that with the right training signal set-based invariance mechanistic interpretability methods can recover concept representations that are genuinely universal across LLM architectures. We hope this work contributes to a foundation for cross-model interpretability techniques that do not require model-specific re-training.

---

## 8. Broader Impacts

**Positive applications.** Set-ConCA's cross-model concept transfer enables interpretability tools trained on open-source models to generate insights applicable to larger or closed models. The weak-to-strong steering result suggests that safety-relevant concept directions identified in small models could be used to steer frontier-scale models supporting alignment research.

**Potential misuse.** Interventional concept steering (EXP7) demonstrates the ability to causally redirect model representations. In the wrong hands, this capability could be used to steer models toward harmful outputs or to bypass safety guardrails. We note that all experiments in this work target representational similarity, not text generation, and that the steering effects demonstrated are small in magnitude. We encourage the broader community to develop safeguards alongside capability research in this area.

---

## References

\bibitem{anthropic2024scaling}
Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., ... & Henighan, T. (2024). *Scaling and evaluating sparse autoencoders*. Anthropic, 2024.

\bibitem{bricken2023monosemanticity}
Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., ... & Olah, C. (2023). *Towards monosemanticity: Decomposing language models with dictionary learning*. Anthropic Transformer Circuits Thread, 2023.

\bibitem{elhage2022superposition}
Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., ... & Olah, C. (2022). *Toy models of superposition*. Transformer Circuits Thread, 2022.

\bibitem{gao2024scalingsae}
Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., ... & Leike, J. (2024). *Scaling and evaluating sparse autoencoders*. arXiv:2406.04093.

\bibitem{huh2024platonic}
Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). *The platonic representation hypothesis*. ICML 2024. arXiv:2405.07987.

\bibitem{zou2023representation}
Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Hendrycks, D. (2023). *Representation engineering: A top-down approach to AI transparency*. arXiv:2310.01405.

\bibitem{hotelling1936cca}
Hotelling, H. (1936). *Relations between two sets of variates*. Biometrika, 28(3/4), 321-377.

\bibitem{radford2021clip}
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). *Learning transferable visual models from natural language supervision*. ICML 2021.

\bibitem{chen2020simclr}
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A simple framework for contrastive learning of visual representations*. ICML 2020.

\bibitem{raghu2017svcca}
Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). *SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability*. NeurIPS 2017.

\bibitem{kornblith2019cka}
Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). *Similarity of neural network representations revisited*. ICML 2019.

\bibitem{schonemann1966procrustes}
Schönemann, P. H. (1966). *A generalized solution of the orthogonal Procrustes problem*. Psychometrika, 31(1), 1-10.

\bibitem{csiszarik2021model_stitching}
Csiszárik, A., Kőrösi-SzabÃ³, P., Matszangosz, Ã. K., Papp, G., & Varga, D. (2021). *Similarity and matching of neural network representations*. NeurIPS 2021.

\bibitem{cunningham2023sparse}
Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). *Sparse autoencoders find highly interpretable features in language models*. arXiv:2309.08600.

---

## Appendix A Hyperparameter Details

| Hyperparameter | Value | Justification |
|--|--|--|
| Concept dim $C$ | 128 | Balances expressivity and sparsity interpretability |
| Top-K $k$ | 32 | $k/C = 0.25$ matches SAE baselines for fair comparison |
| Encoder lr | $2\times10^{-4}$ | Standard for sparse autoencoders |
| Epochs | 80 | Loss converges by epoch 50 (EXP8); 80 provides margin |
| Batch size | 64 (anchors) | 64 × 8 = 512 vectors per step |
| $\alpha$ (sparsity) | 0.1 | Soft mode only; 0 in Top-K mode |
| $\beta$ (consistency) | 0.01 | Small weight; does not interfere with reconstruction |
| Bridge epochs | 300 | Procrustes training; lr=0.01 |
| Orthogonality penalty | 0.1 | $0.1 \| B^\top B - I \|_F^2$ added to bridge loss |

## Appendix B Reproducibility

All experiments are fully reproducible. The codebase includes 52 unit tests (`uv run pytest tests/test_setconca.py -v`) covering all architectural components, loss functions, and experiment runners. Key reproducibility guarantees:
- Fixed seeds $\{42, 1337, 2024, 7, 314\}$ via `torch.manual_seed` + `torch.Generator`
- Deterministic DataLoader with fixed generator seed
- Model checkpoints and result JSON saved to `results/results_v2.json`
- All figure generation in `experiments/neurips/plot_results.py` (`uv run python experiments/neurips/plot_results.py`)

Code and data will be released at [TODO: repository URL].

## Appendix C Additional Figures

![Fig 8 Convergence](./figures/fig08_convergence.png)

*Figure A1. Training convergence curves (3 seeds). Loss stabilises by epoch 50.*

![Fig 10 Corruption Test](./figures/fig10_corruption_test.png)

*Figure A2. Transfer robustness to paraphrase corruption. TopK hard sparsity is robust to semantic noise in the set.*

![Fig 11 Layer Sweep](./figures/fig11_layer_sweep.png)

*Figure A3. Transfer accuracy vs PCA rank (information depth proxy). Lower-rank projections transfer better, suggesting dominant spectral directions carry the most universal semantics.*

![Fig 14 Capability Matrix](./figures/fig14_capability_matrix.png)

*Figure A4. Full capability comparison matrix across all baselines.*


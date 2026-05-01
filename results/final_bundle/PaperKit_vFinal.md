# PaperKit_vFinal

## Abstract skeleton
We present Set-ConCA, a set-based extension of ConCA for concept extraction from representation sets. We validate strong cross-family transfer and causal steering while documenting failure modes and claim boundaries through an explicit evidence ledger.

## Intro framing
- Problem: mechanistic concept extraction lacks a unified theoretical-to-empirical bridge.
- Thesis: ConCA provides the theoretical basis; Set-ConCA extends it to local representation distributions.
- Contribution style: strongest on transfer+steering evidence, conservative on raw-overlap dominance.

## Abstract draft (expanded)
Concept-level interpretability in large language models often relies on sparse decomposition methods that provide useful empirical signals but limited theoretical grounding. We build on Concept Component Analysis (ConCA), which frames representation decomposition through a latent-variable, log-posterior perspective, and extend it to representation sets via Set-ConCA. Set-ConCA replaces pointwise inputs with local sets, combines permutation-invariant aggregation with sparse concept estimation, and includes consistency-oriented regularization for set-level robustness. Using a reproducible evaluation pipeline, we report strong cross-family transfer and causal steering signals while explicitly preserving mixed and negative findings. In particular, we show competitive but non-dominant behavior on raw overlap versus strong pointwise TopK baselines, and we provide a claim-evidence ledger that constrains all report conclusions to verifiable artifacts. This produces a publication-ready, evidence-disciplined framing: Set-ConCA is a credible set-based sparse concept method with clear strengths, explicit limitations, and reproducible evaluation boundaries.

## Introduction draft blocks
### Problem setup
Mechanistic interpretability requires extracting latent features that are both behaviorally useful and scientifically interpretable. Existing sparse decomposition approaches often optimize useful objectives but leave ambiguity around the semantic meaning of latent coordinates and the scope of valid claims.

### ConCA motivation
ConCA addresses this gap by introducing a principled latent-variable framing where model representations are interpreted through approximate concept-conditioned log-posterior structure. This provides a clearer theoretical basis for sparse concept recovery than purely heuristic sparse reconstruction.

### Set-ConCA motivation
Pointwise decomposition can be brittle under local context variation. Many semantic factors in language emerge over local distributions of paraphrases or related states, motivating extension from single points to representation sets.

### Contributions framing
1. Reframe concept extraction from pointwise to set-conditioned estimation.
2. Provide reproducible empirical evidence for transfer and steering behavior.
3. Introduce explicit claim governance through machine-readable evidence mapping.
4. Preserve negative/mixed results as first-class scientific outputs.

## Methods writing blocks
- ConCA latent-variable and log-posterior perspective.
- Set aggregation and subset consistency mechanics.
- Implementation deltas and practical training recipe.

## Methods draft blocks (expanded)
### Core formulation
- ConCA-style objective provides sparse concept coordinates under posterior-motivated interpretation.
- Set-ConCA extends this by encoding set elements, aggregating via permutation-invariant pooling, and decoding with shared/residual structure.

### Practical training setup
- Concept dimension: 128
- TopK sparsity: 32
- Epoch budget: 80
- Seeds: 5
- Primary transfer bridge comparisons include linear and nonlinear variants.

### Evaluation architecture
- Core suite: EXP1–EXP16
- Extended diagnostics: additional alignment baselines and layer-proxy analyses
- Multilingual matrix benchmarking: WMT14 FR-EN and OPUS100 multi-EN

### Claim governance layer
All manuscript claims must map to ledger-backed artifact keys before inclusion in abstract, highlights, or conclusion sections.

## Results writing blocks
- Primary positive findings: transfer and steering.
- Mixed findings: consistency effects and single-model interpretability proxies.
- Negative findings: corruption-collapse non-support and pointwise TopK superiority on raw overlap.

## Results draft blocks (expanded)
### Primary outcomes
- Cross-family transfer remains substantially above chance.
- Causal steering gains are reproducible and materially separated from random-control behavior.
- Linear bridge remains strong and competitive relative to nonlinear alternatives in current reruns.

### Mixed outcomes
- Consistency objective shows limited effect in current TopK regime.
- Some proxy interpretability metrics are competitive rather than dominant.

### Negative outcomes
- Pointwise TopK surpasses Set-ConCA on raw overlap in the current setting.
- Corruption test does not support collapse-to-chance semantics under the tested protocol.

### Multilingual outcomes
- Benchmark path is operational across both target datasets.
- Set-ConCA is competitive in matrix summaries but does not support dominance framing.

## Limitations block
- Pseudo-layer diagnostics.
- Metric non-equivalence across baseline families.
- Compute-limited full SOTA parity in some settings.

## Discussion/conclusion draft blocks
### What is now defensible
Set-ConCA can be defended as a credible set-based sparse concept method with robust transfer and steering evidence, plus operational multilingual evaluation infrastructure.

### What remains open
- True per-layer extraction across heterogeneous architectures.
- Stronger corruption protocols that isolate semantic disruption mechanisms.
- Tighter apples-to-apples parity for heterogeneous baseline families.

### Final conclusion language
The central contribution is not a universal leaderboard claim; it is a disciplined, theory-linked, empirically transparent concept-method framing that preserves both strengths and failure modes.

## Writing guardrails for final paper
- Do not claim universal baseline superiority.
- Explicitly separate `core_result` and `supporting/exploratory` findings.
- Report mixed/null outcomes in main text, not only appendix.
- Ensure each headline number cites artifact-backed evidence paths.

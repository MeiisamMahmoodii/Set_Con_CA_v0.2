# MasterReport_vFinal

## Executive position
Set-ConCA is supported as a credible set-based sparse concept method with strong cross-family transfer and steering evidence, while several earlier broad claims are now downgraded to mixed or negative findings.

## Scope and evidence policy
This master report is publication-oriented and evidence-constrained. It is intended as the canonical long-form source for paper writing, presentation preparation, and technical audit.

All primary claims in this document are restricted to local, reproducible artifacts:
- `results/results_v2.json`
- `results/extended_alignment_results.json`
- `results/benchmark_matrix_wmt14_fr_en.json`
- `results/benchmark_matrix_opus100_multi_en.json`
- `results/run_manifest_v2.json`

Theoretical framing is anchored to:
- `c:\Users\MPC\Documents\papers\ConCA.pdf`
- `c:\Users\MPC\Documents\papers\Set_ConCA.pdf`

## ConCA -> Set-ConCA -> current repo story
1. ConCA introduced a probabilistic concept-extraction frame via latent log-posterior unmixing.
2. Set-ConCA generalized this to representation sets to stabilize concept estimation under local variation.
3. Current repo operationalized this with set aggregation, sparse training, bridge transfer tests, extended alignment diagnostics, and multilingual benchmark matrices.

## Scientific narrative reconstruction
### Why ConCA matters
ConCA addresses a core ambiguity in sparse interpretability methods: what sparse dimensions represent. Instead of treating sparse coding as only a reconstruction heuristic, ConCA motivates a latent-variable view where recovered coordinates approximate concept-conditioned posterior structure.

### Why Set-ConCA is a principled extension
Pointwise decomposition is sensitive to local noise and context idiosyncrasy. Set-ConCA extends the objective to local representation sets, allowing concept estimation from shared evidence across paraphrase neighborhoods rather than single points.

### What the current repo contributes
The current implementation contributes an evidence-producing workflow:
- EXP1–EXP16 structured experiment suite,
- cross-model transfer and steering tests,
- expanded baseline matrix with multiple method families,
- and claim-verification tooling to prevent narrative drift.

## Methods and protocol transparency
### Canonical run configuration
- Device: CUDA
- Anchors: 2048
- Epochs: 80
- Concept dimension: 128
- TopK sparsity: 32
- Seeds: 5 (`42, 1337, 2024, 7, 314`)

### Pipeline structure
- Core experiments: `evaluation/run_evaluation_v2.py`
- Extended diagnostics: `evaluation/run_extended_alignment.py`
- Multilingual benchmark build: `evaluation/build_multilingual_benchmarks.py`
- Multilingual matrix runs: `evaluation/run_benchmark_matrix.py`
- Report verification: `scripts/verify_reports_against_artifacts.py`

## Core validated findings (from claim ledger)
- Cross-family transfer remains strong.
- Steering gains are reproducible and materially above random control.
- Linear bridge remains competitive/strong versus nonlinear bridge.
- Set-ConCA is competitive but not dominant on raw overlap against strong pointwise TopK.
- Consistency and corruption narratives must be conservative in current TopK regime.

## Quantitative claim table (publication-safe)
| Claim ID | Statement | Value | Source |
|---|---|---:|---|
| C001 | Cross-family transfer (Gemma->LLaMA) is strong | 69.5% | `results_v2.exp4_cross_family` |
| C002 | Steering gain at alpha=10 is material | +9.8pp | `results_v2.exp7_steering` |
| C003 | Linear bridge outperforms nonlinear bridge in rerun | 69.3% vs 64.2% | `results_v2.exp12_nonlinear_bridge` |
| C004 | Pointwise TopK exceeds Set-ConCA on raw overlap | 78.4% vs 69.5% | `results_v2.exp16_topk_pointwise_vs_set` |
| C005 | Consistency effect in TopK mode is marginal | +0.1pp | `results_v2.exp9_consistency_ablation` |
| C006 | Corruption does not collapse transfer | 69.3% -> 69.2% | `results_v2.exp10_corruption_test` |
| C007 | Multilingual path operational (Set-ConCA means) | 0.3802 / 0.3688 | WMT14 / OPUS100 matrices |
| C008 | Extended controls include strong non-Set baselines | NMF 0.8348 | `extended_alignment_results.sota_extensions` |

## Failure and limitation accounting
- Pointwise TopK exceeds Set-ConCA on raw overlap in current setup.
- Corruption test does not support semantic collapse framing.
- Pseudo-layer diagnostics are exploratory until true per-layer extraction is added.
- SOTA comparisons require parity checks before headline interpretation.

## Mixed and negative findings (must be retained)
### Consistency objective
In current TopK settings, removing consistency changes transfer only marginally. This is a meaningful boundary condition and should be reported as a mixed/negative finding, not hidden.

### Corruption sensitivity
The tested corruption mechanism does not induce collapse-to-chance behavior. This should be framed as robustness under the tested protocol, plus motivation for stronger semantic-disruption designs.

### Raw-overlap competitiveness
Set-ConCA cannot claim raw-overlap superiority in this run. Its strongest defensible framing remains transfer/steering credibility with conservative comparison language.

## Cross-language alignment interpretation
- WMT14 and OPUS100 matrices are fully operational in final artifacts.
- Cross-language comparisons should be reported as competitive/complementary, not absolute dominance.
- Directional asymmetry and model-family effects should be discussed explicitly in the paper.

## Expanded multilingual interpretation
### Operational claim
The multilingual benchmark system is now operational and reproducible across both datasets in scope.

### Reporting guidance
- Use matrix means and pairwise structure as comparative signals.
- Avoid over-attributing causes from matrix-level summaries alone.
- Keep method-family assumptions explicit when comparing across dense/sparse/causal baselines.

## SOTA comparison policy
### Baseline diversity and risk
The benchmark includes sparse coding, factorization, transport, correlation, intervention, and alignment families. This breadth is valuable but creates fairness risk if assumptions are collapsed into one leaderboard narrative.

### Fair-comparison checklist
Before putting a method in headline claims, require:
- metric equivalence,
- dataset equivalence,
- supervision comparability,
- compute-budget disclosure/parity,
- and reproducibility trace.

### Writing rule
If parity is incomplete, use "contextual comparison" wording, not "direct superiority" wording.

## Theory-to-implementation traceability
Read this report with `ConCA_SetConCA_Math_Foundation.md`, which maps paper-level assumptions/equations to implementation status (`implemented`, `partially_implemented`, `not_yet_implemented`). This protects against theoretical over-claiming.

## Reproducibility and quality gates
This bundle is considered report-ready when:
- `scripts/verify_reports_against_artifacts.py` passes,
- `pytest` passes (currently 62),
- and claim ledger evidence keys remain valid.

Any metrics update requires bundle regeneration and verification reruns before publication edits.

## Publication-safe claim hierarchy
### Primary claims
- Cross-family transfer signal is reproducible and well above chance.
- Steering signal is causal and robust relative to random control.
- Set-ConCA is a credible, competitive set-based sparse concept method.

### Secondary claims
- Linear bridge sufficiency is supported in current rerun.
- Extended diagnostics add useful controls and stress tests.

### Disallowed claims
- Universal baseline dominance.
- Necessity of consistency in current TopK mode.
- Corruption-collapse semantic claim.
- Broad layer-mechanism claims from proxy-layer diagnostics.

## Recommended manuscript mapping
Use this report to draft:
- abstract/introduction around primary claims only,
- methods with clear paper-to-code deltas,
- results including success + mixed + negative findings,
- discussion with explicit limitations and residual risks,
- related work with strict comparability notes.

## Deliverable links
- `ClaimLedger_vFinal.md`
- `ConCA_SetConCA_Math_Foundation.md`
- `Data_Provenance_and_Quality_vFinal.md`
- `SOTAReproductionAppendix_vFinal.md`
- `PaperKit_vFinal.md`
- `PresentationKit_vFinal.md`
- `CheatSheet_vFinal.md`

## Final bottom line
Set-ConCA is paper-viable under an honest, evidence-disciplined framing: strong on cross-family transfer and steering, operational on multilingual benchmarking, competitive but not dominant on raw overlap, and transparent about mixed/negative findings.

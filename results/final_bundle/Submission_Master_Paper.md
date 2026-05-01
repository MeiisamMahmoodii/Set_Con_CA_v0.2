# Set-ConCA Submission Master Paper (Markdown Primary)

## Abstract
We present Set-ConCA, a set-based extension of concept component analysis for sparse concept extraction from representation sets rather than isolated points. In the current verified rerun, Set-ConCA demonstrates strong cross-family transfer and causal steering evidence, while several previously broad claims are narrowed by explicit negative/mixed findings. Specifically, cross-family transfer from Gemma-3 4B to LLaMA-3 8B reaches 69.5% +/- 0.6pp (chance 25%), steering gain at alpha=10 is +9.8pp (weak-to-strong +10.7pp), and linear bridge mapping outperforms nonlinear MLP (69.3% vs 64.2%). At the same time, pointwise SAE-TopK exceeds Set-ConCA on raw overlap (78.4% vs 69.5%), consistency contribution is small in current TopK mode (+0.1pp), and corruption does not produce collapse-to-chance behavior. The multilingual benchmark pipeline is operational across WMT14 FR-EN and OPUS100 multi-EN, where Set-ConCA is competitive rather than dominant. We release a claim-evidence ledger to enforce report-level traceability and claim discipline.

## Introduction
Mechanistic interpretability methods must be judged not only by isolated benchmark wins but by whether their claims remain aligned with reproducible artifacts. Set-ConCA is motivated by the observation that many semantic factors appear across local representation distributions (paraphrase/neighbor sets), not single activations. Extending ConCA from pointwise to setwise extraction introduces permutation-invariant aggregation and subset-consistency regularization, enabling concept estimation under local variation.

## Method overview
- ConCA foundation: sparse concept extraction under latent/posterior-motivated framing.
- Set extension: set encoding + aggregation + shared/residual decode + subset consistency.
- Bridge transfer: source-to-target concept mapping evaluated by overlap on held-out anchors.
- Steering: intervention along mapped concept directions with random controls.

## Experimental setup
- Anchors: 2048
- Epochs: 80
- Concept dimension: 128
- TopK: 32
- Seeds: 5
- Canonical artifacts: `results/results_v2.json`, `results/extended_alignment_results.json`, multilingual matrix JSON files.

## Results summary
### Strong evidence
- EXP4 cross-family transfer: 69.5% +/- 0.6pp
- EXP7 steering gain: +9.8pp
- EXP12 linear vs MLP bridge: 69.3% vs 64.2%

### Mixed/negative evidence
- EXP16 raw overlap: pointwise SAE-TopK > Set-ConCA (78.4% vs 69.5%)
- EXP9 consistency effect: +0.1pp
- EXP10 corruption: no collapse-to-chance

### Multilingual
- Pipeline operational for WMT14 and OPUS100.
- Set-ConCA means: 0.3802 / 0.3688.
- Framing: competitive, not dominant.

## Limitations
- True per-layer heterogeneous extraction is pending (proxy-layer diagnostics are exploratory).
- Baseline comparability is heterogeneous across method families.
- Stronger semantic corruption protocols are needed for broader robustness claims.

## Claim safety
Safe: transfer/steering credibility, operational multilingual pipeline, linear bridge sufficiency in current rerun.  
Unsafe: universal baseline dominance, strict consistency necessity, corruption-collapse claim.

## Pointers
- Full technical report: `results/REPORT.md`
- Full deep-dive and Q&A: `results/final_bundle/Supervisor_Meeting_Paper.md`
- LaTeX submission primary: `docs/paper/setconca_neurips.tex`

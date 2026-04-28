# Figure and Table Specification Map

This document binds each result subsection to required outputs, metrics, significance procedures, and artifact filenames.

## Section 8.1 Main Results

**Table 3: Main Results**
- Rows: SAE best baseline, ConCA baseline, Set-ConCA variants (mean/attention, sigmoid/top-k).
- Columns:
  - Reconstruction MSE
  - Stability metric (consistency loss + cosine stability)
  - Cross-model overlap metric
  - Faithfulness metric
  - Seeds (`n`)
- Statistics:
  - mean ± std or 95% bootstrap CI
  - fixed seed list explicitly shown in caption or footnote
- Artifact inputs:
  - `docs/paper/artifacts/main_results.csv`
  - script: `scripts/generate_main_results.py`

## Section 8.2 Set-Size Scaling

**Figure 3: S vs Stability Curve**
- X-axis: `S={1,3,4,8,12,16,24,32}`
- Y-axis: consistency loss and/or cosine stability.
- Visual:
  - mean curve
  - CI band across seeds
- Artifact:
  - `docs/paper/artifacts/s_sweep.csv`
  - `docs/paper/artifacts/s_curve.png`
  - script: `scripts/run_s_sweep.py`

**Table 4: Key S Values**
- Main paper: `S=3,8,16`
- Appendix: full S sweep table.
- Columns:
  - MSE
  - consistency
  - faithfulness
  - overlap
  - CI

## Section 8.3 Top-K Ablation

**Table 5: Top-K Ablation**
- Grid:
  - `k={16,32,50,64}`
  - sparsity mode: top-k vs sigmoid/L1 path
  - aggregator: mean vs attention
- Columns:
  - reconstruction
  - stability
  - overlap
  - faithfulness
  - runtime/memory (optional)
- Artifact:
  - `docs/paper/artifacts/topk_ablation.csv`
  - script: `scripts/run_topk_ablation.py`

**Figure 4: Ablation Heatmap**
- Axes: set size `S` by `k` (split by aggregator mode).
- Color: selected metric (default: transfer overlap or composite score).
- Artifact:
  - `docs/paper/artifacts/topk_heatmap.png`

## Section 8.4 Cross-Model Transplant Significance

**Table 6: Bridge Significance**
- Directions:
  - Gemma -> Llama
  - Llama -> Gemma (if feasible)
- Metrics:
  - top-k overlap
  - Jaccard
  - retrieval metric (MRR or top-1/top-5)
- Null baselines:
  - shuffled pairings
  - random projection bridge
  - pointwise baseline bridge
- Statistics:
  - permutation p-value
  - bootstrap CI
  - effect size
- Artifact:
  - `docs/paper/artifacts/bridge_significance.csv`
  - script: `scripts/run_bridge_significance.py`

**Figure 5: Overlap vs Baseline**
- Bars with error bars for each baseline.
- Artifact:
  - `docs/paper/artifacts/bridge_baseline_bars.png`

## Section 8.5 Causal Faithfulness

**Table 7: Layer-Selected Faithfulness**
- Main paper layers: selected representative layers (e.g., low/mid/high).
- Columns:
  - KL divergence
  - logit agreement
  - final-token agreement
  - normalized faithfulness score
  - with/without norm alignment
- Artifact:
  - `docs/paper/artifacts/faithfulness_layers_main.csv`
  - script: `scripts/run_faithfulness_eval.py`

**Figure 6: Faithfulness vs Layer**
- Curves:
  - aligned vs non-aligned
  - optionally baseline methods
- Artifact:
  - `docs/paper/artifacts/faithfulness_vs_layer.png`

## Section 8.6 Steering Transfer

**Table 8: Steering Transfer Effects**
- Conditions:
  - transplanted direction
  - random direction control
  - unbridged source direction control
- Columns:
  - target behavior shift metric
  - side-effect/regression metric
  - CI and p-value
- Artifact:
  - `docs/paper/artifacts/steering_transfer.csv`
  - script: `scripts/run_steering_transfer.py`

## Mandatory Statistical Footnote Template (for every main table)

- number of seeds
- CI method and confidence level
- hypothesis test type
- multiple-comparison correction (if used)
- exact artifact path and script command

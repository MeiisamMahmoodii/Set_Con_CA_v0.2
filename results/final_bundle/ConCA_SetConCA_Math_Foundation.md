# ConCA_SetConCA_Math_Foundation

## Purpose and scope
This document is the math-heavy third-pass foundation for ConCA -> Set-ConCA -> current SetConCA implementation. It is designed to support paper drafting and reviewer-facing methodological clarity while preserving strict claim discipline.

Theoretical anchors:
- `c:\Users\MPC\Documents\papers\ConCA.pdf`
- `c:\Users\MPC\Documents\papers\Set_ConCA.pdf`

Empirical anchors:
- `results/results_v2.json`
- `results/extended_alignment_results.json`
- multilingual matrix artifacts

## 1) Notation
| Symbol | Meaning |
|---|---|
| `x` | input text context (pointwise sample) |
| `X = {x1,...,xm}` | local representation set (paraphrase/neighbor set) |
| `f(x)` | model hidden representation for point `x` |
| `f(xi)` | hidden representation for set element `xi` |
| `z` | latent concept coordinate vector (pointwise) |
| `zX` | set-level latent concept coordinate vector |
| `zi` | i-th latent concept variable |
| `We, be` | encoder parameters |
| `Wd, bd` | pointwise decoder parameters |
| `Wds, Wdr` | shared and residual set decoder parameters |
| `R(.)` | regularization/normalization operator in latent path |
| `g(.)` | stable mapping from latent log-posterior-like domain to probability-like domain |
| `S(.)` | sparsity penalty operator |
| `alpha` | sparsity regularization weight |
| `beta` | subset consistency regularization weight |
| `Lcons` | subset consistency loss |
| `theta` | full trainable parameter set |

## 2) ConCA mathematical foundation
### 2.1 Latent-variable perspective
ConCA adopts a latent-variable generative view where concepts are represented by latent variables `z = (z1,...,zl)`. Under this view, learned representations are modeled as approximately encoding a linear mixture of concept log-posteriors conditioned on input context.

### 2.2 Core representation relation (paper-level)
The key conceptual relation is:

`f(x) ~= A * phi(x) + b`,

where `phi(x)` stacks concept-wise log-posterior terms. ConCA seeks an unsupervised linear unmixing map that recovers a sparse, concept-meaningful coordinate system.

### 2.3 Pointwise sparse ConCA parameterization
Pointwise latent and reconstruction:

`z_hat = R(We f(x) + be)`

`f_hat(x) = Wd z_hat + bd`

Objective:

`min_theta E_x [ ||f_hat(x)-f(x)||_2^2 + alpha * S(g(z_hat)) ]`

Interpretation:
- reconstruction term enforces representational fidelity,
- sparsity term regularizes the underdetermined unmixing problem.

## 3) Set-ConCA mathematical extension
### 3.1 From points to sets
Set-ConCA replaces single input `x` with set `X = {x1,...,xm}` to estimate shared concept evidence across local variation.

### 3.2 Element encoding and permutation-invariant aggregation
Element-level pre-latents:

`ui = We f(xi) + be`

Set aggregation:

`u_bar_X = P({u1,...,um})`

Set latent:

`z_hat_X = R(u_bar_X)`

where `P` is permutation-invariant (mean pooling in the base paper version; attention as an implementation variant).

### 3.3 Shared + residual decoder
Set-ConCA decomposes reconstruction into shared concept and element-specific residual:

`f_hat(xi) = Wds z_hat_X + Wdr ui + bd`

This prevents trivial set collapse while preserving a common concept factor.

### 3.4 Full set objective
`min_theta E_X [ (1/m) * sum_i ||f_hat(xi)-f(xi)||_2^2 + alpha * S(g(z_hat_X)) + beta * Lcons ]`

with subset consistency:

`Lcons = || z_hat_Xa - z_hat_Xb ||_2^2`

for random subsets `Xa, Xb subseteq X`.

## 4) Proposition-style statements and proof sketches
### Proposition 1 (Permutation invariance of set latent)
If `P` is permutation-invariant, then `z_hat_X` is invariant to ordering of elements in `X`.

Proof sketch:
- `u_bar_X = P({u1,...,um})` is unchanged under any permutation by definition of `P`.
- `z_hat_X = R(u_bar_X)` depends only on `u_bar_X`.
- therefore set latent is order-invariant.

### Proposition 2 (Set posterior accumulation intuition)
Under conditional-independence-style assumptions used in `Set_ConCA.pdf`, log-posterior evidence over a set approximately accumulates across elements; mean/sum-style pooling provides an empirical estimator of shared evidence.

Proof sketch:
- write `log p(zi|X)` as prior + sum of per-element evidence terms up to constants.
- each `ui` approximates pointwise concept evidence in latent space.
- pooled `u_bar_X` therefore approximates normalized accumulated evidence.

### Proposition 3 (Subset consistency as stability regularizer)
For fixed `beta > 0`, minimizing `Lcons` reduces variance of latent set codes under random subset sampling.

Proof sketch:
- `Lcons` penalizes pairwise subset-code deviation directly.
- expected penalty upper-bounds mean squared subset discrepancy.
- optimizer pressure reduces subset-induced latent fluctuation.

### Proposition 4 (No dominance guarantee from objective alone)
The Set-ConCA objective does not imply universal raw-overlap superiority over all baselines.

Proof sketch:
- objective optimizes reconstruction+sparsity+consistency terms, not any single benchmark overlap metric.
- different baseline objectives may align better with specific overlap metrics.
- empirical comparison is therefore required and can yield mixed outcomes.

## 5) Paper-to-code traceability map
| Paper component | Repo implementation | Status |
|---|---|---|
| ConCA-style sparse concept reconstruction | `evaluation/run_evaluation_v2.py` + `setconca/model/setconca.py` | implemented |
| Set aggregation (mean/attention) | `train_setconca(..., agg_mode)` | implemented |
| Shared/residual decoding behavior | `compute_loss(..., lambda_res=0.1)` path | partially_implemented |
| Subset consistency term | `compute_loss(..., beta=...)` and EXP9/EXP15 | implemented |
| True per-layer heterogeneous extraction | pseudo-layer diagnostics in extended alignment | not_yet_implemented |

## 6) Implementation deltas from paper idealization
### Delta A: aggregation variants
Paper base exposition emphasizes mean pooling; repo includes attention aggregation mode for empirical ablation.

### Delta B: bridge/evaluation layer
Cross-model transfer claims depend on learned bridge and overlap metric choices; this is an empirical layer above core objective.

### Delta C: layer analysis constraints
Current layerwise analysis includes proxy-layer diagnostics due to artifact constraints; true heterogeneous per-layer extraction remains open.

## 7) Mathematical claim boundary table
| Claim type | Allowed statement style |
|---|---|
| Theoretical | "Under stated assumptions, this objective/formulation implies ..." |
| Empirical core | "In current artifacts/reruns, we observe ..." |
| Empirical mixed | "Effect appears weak/uncertain in current setup ..." |
| Prohibited | "Universally superior", "strictly necessary in all regimes", "proven by single proxy test" |

## 8) Provable vs empirical
- Provable/theoretical (paper-level): latent-variable interpretation and log-posterior motivation, set invariance properties, objective-level regularization behavior.
- Empirical (repo-level): transfer, steering, robustness, SOTA comparison, multilingual matrices.
- Guardrail: empirical claims are only allowed when matched to `ClaimLedger_vFinal.json`.

## 9) Reviewer-facing concise statements
- ConCA contribution: principled concept-extraction framing via latent log-posterior interpretation.
- Set-ConCA contribution: extension from pointwise to set-conditioned concept estimation with permutation-invariant aggregation and consistency stabilization.
- Current repo contribution: evidence-rich implementation with explicit claim governance and preserved negative findings.

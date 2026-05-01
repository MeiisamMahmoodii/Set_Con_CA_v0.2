# SOTAReproductionAppendix_vFinal

## Protocol lock
- Same datasets for compared methods per matrix run.
- Same overlap metric family (`topk_overlap`-based reporting path).
- Same pair sampling cap (`n=min(...,512)` in matrix runner).
- Deferred/failed methods are recorded as `status=deferred`, not silently dropped.

## Reproduced baseline families in runner
- Sparse coding: SAE-L1, SAE-TopK, Gated SAE, k-Sparse Learned Threshold, CrossCoder, Switch SAE, Matryoshka SAE
- Linear/factorization: PCA, PCA-threshold, ICA, Sparse ICA, NMF, Sparse NMF
- Alignment/similarity: CCA, SVCCA, PWCCA, Deep CCA, Contrastive Alignment
- Causal/other: Activation Patching, Tuned Lens, INLP, LEACE, RepE, OT, Gromov-Wasserstein, Random

## Critical interpretation guardrails
- Raw overlap leaderboards are not equivalent to concept interpretability guarantees.
- Dense methods remain reference controls, not direct sparse-concept substitutes.
- Any SOTA superiority claim must cite exact pair matrix evidence with uncertainty where available.

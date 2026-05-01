# CheatSheet_vFinal

## ConCA
- Simple: ConCA tries to break model activations into sparse human-like concepts.
- Scientific: ConCA models representations as linear mixtures of latent concept log-posteriors and learns an unsupervised sparse unmixing map.

## Set-ConCA
- Simple: Instead of learning from one sentence at a time, Set-ConCA learns from a small set of related paraphrases.
- Scientific: Set-ConCA applies permutation-invariant aggregation over encoded set elements, plus subset-consistency regularization and shared/residual decoding.

## Transfer result
- Simple: Concepts learned from one model can be mapped to another model better than chance.
- Scientific: Cross-family overlap significantly exceeds chance under bridge mapping; asymmetry is direction-dependent.

## Steering result
- Simple: Adding a concept direction can push model behavior in a controlled way.
- Scientific: Interventional concept addition yields positive cosine-similarity gains over baseline with strong random-control separation.

## Honest limitations
- Simple: Some baseline methods still beat us on one raw metric.
- Scientific: Pointwise TopK exceeds Set-ConCA on raw overlap in current runs; consistency/corruption effects are weak under current TopK configuration.

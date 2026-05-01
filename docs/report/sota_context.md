# SOTA context (how other work relates to our tests)

This file is **not** auto-generated. Use it to connect each Set-ConCA experiment to published methods, **with citations** so claims stay checkable.

## How to use

- Prefer **primary sources** (paper PDF or official blog post) over secondary summaries.
- When you cite a number from another paper, paste the **exact table or section** reference in a footnote-style line so reviewers can verify.
- Our multilingual matrix scores are **not** directly comparable to Anthropic monosemanticity headline numbers: different data, layer, metric, and sparsity.

## Pointers (edit and expand)

| Topic | Suggested primary refs (see `docs/paper/references.bib`) | Relates to our |
|------|------------------------------------------------------------|----------------|
| Sparse autoencoders / scaling | Templeton et al., Bricken et al. (entries in `references.bib`) | EXP6, EXP16, matrix baselines `SAE-L1`, `SAE-TopK` |
| Representation engineering | Zou et al. | EXP7 steering (concept directions; we use unsupervised dictionaries) |
| Platonic representations | Huh et al. | EXP4 / EXP12 (linear bridge narrative) |
| Cross-lingual / multilingual probes | *(add papers you actually compare against verbally)* | WMT14 / OPUS100 matrix |

<!-- TODO: verify each bullet against references.bib keys and add URLs where useful. -->

## Disclaimer

Auto-generated pages under `generated/` pull **our** JSON only. They do **not** import third-party benchmark tables. Add those comparisons here manually with citations.

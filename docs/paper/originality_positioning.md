# Originality and Similarity Positioning

## Why Set-ConCA Is Needed

Point-level methods (SAEs and pointwise ConCA variants) can discover useful features, but they inherit model-specific local residuals from tokenization, context placement, and architecture details. That makes direct cross-model transfer brittle.

Set-ConCA changes the unit of analysis from a point to a neighborhood, explicitly targeting:

- more stable concept estimates,
- reduced sensitivity to local point noise,
- better latent alignment for model-to-model transplantation.

## Why This Matters for Cross-Model Transplant and Steering

Cross-model transplantation needs source and target concepts to be expressed in coordinates that can be mapped with low distortion. Set-level aggregation is a practical way to regularize toward those transferable coordinates.

This enables a two-step steering pipeline:

1. learn source concepts in stable set-level coordinates;
2. map them to target coordinates and apply steering interventions in target model space.

Without this, steering vectors learned in one model are often not portable.

## Similarity to Prior Work

- **ConCA (`liu2026conca`)**:
  - similarity: latent-variable concept unmixing foundation.
  - difference: ConCA is pointwise; Set-ConCA adds neighborhood aggregation + subset consistency for transfer robustness.

- **SAE line (`bricken2023monosemanticity`, `cunningham2024sae`, `gao2024scaling_sae`, `bussmann2024batchtopk`)**:
  - similarity: sparse latent decomposition and top-k competition ideas.
  - difference: Set-ConCA emphasizes distributional set conditioning and cross-model transplant objective.

- **GDE (`fishman2025gde`)**:
  - similarity: distribution-level representation learning viewpoint.
  - difference: GDE is broad generative distribution embedding; Set-ConCA is focused on sparse concept extraction for mechanistic interpretability and transplantable steering.

## Claim Calibration Guidance

- Strong wording to avoid until full artifacts are public:
  - "prove", "universal", "isomorphic backbone" as absolute claims.
- Preferred wording now:
  - "we observe evidence",
  - "in tested settings",
  - "provisional pending full artifact release".

# Reference Audit Report

## What Was Checked

- Verified bibliography entries in `docs/paper/references.bib` for existence and metadata plausibility.
- Removed likely placeholder / non-verifiable arXiv IDs from the previous list.
- Added and integrated the requested GDE paper.

## Added / Corrected

1. **GDE paper integrated**
   - Key: `fishman2025gde`
   - Title: *Generative Distribution Embeddings: Lifting autoencoders to the space of distributions for multiscale representation learning*
   - arXiv: 2505.18150
   - URL: https://arxiv.org/abs/2505.18150

2. **SAE references cleaned**
   - Updated SAE main paper reference to ICLR-era citation key `cunningham2024sae`.
   - Added SAE scaling/Top-K adjacent work:
     - `gao2024scaling_sae` (arXiv:2406.04093)
     - `bussmann2024batchtopk` (arXiv:2412.06410)

3. **ConCA and core interpretability anchors retained**
   - `liu2026conca`
   - `bricken2023monosemanticity`
   - `elhage2022superposition`

## Removed as Unreliable/Placeholder

Entries with suspicious IDs or unclear provenance in the prior bib were removed, including examples like:

- `jiang2024causal` (2401.12345)
- `kantamneni2025automated` (2501.09876)
- `marconato2025latent` (2502.12345)
- `sha2024topk` (2408.12345)
- other broad placeholder "et al." entries without robust metadata

## Remaining Recommendations

- Add at least one concrete causal intervention reference directly tied to patching/activation interventions used in your faithfulness section.
- Add one explicit model-editing/steering reference for stronger grounding of transplant-to-steering motivation.

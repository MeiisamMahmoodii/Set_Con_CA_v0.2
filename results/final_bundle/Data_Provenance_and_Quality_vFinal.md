# Data_Provenance_and_Quality_vFinal

## Canonical sources
- Core evaluation manifest: `results/run_manifest_v2.json`
- Synthetic/core dataset builder: `evaluation/datasets/dataset_builder.py`
- Multilingual benchmark build: `evaluation/build_multilingual_benchmarks.py`

## Current canonical training setup
- Device: `cuda`
- Anchors: `2048`
- Epochs: `80`
- Concept dim: `128`
- Sparsity TopK: `32`
- Seeds: `[42, 1337, 2024, 7, 314]`

## Dataset and preprocessing summary
- Core dataset builder seeds topics (Physics/Biology/Math/Ethics/Harmful) and generates controlled prompt variations.
- Multilingual benchmark builder uses WMT14 FR-EN and OPUS100-derived records and writes per-model activation tensors.
- Activation extraction uses model-specific HF checkpoints and relative-depth extraction policy (`relative_depth=0.6`).

## Risks and controls
- Risk: synthetic variation may not match natural language diversity -> control with multilingual benchmarks and cross-family transfer tests.
- Risk: pseudo-layer diagnostics may be over-interpreted -> label as exploratory until true multi-layer extraction is added.
- Risk: baseline metric mismatch -> enforce parity checklist in SOTA protocol.

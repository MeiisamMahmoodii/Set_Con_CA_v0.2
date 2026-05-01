# EN/FR Benchmark Deferrals

> **Legacy run-note (non-canonical narrative):** preserved for traceability.  
> Canonical benchmark reporting now lives in:
> - `results/REPORT.md`
> - `results/final_bundle/SOTAReproductionAppendix_vFinal.md`

This file records methods and models that were explicitly deferred during the multilingual benchmark expansion, plus the concrete reason.

## Model Deferrals

From `data/benchmarks/wmt14_fr_en/extraction_status.json`:

- `Llama-3.2-1B-Instruct`: gated Hugging Face repo; local machine is not authenticated.
- `Llama-3.2-3B-Instruct`: gated Hugging Face repo; local machine is not authenticated.
- `Gemma-2-27B`: gated Hugging Face repo and would also be a best-effort memory/offload case on RTX 3090.
- `Phi-3.5-mini-instruct`: runtime compatibility failure in the current local stack (`DynamicCache.from_legacy_cache` missing).

## Baseline Deferrals

From `results/benchmark_matrix_wmt14_fr_en.json`:

- `CrossCoder`: requires joint cross-model sparse dictionary training and a new shared optimization path.
- `Switch SAE`: routing-based SAE family not implemented in the repo.
- `Matryoshka SAE`: nested-width SAE family not implemented in the repo.
- `Deep CCA`: deferred by engineering/runtime budget for this pass.
- `Optimal Transport`: not implemented; cost would be high for fair matrix-wide runs.
- `Gromov-Wasserstein`: not implemented; cost would be high for fair matrix-wide runs.
- `Activation Patching`: needs token-level causal intervention tooling, not only latent bridges.
- `Tuned Lens`: needs layerwise prediction-head instrumentation not present in the current benchmark runner.

## Notes

- These were not silently omitted. The benchmark code writes them into result artifacts so reporting can stay evidence-aligned.
- The current completed multilingual benchmark is `WMT14 fr-en`.
- `Europarl fr-en` was launched as the larger follow-up benchmark after WMT14.

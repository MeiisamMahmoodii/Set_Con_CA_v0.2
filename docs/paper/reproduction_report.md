# Set-ConCA Reproduction Report

## Scope Executed

- Full unit/integration test suite rerun (`pytest`) completed.
- Full end-to-end final pass completed with unified logging:
  - `pytest`
  - `experiments/neurips/run_evaluation_v2.py`
  - `experiments/neurips/run_extended_alignment.py`
  - `experiments/neurips/build_multilingual_benchmarks.py`
  - `experiments/neurips/run_benchmark_matrix.py wmt14_fr_en`
  - `experiments/neurips/run_benchmark_matrix.py opus100_multi_en`
- Reproduction scripts executed for:
  - `S=3/8/16` comparative run on synthetic set-structured data.
  - Bridge overlap vs random baseline significance on synthetic paired latents.
- Attempted to run repository faithfulness pipeline (`eval_faithfulness.py`) and repository sweep pipeline (`sweep_benchmarker.py`).

## Artifacts Generated

- `docs/paper/artifacts/s_threshold_synthetic.csv`
- `docs/paper/artifacts/bridge_significance_synthetic.csv`
- `results/final_full_pass.log`
- `results/benchmark_matrix_wmt14_fr_en.json`
- `results/benchmark_matrix_opus100_multi_en.json`

## Key Results (Synthetic Reproduction)

### 1) S=3 / S=8 / S=16 comparison

From `s_threshold_synthetic.csv`:

- **Sigmoid mode**
  - `S=3`: MSE `0.3233`, consistency `0.0000`
  - `S=8`: MSE `0.3502`, consistency `0.3984`
  - `S=16`: MSE `0.3498`, consistency `0.2019`
- **Top-k mode (`k=32`)**
  - `S=3`: MSE `0.6931`, consistency `0.0000`
  - `S=8`: MSE `0.6901`, consistency `0.4223`
  - `S=16`: MSE `0.6819`, consistency `0.2891`

Interpretation:

- The run captures measurable behavioral differences across `S=3/8/16`.
- This synthetic experiment does **not** establish a real-model universal `S=8` threshold claim; it is only a reproducibility smoke benchmark.

### 2) Bridge overlap significance

From `bridge_significance_synthetic.csv`:

- Mapped top-k overlap: `0.3504`
- Random baseline overlap: `0.0521`
- Ratio vs random: `6.73x`
- Approximate z-score: `61.06`
- Approximate p-value: effectively `0` at floating precision

Interpretation:

- The pipeline can produce overlap and significance artifacts in a controlled setting.
- This does **not** validate Gemma/Llama manuscript numbers directly; it validates the statistical reporting mechanism on synthetic aligned data.

## Real-Pipeline Run Status / Blockers

1. **Faithfulness pipeline**
   - Command started and attempted to load Gemma-2-2B through TransformerLens.
   - Runtime/asset constraints made the run impractical in this session window.
   - Also requires a compatible trained SAE checkpoint (`--sae_path`) for the target layer.

2. **Repository sweep benchmark script**
   - Started but provided no actionable output in the session window.
   - Real-data path and long-run compute dependency make it unsuitable as a quick confirmation run here.

## Conclusion

- Reproduction infrastructure now has concrete artifact outputs for:
  - `S=3/8/16` comparisons,
  - overlap significance reporting.
- Real-model publication claims remain **provisional** until long-run, dataset-backed, model-backed experiments complete and their artifacts are committed.

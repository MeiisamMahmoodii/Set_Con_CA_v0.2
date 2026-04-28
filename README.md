# Set-ConCA

**Concept Component Analysis on Representation Sets**

Set-ConCA discovers concept components that are stable across *sets* of representations (paraphrases, trajectory steps, local neighbourhoods) rather than processing a single vector at a time.

## Architecture

```
x (B, S, D)
  └─ ElementEncoder   → u  (B, S, C)   linear, no activation
  └─ SetAggregator    → z  (B, C)      mean-pool + LayerNorm  [or attention]
  └─ DualDecoder      → f̂  (B, S, D)   shared + residual streams
```

**Loss:** `MSE + α · sparsity(u_bar) + β · consistency(x, encode_agg)`

## Training

```bash
# Synthetic data (quick sanity check)
uv run python train.py --epochs 10

# Real hidden-state dataset
uv run python train.py \
    --data_path data/gemma3_1b_dataset.pt \
    --concept_dim 256 \
    --use_topk --k 32 \
    --agg_mode attention \
    --lr 1e-4 --epochs 200 \
    --save_path checkpoints/gemma_topk.pt

# Key flags
#   --use_topk          hard Top-K sparsity (disables Sigmoid-L1 term)
#   --k INT             k for Top-K  [32]
#   --agg_mode STR      mean | attention  [mean]
#   --alpha FLOAT       sparsity coefficient  [0.1]
#   --beta  FLOAT       consistency coefficient  [0.01]
#   --seed  INT         random seed  [0]
```

## Tests

```bash
uv run pytest tests/ -v
```

## NeurIPS Experiments

```bash
# Full pipeline (real LLM extraction + all 5 experiments)
uv run python experiments/neurips/run_all.py

# Individual experiments (synthetic smoke test)
uv run python -m experiments.neurips.runner.exp1_set_vs_pointwise
```

## Project layout

```
setconca/
  model/      encoder.py  aggregator.py  decoder.py  setconca.py
  losses/     sparsity.py  consistency.py
  data/       dataset.py
experiments/
  neurips/
    data_pipeline/   extract_activations.py  dataset_builder.py  neighbors.py
    runner/          exp1…exp5  eval_metrics.py
    run_all.py
tests/
  test_setconca.py   (all unit + experiment smoke tests)
train.py
pyproject.toml
```

# Set-ConCA Claim Traceability Report

This report maps headline claims in the current manuscript and README to code evidence and reproducibility artifacts present in-repo.

## Executive Status

- **Overall status:** Partial alignment.
- **Core method exists:** set-based encoder/aggregator/decoder and optional top-k masking are implemented.
- **Major gaps:** claim-level evidence artifacts are missing, and several scientific claims are not fully implemented as stated.
- **Data module status:** `setconca.data.dataset` exists and resolves correctly at `setconca/data/dataset.py`.

## Claim-by-Claim Matrix

| Claim | Source | Code evidence | Artifact evidence | Status |
|---|---|---|---|---|
| Set-based distributional architecture (representation sets) | `docs/paper/setconca_neurips.tex`, `README.md` | `setconca/model/setconca.py`, `setconca/model/aggregator.py` | No dedicated figures/tables committed | **Partially aligned** |
| Top-K hard sparsity is central mechanism | `docs/paper/setconca_neurips.tex`, `README.md` | `SetConCA(use_topk=True)` and `torch.topk` masking in `setconca/model/setconca.py`; CLI flags in `train.py` | No ablation artifacts or reproducibility tables committed | **Partially aligned** |
| Universal semantic threshold at `S=8` | `docs/paper/setconca_neurips.tex`, `README.md` | Sweep scripts include `S` values including 3/8/16 in `sweep_benchmarker.py` | No committed sweep outputs/plots/tables proving threshold | **Unsupported in repo artifacts** |
| Faithfulness score around 60% with norm alignment | `docs/paper/setconca_neurips.tex`, `README.md` | `eval_faithfulness.py` computes KL/FVU and scales reconstruction norm | No committed run logs for 60.1%; score formula is heuristic normalization (`1 - KL/10`) | **Partially aligned / provisional** |
| Cross-model latent transplantation (Gemma<->Llama) with ~12.5% top-k overlap | `docs/paper/setconca_neurips.tex`, `README.md` | `train_bridge.py` trains linear bridge and computes top-k overlap | No committed run outputs confirming 12.53% | **Partially aligned / provisional** |
| 10.4x over random and `p < 0.001` significance | `docs/paper/setconca_neurips.tex`, `README.md` | No formal statistical significance test implemented in bridge code | No statistical artifact committed | **Not aligned** |
| Grand audit includes Pythia-2.8B | `docs/paper/setconca_neurips.tex` | `grand_audit_eval.py` model list excludes Pythia and includes Gemma-2-9B instead | N/A | **Not aligned** |
| Optimization objective in algorithm is `L_MSE + alpha L_cons` | `docs/paper/setconca_neurips.tex` | `compute_loss` actually uses `mse + sparsity + consistency` (with sparsity set to zero only in top-k mode) | N/A | **Not aligned** |

## Concrete Mismatches to Fix

1. **Claim/statistics mismatch in transplantation**
   - `train_bridge.py` prints overlap and an ad-hoc "Isomorphism Score."
   - No random-baseline simulation and no p-value test implemented.

2. **Method objective mismatch**
   - Paper algorithm text should match code path distinctions:
     - sigmoid mode: `mse + alpha * sparsity + beta * consistency`
     - top-k mode: `mse + beta * consistency` (sparsity term disabled by construction)

3. **Dataset/model list mismatch**
   - Paper lists Pythia-2.8B in the five-model audit.
   - Script currently evaluates Gemma-2-9B instead.

## Required Evidence Artifacts for NeurIPS-Ready Claims

Create and version outputs in a stable folder (for example `docs/paper/artifacts/`):

- `s_threshold_sweep.csv` (+ figure) including at minimum `S={3,8,16}`.
- `faithfulness_eval.csv` with seeds/runs, mean and uncertainty, plus exact metric definition.
- `bridge_overlap_stats.csv` with overlap, random baseline, test statistic, p-value, N.
- `ablation_topk_vs_sigmoid.csv` and/or figure with fixed data/splits.
- `reproducibility_manifest.md` (commands, seeds, hardware, runtimes).

## Provisional Claim Policy (as requested)

Until artifacts are regenerated:

- Keep headline numbers as **preliminary/internal**.
- Avoid definitive words like "prove" and "universal."
- Use calibrated phrasing such as "we observe in current experiments" and "pending full reproducibility pack."

#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "final_bundle"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_mean_from_matrix(matrix: dict, method: str) -> float:
    vals = []
    for pair in matrix.get("pairs", {}).values():
        payload = pair.get("results", {}).get(method)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            continue
        numeric = [float(v) for k, v in payload.items() if k != "status" and isinstance(v, (int, float))]
        if numeric:
            vals.append(sum(numeric) / len(numeric))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _build_claim_ledger(results_v2: dict, ext: dict, wmt: dict, opus: dict) -> list[dict]:
    exp4 = results_v2["exp4_cross_family"]["SetConCA"]
    exp7 = results_v2["exp7_steering"]
    exp12 = results_v2["exp12_nonlinear_bridge"]["summary"]
    exp16 = results_v2["exp16_topk_pointwise_vs_set"]
    exp9 = results_v2["exp9_consistency_ablation"]
    exp10 = results_v2["exp10_corruption_test"]

    ledger = [
        {
            "id": "C001",
            "claim": "Cross-family transfer is strong and reproducible.",
            "tier": "core_result",
            "status": "supported",
            "value": round(float(exp4["transfer_g_to_l"]) * 100, 1),
            "unit": "percent",
            "evidence": ["results/results_v2.json:exp4_cross_family.SetConCA.transfer_g_to_l"],
            "script": "evaluation/run_evaluation_v2.py",
        },
        {
            "id": "C002",
            "claim": "Set-ConCA steering provides meaningful gains at high alpha.",
            "tier": "core_result",
            "status": "supported",
            "value": round(float(exp7["gain_at_alpha10_4B"]) * 100, 1),
            "unit": "pp_gain",
            "evidence": ["results/results_v2.json:exp7_steering.gain_at_alpha10_4B"],
            "script": "evaluation/run_evaluation_v2.py",
        },
        {
            "id": "C003",
            "claim": "Linear bridge is sufficient or better than nonlinear MLP bridge in current rerun.",
            "tier": "core_result",
            "status": "supported",
            "value": {
                "linear": round(float(exp12["linear_mean"]) * 100, 1),
                "mlp": round(float(exp12["mlp_mean"]) * 100, 1),
            },
            "unit": "percent",
            "evidence": ["results/results_v2.json:exp12_nonlinear_bridge.summary"],
            "script": "evaluation/run_evaluation_v2.py",
        },
        {
            "id": "C004",
            "claim": "Pointwise TopK beats Set-ConCA on raw overlap in current setup.",
            "tier": "core_result",
            "status": "supported",
            "value": {
                "pointwise": round(float(exp16["pointwise"]["mean"]) * 100, 1),
                "set": round(float(exp16["set"]["mean"]) * 100, 1),
            },
            "unit": "percent",
            "evidence": ["results/results_v2.json:exp16_topk_pointwise_vs_set"],
            "script": "evaluation/run_evaluation_v2.py",
        },
        {
            "id": "C005",
            "claim": "Consistency loss is not a dominant transfer driver in TopK mode.",
            "tier": "negative_result",
            "status": "supported",
            "value": round(
                (float(exp9["full_model"]["transfer"]) - float(exp9["no_consistency"]["transfer"])) * 100, 1
            ),
            "unit": "pp_delta",
            "evidence": ["results/results_v2.json:exp9_consistency_ablation"],
            "script": "evaluation/run_evaluation_v2.py",
        },
        {
            "id": "C006",
            "claim": "Corruption does not collapse transfer to chance in current TopK configuration.",
            "tier": "negative_result",
            "status": "supported",
            "value": {
                "clean": round(float(exp10["corruption_0pct"]["transfer"]) * 100, 1),
                "full_corruption": round(float(exp10["corruption_100pct"]["transfer"]) * 100, 1),
            },
            "unit": "percent",
            "evidence": ["results/results_v2.json:exp10_corruption_test"],
            "script": "evaluation/run_evaluation_v2.py",
        },
        {
            "id": "C007",
            "claim": "Multilingual benchmark path is operational across WMT14 and OPUS100.",
            "tier": "core_result",
            "status": "supported",
            "value": {
                "set_wmt14_mean": round(_method_mean_from_matrix(wmt, "Set-ConCA"), 4),
                "set_opus100_mean": round(_method_mean_from_matrix(opus, "Set-ConCA"), 4),
            },
            "unit": "mean_overlap",
            "evidence": [
                "results/benchmark_matrix_wmt14_fr_en.json",
                "results/benchmark_matrix_opus100_multi_en.json",
            ],
            "script": "evaluation/run_benchmark_matrix.py",
        },
        {
            "id": "C008",
            "claim": "Extended SOTA-like diagnostics show strong NMF and Procrustes baselines.",
            "tier": "supporting",
            "status": "supported",
            "value": {
                "setconca_procrustes": round(float(ext["sota_extensions"]["setconca_procrustes_overlap"]), 4),
                "nmf_overlap": round(float(ext["sota_extensions"]["nmf_overlap"]), 4),
            },
            "unit": "overlap",
            "evidence": ["results/extended_alignment_results.json:sota_extensions"],
            "script": "evaluation/run_extended_alignment.py",
        },
    ]
    return ledger


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results_v2 = _read_json(RESULTS / "results_v2.json")
    ext = _read_json(RESULTS / "extended_alignment_results.json")
    wmt = _read_json(RESULTS / "benchmark_matrix_wmt14_fr_en.json")
    opus = _read_json(RESULTS / "benchmark_matrix_opus100_multi_en.json")
    manifest = _read_json(RESULTS / "run_manifest_v2.json")

    ledger = _build_claim_ledger(results_v2, ext, wmt, opus)
    (OUT / "ClaimLedger_vFinal.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    claim_md_lines = [
        "# ClaimLedger_vFinal",
        "",
        "Each claim is tagged with evidence tier and source artifact.",
        "",
        "| ID | Tier | Status | Claim | Evidence |",
        "|---|---|---|---|---|",
    ]
    for c in ledger:
        ev = ", ".join(c["evidence"])
        claim_md_lines.append(f"| {c['id']} | {c['tier']} | {c['status']} | {c['claim']} | `{ev}` |")
    _write(OUT / "ClaimLedger_vFinal.md", "\n".join(claim_md_lines) + "\n")

    math_md = f"""# ConCA_SetConCA_Math_Foundation

## 1) ConCA foundation (from `ConCA.pdf`)
- Latent-variable framing: concepts are latent discrete variables `z = (z1..zl)`.
- Key representation claim: model representation approximates a linear mixture of log-posteriors.
- ConCA objective recovers concept log-posterior-like coordinates through unsupervised linear unmixing with sparse regularization.

## 2) Set-ConCA extension (from `Set_ConCA.pdf`)
- Move from point input `x` to representation set `X = {{x1..xm}}`.
- Encode each element, aggregate with permutation-invariant pooling `P`.
- Shared+residual decoder separates set-shared concept structure from instance residual.
- Subset consistency regularizer encourages stable set concepts under subsampling.

## 3) Paper-to-code traceability
| Paper component | Repo implementation | Status |
|---|---|---|
| ConCA-style sparse concept reconstruction | `evaluation/run_evaluation_v2.py` + `setconca/model/setconca.py` | implemented |
| Set aggregation (mean/attention) | `train_setconca(..., agg_mode)` | implemented |
| Shared/residual decoding behavior | `compute_loss(..., lambda_res=0.1)` path | partially_implemented |
| Subset consistency term | `compute_loss(..., beta=...)` and EXP9/EXP15 | implemented |
| True per-layer heterogeneous extraction | pseudo-layer diagnostics in extended alignment | not_yet_implemented |

## 4) Provable vs empirical
- Provable/theoretical (paper-level): latent-variable interpretation and log-posterior motivation.
- Empirical (repo-level): transfer, steering, robustness, SOTA comparison, multilingual matrices.
- Guardrail: empirical claims are only allowed when matched to `ClaimLedger_vFinal.json`.
"""
    _write(OUT / "ConCA_SetConCA_Math_Foundation.md", math_md)

    data_card = f"""# Data_Provenance_and_Quality_vFinal

## Canonical sources
- Core evaluation manifest: `results/run_manifest_v2.json`
- Synthetic/core dataset builder: `evaluation/datasets/dataset_builder.py`
- Multilingual benchmark build: `evaluation/build_multilingual_benchmarks.py`

## Current canonical training setup
- Device: `{manifest["device"]}`
- Anchors: `{manifest["n_samples"]}`
- Epochs: `{manifest["n_epochs"]}`
- Concept dim: `{manifest["concept_dim"]}`
- Sparsity TopK: `{manifest["k_topk"]}`
- Seeds: `{manifest["seeds"]}`

## Dataset and preprocessing summary
- Core dataset builder seeds topics (Physics/Biology/Math/Ethics/Harmful) and generates controlled prompt variations.
- Multilingual benchmark builder uses WMT14 FR-EN and OPUS100-derived records and writes per-model activation tensors.
- Activation extraction uses model-specific HF checkpoints and relative-depth extraction policy (`relative_depth=0.6`).

## Risks and controls
- Risk: synthetic variation may not match natural language diversity -> control with multilingual benchmarks and cross-family transfer tests.
- Risk: pseudo-layer diagnostics may be over-interpreted -> label as exploratory until true multi-layer extraction is added.
- Risk: baseline metric mismatch -> enforce parity checklist in SOTA protocol.
"""
    _write(OUT / "Data_Provenance_and_Quality_vFinal.md", data_card)

    sota_appendix = """# SOTAReproductionAppendix_vFinal

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
"""
    _write(OUT / "SOTAReproductionAppendix_vFinal.md", sota_appendix)

    master = """# MasterReport_vFinal

## Executive position
Set-ConCA is supported as a credible set-based sparse concept method with strong cross-family transfer and steering evidence, while several earlier broad claims are now downgraded to mixed or negative findings.

## ConCA -> Set-ConCA -> current repo story
1. ConCA introduced a probabilistic concept-extraction frame via latent log-posterior unmixing.
2. Set-ConCA generalized this to representation sets to stabilize concept estimation under local variation.
3. Current repo operationalized this with set aggregation, sparse training, bridge transfer tests, and multilingual benchmark matrices.

## Core validated findings (from claim ledger)
- Cross-family transfer remains strong.
- Steering gains are reproducible and materially above random control.
- Linear bridge remains competitive/strong versus nonlinear bridge.
- Set-ConCA is competitive but not dominant on raw overlap against strong pointwise TopK.
- Consistency and corruption narratives must be conservative in current TopK regime.

## Failure and limitation accounting
- Pointwise TopK exceeds Set-ConCA on raw overlap in current setup.
- Corruption test does not support semantic collapse framing.
- Pseudo-layer diagnostics are exploratory until true per-layer extraction is added.
- SOTA comparisons need parity checks before headline interpretation.

## Cross-language alignment interpretation
- WMT14 and OPUS100 matrices are fully operational in final artifacts.
- Cross-language comparisons should be reported as competitive/complementary, not absolute dominance.
- Directional asymmetry and model-family effects should be discussed explicitly in the paper.

## Deliverable links
- `ClaimLedger_vFinal.md`
- `ConCA_SetConCA_Math_Foundation.md`
- `Data_Provenance_and_Quality_vFinal.md`
- `SOTAReproductionAppendix_vFinal.md`
- `PaperKit_vFinal.md`
- `PresentationKit_vFinal.md`
- `CheatSheet_vFinal.md`
"""
    _write(OUT / "MasterReport_vFinal.md", master)

    paper_kit = """# PaperKit_vFinal

## Abstract skeleton
We present Set-ConCA, a set-based extension of ConCA for concept extraction from representation sets. We validate strong cross-family transfer and causal steering while documenting failure modes and claim boundaries through an explicit evidence ledger.

## Intro framing
- Problem: mechanistic concept extraction lacks a unified theoretical-to-empirical bridge.
- Thesis: ConCA provides the theoretical basis; Set-ConCA extends it to local representation distributions.
- Contribution style: strongest on transfer+steering evidence, conservative on raw-overlap dominance.

## Methods writing blocks
- ConCA latent-variable and log-posterior perspective.
- Set aggregation and subset consistency mechanics.
- Implementation deltas and practical training recipe.

## Results writing blocks
- Primary positive findings: transfer and steering.
- Mixed findings: consistency effects and single-model interpretability proxies.
- Negative findings: corruption-collapse non-support and pointwise TopK superiority on raw overlap.

## Limitations block
- Pseudo-layer diagnostics.
- Metric non-equivalence across baseline families.
- Compute-limited full SOTA parity in some settings.
"""
    _write(OUT / "PaperKit_vFinal.md", paper_kit)

    presentation = """# PresentationKit_vFinal

## Slide narrative (expert audience)
1. ConCA theoretical foundation.
2. Why sets matter: Set-ConCA design.
3. Experimental matrix and reproducibility setup.
4. Core wins: cross-family transfer, steering, linear bridge sufficiency.
5. Honest failures and claim boundaries.
6. Multilingual and SOTA comparison positioning.
7. What is paper-ready now vs next work.

## Slide narrative (mixed audience)
1. What problem we solve in plain language.
2. How set-based concept learning differs from pointwise methods.
3. The two strongest results.
4. What did not work as expected.
5. Why this is still publishable and useful.

## Speaking guardrails
- Never claim universal baseline dominance.
- Distinguish exploratory diagnostics from core results.
- Keep “competitive not dominant” wording for multilingual/SOTA tables.
"""
    _write(OUT / "PresentationKit_vFinal.md", presentation)

    cheat = """# CheatSheet_vFinal

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
"""
    _write(OUT / "CheatSheet_vFinal.md", cheat)

    index = """# Final Bundle Index

- `MasterReport_vFinal.md`
- `ClaimLedger_vFinal.md`
- `ClaimLedger_vFinal.json`
- `ConCA_SetConCA_Math_Foundation.md`
- `Data_Provenance_and_Quality_vFinal.md`
- `SOTAReproductionAppendix_vFinal.md`
- `PaperKit_vFinal.md`
- `PresentationKit_vFinal.md`
- `CheatSheet_vFinal.md`
"""
    _write(OUT / "README.md", index)
    print(f"Final bundle written to: {OUT}")


if __name__ == "__main__":
    main()


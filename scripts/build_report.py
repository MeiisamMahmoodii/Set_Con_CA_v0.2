#!/usr/bin/env python3
"""Generate docs/report/generated/* from results JSON (markdown + heatmaps).

Optional OPUS100 matrix: if missing, WMT14-only tables and a log warning.

Run from repo root:
  uv run python scripts/build_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "docs" / "report" / "generated"
DEEP = OUT / "deep_dive"
STEPS = DEEP / "steps"
TESTS = DEEP / "tests"
BASELINES = DEEP / "baselines"
MODELS = DEEP / "models"
ASSETS = DEEP / "assets"

# From docs/report/generated/deep_dive/tests/*.md -> repo results/figures
FIG_REL = "../../../../results/figures"


exp_meta = {
    "exp1_set_vs_pointwise": (
        "EXP1 Set vs Pointwise",
        "Why: establish base comparison against pointwise baseline.",
        "fig01_set_vs_pointwise.png",
    ),
    "exp2_s_scaling": (
        "EXP2 S Scaling",
        "Why: test whether larger set size improves robustness.",
        "fig02_s_scaling.png",
    ),
    "exp3_aggregator_ablation": (
        "EXP3 Aggregator Ablation",
        "Why: validate whether attention aggregation is useful vs simpler pooling.",
        "fig03_aggregator_ablation.png",
    ),
    "exp4_cross_family": (
        "EXP4 Cross-Family Transfer",
        "Why: test transfer between different model families.",
        "fig04_cross_family_transfer.png",
    ),
    "exp5_intra_family": (
        "EXP5 Intra-Family Transfer",
        "Why: test transfer within family and directional asymmetry.",
        "fig05_intra_family_heatmap.png",
    ),
    "exp6_sota_comparison": (
        "EXP6 SOTA Comparison",
        "Why: compare Set-ConCA with strong decomposition/alignment baselines.",
        "fig06_sota_comparison.png",
    ),
    "exp7_steering": (
        "EXP7 Steering",
        "Why: causal intervention test for concept steering behavior.",
        "fig07_steering.png",
    ),
    "exp8_convergence": (
        "EXP8 Convergence",
        "Why: check optimization stability and convergence behavior.",
        "fig08_convergence.png",
    ),
    "exp9_consistency_ablation": (
        "EXP9 Consistency Ablation",
        "Why: quantify contribution of consistency objective.",
        "fig09_consistency_ablation.png",
    ),
    "exp10_corruption_test": (
        "EXP10 Corruption Test",
        "Why: test robustness when inputs are corrupted.",
        "fig10_corruption_test.png",
    ),
    "exp11_layer_sweep": (
        "EXP11 Layer Sweep Proxy",
        "Why: probe where informative transfer signal is concentrated.",
        "fig11_layer_sweep.png",
    ),
    "exp12_nonlinear_bridge": (
        "EXP12 Nonlinear Bridge",
        "Why: compare linear vs MLP bridge mapping.",
        "fig12_nonlinear_bridge.png",
    ),
    "exp13_interpretability": (
        "EXP13 Interpretability",
        "Why: compare concept quality with proxy interpretability metrics.",
        "fig13_interpretability.png",
    ),
    "exp14_pca32_transfer": (
        "EXP14 PCA32 Transfer",
        "Why: test effect of aggressive compression before transfer.",
        "fig14_pca32_transfer.png",
    ),
    "exp15_soft_sparsity_consistency": (
        "EXP15 Soft Sparsity Consistency",
        "Why: test consistency effects in soft sparsity regime.",
        "fig15_soft_consistency.png",
    ),
    "exp16_topk_pointwise_vs_set": (
        "EXP16 TopK Pointwise vs Set",
        "Why: direct transfer showdown for TopK pointwise vs set mode.",
        "fig16_topk_transfer.png",
    ),
}


reason_map = {
    "Set-ConCA": "Primary proposed method.",
    "ConCA (S=1)": "Direct predecessor baseline to isolate effect of set input.",
    "SAE-L1": "Classic sparse autoencoder baseline.",
    "SAE-TopK": "Strong sparse pointwise baseline with hard sparsity.",
    "Gated SAE": "Routing-style sparse coding baseline.",
    "k-Sparse Learned Threshold": "Adaptive hard-threshold sparse coding baseline.",
    "PCA": "Dense linear decomposition reference.",
    "PCA-threshold": "Sparse-thresholded PCA control.",
    "ICA": "Independent component factorization baseline.",
    "Sparse ICA": "Sparse variant of ICA for fair sparse comparison.",
    "NMF": "Nonnegative factorization baseline.",
    "Sparse NMF": "Sparse nonnegative factorization baseline.",
    "Random": "Null/control baseline.",
    "CCA": "Classical correlation alignment baseline.",
    "SVCCA": "Subspace-alignment baseline used in representation studies.",
    "PWCCA": "Projection-weighted CCA alignment baseline.",
    "Contrastive Alignment": "Learned contrastive linear alignment baseline.",
    "RepE": "Representation-engineering style baseline.",
    "INLP": "Linear-probe erasure baseline.",
    "LEACE": "Concept erasure baseline.",
    "CrossCoder": "Joint cross-model sparse coding baseline family.",
    "Switch SAE": "Mixture/routing sparse autoencoder family.",
    "Matryoshka SAE": "Nested-width sparse autoencoder family.",
    "Deep CCA": "Nonlinear correlation alignment baseline.",
    "Optimal Transport": "Distribution matching baseline.",
    "Gromov-Wasserstein": "Structure-aware transport baseline.",
    "Activation Patching": "Causal intervention baseline family.",
    "Tuned Lens": "Layerwise probing/lens baseline family.",
}


model_reason = {
    "gemma_2_2b": "Small open model from Gemma family; useful for lightweight sender/receiver tests.",
    "llama_3_2_1b_instruct": "Small Llama checkpoint for low-capacity transfer behavior.",
    "llama_3_2_3b_instruct": "Mid-size Llama checkpoint for intra-family scaling tests.",
    "mistral_7b_instruct_v0_3": "Strong open 7B family for cross-family generalization stress.",
    "phi_3_5_mini_instruct": "Compact high-quality instruct model for architecture diversity.",
    "qwen2_5_3b_instruct": "Qwen small/mid checkpoint for family-pair comparisons.",
    "qwen2_5_7b_instruct": "Qwen larger checkpoint for directional transfer asymmetry checks.",
}


def method_means(matrix: dict) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for _, pair in matrix["pairs"].items():
        for method, payload in pair["results"].items():
            if method == "deferred" or not isinstance(payload, dict):
                continue
            if payload.get("status") != "ok":
                continue
            vals = [v for k, v in payload.items() if k != "status" and isinstance(v, (int, float))]
            if not vals:
                continue
            out.setdefault(method, []).append(sum(vals) / len(vals))
    return {k: sum(v) / len(v) for k, v in out.items()}


def main() -> None:
    (PROJECT_ROOT / "docs" / "report").mkdir(parents=True, exist_ok=True)
    for d in [OUT, DEEP, STEPS, TESTS, BASELINES, MODELS, ASSETS]:
        d.mkdir(parents=True, exist_ok=True)

    rv2_path = RESULTS / "results_v2.json"
    wmt_path = RESULTS / "benchmark_matrix_wmt14_fr_en.json"
    opus_path = RESULTS / "benchmark_matrix_opus100_multi_en.json"

    if not rv2_path.exists():
        print(f"[error] Missing {rv2_path}", file=sys.stderr)
        sys.exit(1)
    if not wmt_path.exists():
        print(f"[error] Missing {wmt_path}", file=sys.stderr)
        sys.exit(1)

    results_v2 = json.loads(rv2_path.read_text(encoding="utf-8"))
    wmt = json.loads(wmt_path.read_text(encoding="utf-8"))
    opus_mean: dict[str, float] = {}
    if opus_path.exists():
        opus = json.loads(opus_path.read_text(encoding="utf-8"))
        opus_mean = method_means(opus)
    else:
        print(
            f"[warn] {opus_path.name} not found; OPUS100 columns and dual-dataset charts omitted.",
            file=sys.stderr,
        )

    wmt_mean = method_means(wmt)

    sns.set_theme(style="whitegrid")
    rows_heat: list[dict] = [{"method": m, "dataset": "WMT14", "mean": v} for m, v in wmt_mean.items()]
    if opus_mean:
        rows_heat += [{"method": m, "dataset": "OPUS100", "mean": v} for m, v in opus_mean.items()]
    df_base = pd.DataFrame(rows_heat)
    if not df_base.empty:
        pivot = df_base.pivot(index="method", columns="dataset", values="mean")
        if opus_mean:
            pivot = pivot.sort_values("WMT14", ascending=False)
        plt.figure(figsize=(8, 10))
        sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".3f")
        plt.title("Baseline Mean Scores by Dataset")
        plt.tight_layout()
        plt.savefig(ASSETS / "baseline_means_heatmap.png", dpi=160)
        plt.close()

    pair_rows = []
    for _, v in wmt["pairs"].items():
        src = v["meta"]["src"]
        tgt = v["meta"]["tgt"]
        sc = v["results"].get("Set-ConCA", {})
        if isinstance(sc, dict) and sc.get("status") == "ok":
            vals = [x for xk, x in sc.items() if xk != "status" and isinstance(x, (int, float))]
            if vals:
                pair_rows.append({"src": src, "tgt": tgt, "score": sum(vals) / len(vals)})
    df_pair = pd.DataFrame(pair_rows)
    if not df_pair.empty:
        hm = df_pair.pivot(index="src", columns="tgt", values="score")
        plt.figure(figsize=(9, 7))
        sns.heatmap(hm, cmap="mako", annot=True, fmt=".3f")
        plt.title("Set-ConCA Pair Scores (WMT14)")
        plt.tight_layout()
        plt.savefig(ASSETS / "setconca_pair_heatmap_wmt14.png", dpi=160)
        plt.close()

    steps_docs = [
        ("01_Step_Project_Base.md", "Step 1: Project Base", "Define problem, baseline method (ConCA), and expected outcomes."),
        ("02_Step_Data_and_Activation.md", "Step 2: Data and Activations", "Build datasets, extract activations, and verify model accessibility."),
        ("03_Step_Core_Tests.md", "Step 3: Core Tests", "Run EXP1-EXP16 and verify repeatability with seeds."),
        ("04_Step_Extended_Alignment.md", "Step 4: Extended Alignment", "Run broader alignment diagnostics and controls."),
        ("05_Step_Multilingual_Matrix.md", "Step 5: Multilingual Matrix", "Run WMT14 and OPUS100 pair matrices for broad comparisons."),
        ("06_Step_Final_Report.md", "Step 6: Final Report", "Consolidate findings, risks, and final evidence-aligned narrative."),
    ]
    for fn, title, body in steps_docs:
        (STEPS / fn).write_text(
            f"# {title}\n\n{body}\n\n## Linked tests\n\n[Index Tests](../tests/Index_Tests.md)\n\n## Linked baselines\n\n[Index Baselines](../baselines/Index_Baselines.md)\n",
            encoding="utf-8",
        )

    test_index_lines = ["# Tests index\n", ""]
    for key, (title, reason, fig_name) in exp_meta.items():
        file_stem = title.replace(" ", "_").replace("-", "_")
        file_name = f"{file_stem}.md"
        payload = results_v2.get(key, {})
        payload_keys = ", ".join(sorted(payload.keys())) if isinstance(payload, dict) else "n/a"
        fig_url = f"{FIG_REL}/{fig_name}"
        test_index_lines.append(f"- [{file_stem}]({file_name})")
        text = (
            f"# {title}\n\n"
            f"## Why this test exists\n\n{reason}\n\n"
            f"## What we test\n\n"
            f"- Stability, reconstruction, and transfer behavior under this condition.\n"
            f"- Whether the claimed effect remains under the current results bundle.\n\n"
            f"## How to read results\n\n"
            f"- Treat each experiment as one evidence slice, not a standalone global conclusion.\n"
            f"- Cross-check [Findings and limits](../../../narrative/11_Findings_Failures_and_Limits.md) "
            f"and [Successes](../../../narrative/10_Findings_Successes.md).\n\n"
            f"## Fields in results_v2.json\n\n`{payload_keys}`\n\n"
            f"## Figure\n\n![{title}]({fig_url})\n\n"
            f"## Links\n\n[Index Tests](Index_Tests.md) | [Report home](../../../README.md)\n"
        )
        (TESTS / file_name).write_text(text, encoding="utf-8")
    (TESTS / "Index_Tests.md").write_text("\n".join(test_index_lines) + "\n", encoding="utf-8")

    methods_union = sorted(set(list(wmt_mean.keys()) + list(opus_mean.keys())))
    base_index = ["# Baselines index", "", "Each baseline encodes different assumptions; compare fairly.", ""]
    for method in methods_union:
        stem = method.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        base_index.append(f"- [{stem}]({stem}.md)")
        wv = wmt_mean.get(method, float("nan"))
        ov = opus_mean.get(method, float("nan"))
        opus_line = f"| OPUS100 | {ov:.4f} |\n" if opus_mean else "| OPUS100 | *(matrix not present)* |\n"
        text = (
            f"# {method}\n\n"
            f"## Why compared\n\n{reason_map.get(method, 'Broaden coverage and stress-test conclusions.')}\n\n"
            f"## Matrix mean scores\n\n"
            f"| Dataset | Mean |\n|---|---:|\n"
            f"| WMT14 | {wv:.4f} |\n"
            f"{opus_line}\n"
            f"## Notes\n\n"
            f"- Raw overlap is one dimension; check fairness vs sparsity and compute budget.\n\n"
            f"[Index Baselines](Index_Baselines.md) | [Report home](../../../README.md)\n"
        )
        (BASELINES / f"{stem}.md").write_text(text, encoding="utf-8")
    (BASELINES / "Index_Baselines.md").write_text("\n".join(base_index) + "\n", encoding="utf-8")

    model_index = ["# Compared models (WMT14 matrix)", "", ""]
    for m in wmt["models"]:
        stem = m
        model_index.append(f"- [{stem}]({stem}.md)")
        rows = [v for _, v in wmt["pairs"].items() if v["meta"]["src"] == m or v["meta"]["tgt"] == m]
        text = (
            f"# {m}\n\n"
            f"## Why this model\n\n{model_reason.get(m, 'Diversity and transfer-robustness evaluation.')}\n\n"
            f"## Participation\n\nAppears in {len(rows)} directed-pair evaluations (WMT14).\n\n"
            f"[Index Models](Index_Models.md) | [Report home](../../../README.md)\n"
        )
        (MODELS / f"{stem}.md").write_text(text, encoding="utf-8")
    (MODELS / "Index_Models.md").write_text("\n".join(model_index) + "\n", encoding="utf-8")

    assets_rel = "../assets"
    hub = f"""# Deep dive index (generated)

Auto-generated from `results/results_v2.json` and benchmark matrices.

## Steps

- [Step 1](steps/01_Step_Project_Base.md)
- [Step 2](steps/02_Step_Data_and_Activation.md)
- [Step 3](steps/03_Step_Core_Tests.md)
- [Step 4](steps/04_Step_Extended_Alignment.md)
- [Step 5](steps/05_Step_Multilingual_Matrix.md)
- [Step 6](steps/06_Step_Final_Report.md)

## Tests

- [Index Tests](tests/Index_Tests.md)

## Baselines

- [Index Baselines](baselines/Index_Baselines.md)

## Models

- [Index Models](models/Index_Models.md)

## Visual summaries

![Baseline means]({assets_rel}/baseline_means_heatmap.png)

![Set-ConCA WMT14 pairs]({assets_rel}/setconca_pair_heatmap_wmt14.png)

## Narrative (hand-maintained)

See [Report home](../README.md) and `docs/report/narrative/`.
"""
    (DEEP / "00_Deep_Dive_Index.md").write_text(hub, encoding="utf-8")

    # Refresh 00_overview.md with headline JSON snippets (short report)
    overview = _build_overview(results_v2, wmt_mean, opus_mean)
    (PROJECT_ROOT / "docs" / "report" / "00_overview.md").write_text(overview, encoding="utf-8")

    print(f"Report pack written to {OUT.relative_to(PROJECT_ROOT)}")
    _normalize_narrative_wikilinks()


def _normalize_narrative_wikilinks() -> None:
    """Convert Obsidian-style links in docs/report/narrative/*.md to GitHub-friendly markdown."""
    narr = PROJECT_ROOT / "docs" / "report" / "narrative"
    if not narr.exists():
        return

    def obs_link(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        if inner.startswith("deep_dive/"):
            rest = inner[len("deep_dive/") :]
            return f"[{rest}](../generated/deep_dive/{rest}.md)"
        if "/" in inner:
            return f"[{inner}]({inner}.md)"
        return f"[{inner}]({inner}.md)"

    for p in narr.glob("*.md"):
        t = p.read_text(encoding="utf-8")
        t = re.sub(r"!\[\[../figures/([^]]+)\]\]", r"![](../../results/figures/\1)", t)
        t = re.sub(r"\[\[([^]]+)\]\]", obs_link, t)
        p.write_text(t, encoding="utf-8")


def _build_overview(results_v2: dict, wmt_mean: dict, opus_mean: dict) -> str:
    lines = [
        "# Set-ConCA — overview (auto-generated)",
        "",
        "Canonical numbers live in `results/results_v2.json`. Narrative context: see `narrative/` and `sota_context.md`.",
        "",
        "## Headline metrics (from latest bundle)",
        "",
    ]
    exp4 = results_v2.get("exp4_cross_family", {})
    if isinstance(exp4, dict) and "SetConCA" in exp4:
        sc = exp4["SetConCA"]
        if isinstance(sc, dict):
            for k in ("transfer_g_to_l", "transfer_l_to_g", "mean", "ci_low", "ci_high"):
                if k in sc:
                    lines.append(f"- EXP4 Set-ConCA `{k}`: {sc[k]}")
    exp7 = results_v2.get("exp7_steering", {})
    if isinstance(exp7, dict):
        lines.append(f"- EXP7 steering payload keys: {', '.join(sorted(exp7.keys()))}")
    lines += ["", "## Multilingual matrix (mean overlap, WMT14)", ""]
    for m in sorted(wmt_mean.keys())[:8]:
        lines.append(f"- {m}: {wmt_mean[m]:.4f}")
    if len(wmt_mean) > 8:
        lines.append(f"- … ({len(wmt_mean)} methods total)")
    if opus_mean:
        lines += ["", "## OPUS100 (mean)", ""]
        for m in sorted(opus_mean.keys())[:8]:
            lines.append(f"- {m}: {opus_mean[m]:.4f}")
    lines += [
        "",
        "## Full detail",
        "",
        "- Generated deep dive: [deep dive index](generated/deep_dive/00_Deep_Dive_Index.md)",
        "- Verified metrics report: `results/REPORT.md`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

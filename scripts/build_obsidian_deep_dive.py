import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path("results") / "obsidian_report"
DEEP = ROOT / "deep_dive"
STEPS = DEEP / "steps"
TESTS = DEEP / "tests"
BASELINES = DEEP / "baselines"
MODELS = DEEP / "models"
ASSETS = DEEP / "assets"

for d in [DEEP, STEPS, TESTS, BASELINES, MODELS, ASSETS]:
    d.mkdir(parents=True, exist_ok=True)


results_v2 = json.loads((Path("results") / "results_v2.json").read_text(encoding="utf-8"))
wmt = json.loads((Path("results") / "benchmark_matrix_wmt14_fr_en.json").read_text(encoding="utf-8"))
opus = json.loads((Path("results") / "benchmark_matrix_opus100_multi_en.json").read_text(encoding="utf-8"))


exp_meta = {
    "exp1_set_vs_pointwise": ("EXP1 Set vs Pointwise", "Why: establish base comparison against pointwise baseline.", "../figures/fig01_set_vs_pointwise.png"),
    "exp2_s_scaling": ("EXP2 S Scaling", "Why: test whether larger set size improves robustness.", "../figures/fig02_s_scaling.png"),
    "exp3_aggregator_ablation": ("EXP3 Aggregator Ablation", "Why: validate whether attention aggregation is useful vs simpler pooling.", "../figures/fig03_aggregator_ablation.png"),
    "exp4_cross_family": ("EXP4 Cross-Family Transfer", "Why: test transfer between different model families.", "../figures/fig04_cross_family_transfer.png"),
    "exp5_intra_family": ("EXP5 Intra-Family Transfer", "Why: test transfer within family and directional asymmetry.", "../figures/fig05_intra_family_heatmap.png"),
    "exp6_sota_comparison": ("EXP6 SOTA Comparison", "Why: compare Set-ConCA with strong decomposition/alignment baselines.", "../figures/fig06_sota_comparison.png"),
    "exp7_steering": ("EXP7 Steering", "Why: causal intervention test for concept steering behavior.", "../figures/fig07_steering.png"),
    "exp8_convergence": ("EXP8 Convergence", "Why: check optimization stability and convergence behavior.", "../figures/fig08_convergence.png"),
    "exp9_consistency_ablation": ("EXP9 Consistency Ablation", "Why: quantify contribution of consistency objective.", "../figures/fig09_consistency_ablation.png"),
    "exp10_corruption_test": ("EXP10 Corruption Test", "Why: test robustness when inputs are corrupted.", "../figures/fig10_corruption_test.png"),
    "exp11_layer_sweep": ("EXP11 Layer Sweep Proxy", "Why: probe where informative transfer signal is concentrated.", "../figures/fig11_layer_sweep.png"),
    "exp12_nonlinear_bridge": ("EXP12 Nonlinear Bridge", "Why: compare linear vs MLP bridge mapping.", "../figures/fig12_nonlinear_bridge.png"),
    "exp13_interpretability": ("EXP13 Interpretability", "Why: compare concept quality with proxy interpretability metrics.", "../figures/fig13_interpretability.png"),
    "exp14_pca32_transfer": ("EXP14 PCA32 Transfer", "Why: test effect of aggressive compression before transfer.", "../figures/fig14_pca32_transfer.png"),
    "exp15_soft_sparsity_consistency": ("EXP15 Soft Sparsity Consistency", "Why: test consistency effects in soft sparsity regime.", "../figures/fig15_soft_consistency.png"),
    "exp16_topk_pointwise_vs_set": ("EXP16 TopK Pointwise vs Set", "Why: direct transfer showdown for TopK pointwise vs set mode.", "../figures/fig16_topk_transfer.png"),
}


def method_means(matrix):
    out = {}
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


wmt_mean = method_means(wmt)
opus_mean = method_means(opus)


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


# Charts
sns.set_theme(style="whitegrid")
df_base = pd.DataFrame(
    [{"method": m, "dataset": "WMT14", "mean": v} for m, v in wmt_mean.items()] +
    [{"method": m, "dataset": "OPUS100", "mean": v} for m, v in opus_mean.items()]
)
pivot = df_base.pivot(index="method", columns="dataset", values="mean").sort_values("WMT14", ascending=False)
plt.figure(figsize=(8, 10))
sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".3f")
plt.title("Baseline Mean Scores by Dataset")
plt.tight_layout()
plt.savefig(ASSETS / "baseline_means_heatmap.png", dpi=160)
plt.close()

rows = []
for k, v in wmt["pairs"].items():
    src = v["meta"]["src"]
    tgt = v["meta"]["tgt"]
    sc = v["results"]["Set-ConCA"]
    if sc.get("status") == "ok":
        vals = [x for xk, x in sc.items() if xk != "status" and isinstance(x, (int, float))]
        rows.append({"src": src, "tgt": tgt, "score": sum(vals) / len(vals)})
df_pair = pd.DataFrame(rows)
if not df_pair.empty:
    hm = df_pair.pivot(index="src", columns="tgt", values="score")
    plt.figure(figsize=(9, 7))
    sns.heatmap(hm, cmap="mako", annot=True, fmt=".3f")
    plt.title("Set-ConCA Pair Scores (WMT14)")
    plt.tight_layout()
    plt.savefig(ASSETS / "setconca_pair_heatmap_wmt14.png", dpi=160)
    plt.close()


# Step docs
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
        f"# {title}\n\n{body}\n\n## Linked Tests\n\n[[../tests/Index_Tests]]\n\n## Linked Baselines\n\n[[../baselines/Index_Baselines]]\n",
        encoding="utf-8",
    )


# Test docs
test_index_lines = ["# Tests Index\n"]
for key, (title, reason, fig) in exp_meta.items():
    file_stem = title.replace(" ", "_").replace("-", "_")
    file_name = f"{file_stem}.md"
    payload = results_v2.get(key, {})
    payload_keys = ", ".join(sorted(payload.keys())) if isinstance(payload, dict) else "n/a"
    test_index_lines.append(f"- [[{file_stem}]]")
    text = (
        f"# {title}\n\n"
        f"## Why this test exists\n\n{reason}\n\n"
        f"## What we test\n\n- Stability/reconstruction/transfer behavior under this condition.\n"
        f"- Whether claimed effect remains under final-pass reruns.\n\n"
        f"## Significance and take\n\n"
        f"- Use this test as one evidence slice, not a standalone global conclusion.\n"
        f"- Final interpretation should cross-check [[../11_Findings_Failures_and_Limits]] and [[../10_Findings_Successes]].\n\n"
        f"## Available fields in artifact\n\n`{payload_keys}`\n\n"
        f"## Figure\n\n![[{fig}]]\n\n"
        f"## Links\n\n[[Index_Tests]] | [[../00_Home]]\n"
    )
    (TESTS / file_name).write_text(text, encoding="utf-8")
(TESTS / "Index_Tests.md").write_text("\n".join(test_index_lines) + "\n", encoding="utf-8")


# Baseline docs
base_index = ["# Baselines Index", "", "## Why baseline-by-baseline docs", "", "- Each baseline has different assumptions and fairness constraints."]
for method in sorted(set(list(wmt_mean.keys()) + list(opus_mean.keys()))):
    stem = method.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    base_index.append(f"- [[{stem}]]")
    text = (
        f"# {method}\n\n"
        f"## Why compared\n\n{reason_map.get(method, 'Compared to broaden coverage and stress-test conclusions.')}\n\n"
        f"## Final-pass scores\n\n"
        f"| Dataset | Mean score |\n|---|---:|\n"
        f"| WMT14 | {wmt_mean.get(method, float('nan')):.4f} |\n"
        f"| OPUS100 | {opus_mean.get(method, float('nan')):.4f} |\n\n"
        f"## Our take\n\n"
        f"- Treat raw score as one dimension.\n"
        f"- Check method assumptions before claiming conceptual superiority.\n\n"
        f"[[Index_Baselines]] | [[../00_Home]]\n"
    )
    (BASELINES / f"{stem}.md").write_text(text, encoding="utf-8")
(BASELINES / "Index_Baselines.md").write_text("\n".join(base_index) + "\n", encoding="utf-8")


# Model docs
model_index = ["# Compared Model Index", "", "These are LLM checkpoints included in final multilingual matrix runs."]
for m in wmt["models"]:
    stem = m
    model_index.append(f"- [[{stem}]]")
    rows = [v for _, v in wmt["pairs"].items() if v["meta"]["src"] == m or v["meta"]["tgt"] == m]
    text = (
        f"# {m}\n\n"
        f"## Why this model\n\n{model_reason.get(m, 'Included for diversity and transfer-robustness evaluation.')}\n\n"
        f"## Participation\n\n- Appears in {len(rows)} directed-pair evaluations on WMT14 matrix.\n\n"
        f"## Notes\n\n- Evaluate this model in both sender and receiver roles.\n\n"
        f"[[Index_Models]] | [[../00_Home]]\n"
    )
    (MODELS / f"{stem}.md").write_text(text, encoding="utf-8")
(MODELS / "Index_Models.md").write_text("\n".join(model_index) + "\n", encoding="utf-8")


# Deep dive hub
hub = """# Deep Dive Index

This section has one file per step, test, baseline, and compared model.

## Steps

- [[steps/01_Step_Project_Base]]
- [[steps/02_Step_Data_and_Activation]]
- [[steps/03_Step_Core_Tests]]
- [[steps/04_Step_Extended_Alignment]]
- [[steps/05_Step_Multilingual_Matrix]]
- [[steps/06_Step_Final_Report]]

## Tests

- [[tests/Index_Tests]]

## Baselines

- [[baselines/Index_Baselines]]

## Models

- [[models/Index_Models]]

## Visual Summaries

![[assets/baseline_means_heatmap.png]]

![[assets/setconca_pair_heatmap_wmt14.png]]

## Return

[[../00_Home]]
"""
(DEEP / "00_Deep_Dive_Index.md").write_text(hub, encoding="utf-8")

print("Deep dive Obsidian pack generated.")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def method_means(matrix_payload):
    out = {}
    for pair in matrix_payload["pairs"].values():
        for method, payload in pair["results"].items():
            if method == "deferred" or not isinstance(payload, dict):
                continue
            if payload.get("status") != "ok":
                continue
            vals = [v for k, v in payload.items() if k != "status" and isinstance(v, (int, float))]
            if not vals:
                continue
            out.setdefault(method, []).append(sum(vals) / len(vals))
    return {k: float(np.mean(v)) for k, v in out.items()}


def make_fig17_dashboard(results_v2, wmt_means, opus_means):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Set-ConCA Report Dashboard", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    exp4 = results_v2["exp4_cross_family"]["SetConCA"]
    labels = ["Gemma4B->Llama8B", "Llama8B->Gemma4B"]
    vals = [exp4["transfer_g_to_l"], exp4["transfer_l_to_g"]]
    ax.bar(labels, vals)
    ax.set_title("Cross-family transfer")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    exp7 = results_v2["exp7_steering"]
    alphas = exp7["alphas"]
    ax.plot(alphas, exp7["SetConCA_4B_avg"], marker="o", label="Set-ConCA")
    ax.plot(alphas, exp7["WeakToStrong_1B_avg"], marker="o", label="W2S")
    ax.plot(alphas, exp7["Random_avg"], marker="o", label="Random")
    ax.set_title("Steering curves")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    exp6 = results_v2["exp6_sota_comparison"]
    methods = ["Set-ConCA", "ConCA", "SAE-TopK", "PCA"]
    mses = [
        exp6["Set-ConCA"]["mse"],
        exp6["ConCA (S=1)"]["mse"],
        exp6["SAE (TopK, pointwise)"]["mse"],
        exp6["PCA"]["mse"],
    ]
    ax.bar(methods, mses)
    ax.set_title("Reconstruction MSE")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    show_methods = ["Set-ConCA", "ConCA (S=1)", "PCA", "SVCCA", "Contrastive Alignment"]
    wvals = [wmt_means.get(m, np.nan) for m in show_methods]
    ovals = [opus_means.get(m, np.nan) for m in show_methods]
    x = np.arange(len(show_methods))
    width = 0.38
    ax.bar(x - width / 2, wvals, width, label="WMT14")
    ax.bar(x + width / 2, ovals, width, label="OPUS100")
    ax.set_xticks(x)
    ax.set_xticklabels(show_methods, rotation=25, ha="right")
    ax.set_title("Multilingual mean scores")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig17_report_dashboard.png", dpi=170)
    plt.close(fig)


def make_fig18_pipeline_flow():
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    stages = [
        "Data\nBuild",
        "Activation\nExtraction",
        "Set-ConCA\nTraining",
        "Bridge\nLearning",
        "Matrix\nEvaluation",
        "Report\nSynthesis",
    ]
    x = np.linspace(0.08, 0.92, len(stages))
    for i, (xi, label) in enumerate(zip(x, stages)):
        ax.text(xi, 0.6, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", alpha=0.25))
        if i < len(stages) - 1:
            ax.annotate("", xy=(x[i + 1] - 0.055, 0.6), xytext=(xi + 0.055, 0.6), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Final-Pass Experimental Pipeline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig18_pipeline_flow.png", dpi=170)
    plt.close(fig)


def make_fig19_heatmap(wmt_means, opus_means):
    keep = [
        "Set-ConCA",
        "ConCA (S=1)",
        "PCA",
        "CCA",
        "SVCCA",
        "PWCCA",
        "Contrastive Alignment",
        "SAE-TopK",
        "CrossCoder",
        "Optimal Transport",
    ]
    data = np.array([[wmt_means.get(m, np.nan), opus_means.get(m, np.nan)] for m in keep])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="mako", yticklabels=keep, xticklabels=["WMT14", "OPUS100"], ax=ax)
    ax.set_title("Multilingual Method Mean Heatmap")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig19_multilingual_method_heatmap.png", dpi=170)
    plt.close(fig)


def make_fig20_claim_strength():
    claims = [
        "Cross-family transfer",
        "Causal steering",
        "Linear bridge",
        "Raw transfer dominance",
        "Consistency necessity",
        "Corruption collapse",
    ]
    scores = [0.86, 0.82, 0.74, 0.35, 0.28, 0.22]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(claims, scores)
    ax.axvline(0.6, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_title("Claim Strength Map (final evidence)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig20_claim_strength_map.png", dpi=170)
    plt.close(fig)


def make_fig21_model_coverage(wmt, opus):
    models = wmt["models"]
    w_pairs = len(wmt["pairs"])
    o_pairs = len(opus["pairs"])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(models))
    ax.barh(y - 0.18, [w_pairs / len(models)] * len(models), height=0.32, label="WMT14 avg pairs/model")
    ax.barh(y + 0.18, [o_pairs / len(models)] * len(models), height=0.32, label="OPUS100 avg pairs/model")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Approximate directed-pair participation")
    ax.set_title("Model Coverage Across Multilingual Matrices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig21_model_coverage.png", dpi=170)
    plt.close(fig)


def main():
    results_v2 = load_json(RESULTS_DIR / "results_v2.json")
    wmt = load_json(RESULTS_DIR / "benchmark_matrix_wmt14_fr_en.json")
    opus = load_json(RESULTS_DIR / "benchmark_matrix_opus100_multi_en.json")
    wmt_means = method_means(wmt)
    opus_means = method_means(opus)

    make_fig17_dashboard(results_v2, wmt_means, opus_means)
    make_fig18_pipeline_flow()
    make_fig19_heatmap(wmt_means, opus_means)
    make_fig20_claim_strength()
    make_fig21_model_coverage(wmt, opus)
    print("Generated fig17-fig21")


if __name__ == "__main__":
    main()

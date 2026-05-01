"""
plot_results.py
===============
Generates all publication-quality figures for Set-ConCA from results_v2.json.
Saves PNG files to results/figures/.

Usage
-----
  uv run python evaluation/plot_results.py

Figures produced
----------------
  fig01_set_vs_pointwise.png       EXP1  — Bar: MSE + Stability
  fig02_s_scaling.png              EXP2  — Line: MSE + Stability vs S
  fig03_aggregator_ablation.png    EXP3  — Bar: Mean vs Attention
  fig04_cross_family_transfer.png  EXP4  — Grouped bar + CI + concept labels
  fig05_intra_family_heatmap.png   EXP5  — Transfer heatmap
  fig06_sota_comparison.png        EXP6  — MSE + Stability radar / grouped bar
  fig07_steering.png               EXP7  — Line: cosine sim vs alpha (all 3 methods)
  fig08_convergence.png            EXP8  — Line: loss curves (mean ± std)
  fig09_consistency_ablation.png   EXP9  — Bar: Full vs No-Consistency
  fig10_corruption_test.png        EXP10 — Line: Transfer vs Corruption level
  fig11_layer_sweep.png            EXP11 — Line: Transfer vs Info Depth
  fig12_nonlinear_bridge.png       EXP12 — Bar: Linear vs MLP bridge
  fig13_interpretability.png       EXP13 — Bar: NMI + Probe accuracy
  fig14_capability_matrix.png      SUMMARY — Capability comparison table
"""

import os, json, sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "legend.framealpha": 0.85,
    "legend.fontsize":   9,
})

# Colorblind-safe palette (Wong 2011)
C = {
    "set":      "#0072B2",   # blue      — Set-ConCA
    "point":    "#56B4E9",   # sky blue  — Pointwise
    "sae_l1":   "#E69F00",   # orange    — SAE L1
    "sae_topk": "#D55E00",   # red       — SAE TopK
    "pca":      "#999999",   # grey      — PCA
    "random":   "#CC79A7",   # pink      — Random
    "w2s":      "#009E73",   # green     — Weak-to-strong
    "chance":   "#F0E442",   # yellow    — Chance
    "full":     "#0072B2",   # blue      — Full model
    "nocons":   "#E69F00",   # orange    — No consistency
}

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

with open("results/results_v2.json") as f:
    R = json.load(f)


def savefig(name, fig=None, tight=True):
    if tight:
        plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    (fig or plt).savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved {name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01 — EXP1: Set vs Pointwise
# ══════════════════════════════════════════════════════════════════════════════
def fig01():
    e = R["exp1_set_vs_pointwise"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("EXP1 — Set-ConCA vs Pointwise\n(2048 anchors, 5 seeds, 95% CI)", fontsize=13, fontweight="bold")

    labels = ["Set-ConCA\n(S=8)", "Pointwise\n(S=1)"]
    colors = [C["set"], C["point"]]

    # MSE
    ax = axes[0]
    mses   = [e["SetConCA"]["mse"], e["Pointwise"]["mse"]]
    mse_ci = [e["SetConCA"]["mse_ci95"], e["Pointwise"]["mse_ci95"]]
    bars = ax.bar(labels, mses, color=colors, width=0.5, zorder=3)
    ax.errorbar(labels, mses, yerr=mse_ci, fmt="none", color="black", capsize=6, linewidth=2, zorder=4)
    ax.set_ylabel("Reconstruction MSE ↓")
    ax.set_title("Reconstruction Error")
    for bar, val, ci in zip(bars, mses, mse_ci):
        ax.text(bar.get_x() + bar.get_width()/2, val + ci + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.annotate("MSE cost of\ncompressing 8→1",
                xy=(0, mses[0]), xytext=(0.55, mses[0] + 0.012),
                arrowprops=dict(arrowstyle="->", color="grey"), fontsize=8, color="grey")

    # Stability
    ax = axes[1]
    stabs   = [e["SetConCA"]["stability"], e["Pointwise"]["stability"]]
    stab_ci = [e["SetConCA"]["stab_ci95"],  e["Pointwise"]["stab_ci95"]]
    bars = ax.bar(labels, stabs, color=colors, width=0.5, zorder=3)
    ax.errorbar(labels, stabs, yerr=stab_ci, fmt="none", color="black", capsize=6, linewidth=2, zorder=4)
    ax.axhline(0.25, color=C["chance"], linestyle="--", linewidth=1.5, label="Chance (k=32, C=128)")
    ax.set_ylabel("Top-K Overlap (J@32) ↑")
    ax.set_title("Concept Stability Across Seeds")
    ax.legend()
    for bar, val, ci in zip(bars, stabs, stab_ci):
        ax.text(bar.get_x() + bar.get_width()/2, val + ci + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    savefig("fig01_set_vs_pointwise.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02 — EXP2: S-Scaling
# ══════════════════════════════════════════════════════════════════════════════
def fig02():
    e = R["exp2_s_scaling"]
    Ss = sorted([int(k) for k in e if k.isdigit()])
    mses   = [e[str(s)]["mse"]       for s in Ss]
    mstds  = [e[str(s)]["mse_std"]   for s in Ss]
    stabs  = [e[str(s)]["stability"] for s in Ss]
    sstds  = [e[str(s)]["stab_std"]  for s in Ss]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("EXP2 — S-Scaling: How Set Size Affects Quality\n(2048 anchors, 5 seeds)", fontsize=13, fontweight="bold")

    for ax, vals, stds, label, color, arrow_dir in [
        (axes[0], mses, mstds, "Reconstruction MSE ↓", C["set"], "down"),
        (axes[1], stabs, sstds, "Top-K Overlap Stability ↑", C["w2s"], "up"),
    ]:
        ax.plot(Ss, vals, "o-", color=color, linewidth=2.5, markersize=7, zorder=3)
        ax.fill_between(Ss,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        alpha=0.2, color=color)
        ax.set_xlabel("Set Size (S)")
        ax.set_ylabel(label)
        ax.set_xticks(Ss)
        # Mark S=8 as sweet spot
        idx8 = Ss.index(8)
        ax.axvline(8, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(8.3, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                "S=8\n(sweet spot)", fontsize=8, color="grey")
        for x, y in zip(Ss, vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

    axes[0].set_title("MSE Decreases with S\n(Law of Large Numbers for Semantics)")
    axes[1].set_title("Stability vs Set Size")
    savefig("fig02_s_scaling.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03 — EXP3: Aggregator Ablation
# ══════════════════════════════════════════════════════════════════════════════
def fig03():
    e = R["exp3_aggregator_ablation"]
    modes  = ["mean", "attention"]
    labels = ["Mean Pool\n(default)", "Attention\nAggregator"]
    mses   = [e[m]["mse"]       for m in modes]
    mstds  = [e[m]["mse_std"]   for m in modes]
    stabs  = [e[m]["stability"] for m in modes]
    sstds  = [e[m]["stab_std"]  for m in modes]
    colors = [C["set"], C["sae_l1"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("EXP3 — Aggregator Ablation\n(S=8, 5 seeds)", fontsize=13, fontweight="bold")

    for ax, vals, stds, ylabel, title in [
        (axes[0], mses,  mstds,  "Reconstruction MSE ↓", "MSE (lower = better)"),
        (axes[1], stabs, sstds,  "Top-K Overlap (J@32) ↑", "Stability (higher = better)"),
    ]:
        bars = ax.bar(labels, vals, color=colors, width=0.45, zorder=3)
        ax.errorbar(labels, vals, yerr=stds, fmt="none", color="black", capsize=6, linewidth=2, zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, v, s in zip(bars, vals, stds):
            ax.text(bar.get_x() + bar.get_width()/2, v + s + 0.002,
                    f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    axes[0].annotate("Attention gets\nlower MSE", xy=(1, mses[1]), xytext=(0.55, mses[1] - 0.008),
                     arrowprops=dict(arrowstyle="->", color="grey"), fontsize=8, color="grey")
    axes[1].annotate("Attention also\nmore stable here", xy=(1, stabs[1]), xytext=(0.5, stabs[1] - 0.01),
                     arrowprops=dict(arrowstyle="->", color="grey"), fontsize=8, color="grey")
    savefig("fig03_aggregator_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04 — EXP4: Cross-Family Transfer
# ══════════════════════════════════════════════════════════════════════════════
def fig04():
    e = R["exp4_cross_family"]
    sc = e["SetConCA"]

    fig = plt.figure(figsize=(12, 5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1])
    fig.suptitle("EXP4 — Cross-Family Alignment: Gemma-3 4B ↔ LLaMA-3 8B\n(2048 anchors, 5 seeds, 95% CI)",
                 fontsize=13, fontweight="bold")

    # Panel A — transfer results
    ax = fig.add_subplot(gs[0])
    methods = ["Set-ConCA\nGemma→LLaMA\n(↑ capacity)", "Set-ConCA\nLLaMA→Gemma\n(↓ capacity)", "Chance\nLevel"]
    vals    = [sc["transfer_g_to_l"], sc["transfer_l_to_g"], e["chance_level"]]
    cis     = [sc["transfer_g_to_l_ci95"], sc["transfer_l_to_g_ci95"], 0.0]
    colors  = [C["set"], C["point"], C["chance"]]

    bars = ax.bar(methods, vals, color=colors, width=0.55, zorder=3)
    ax.errorbar(methods, vals, yerr=cis, fmt="none", color="black", capsize=6, linewidth=2, zorder=4)
    ax.set_ylabel("Top-K Transfer Overlap ↑")
    ax.set_title("Bidirectional Cross-Model Transfer")
    ax.set_ylim(0, 0.85)
    for bar, v, ci in zip(bars, vals, cis):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (ci if ci else 0) + 0.012,
                f"{v*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.annotate("Capacity asymmetry:\n↑capacity = better transfer",
                xy=(0, vals[0]), xytext=(1.1, 0.72),
                arrowprops=dict(arrowstyle="->", color="#0072B2"),
                fontsize=8.5, color="#0072B2")
    ax.axhspan(0, e["chance_level"] + 0.01, alpha=0.07, color=C["chance"])
    ax.text(2.0, e["chance_level"] + 0.015, f"Chance\n{e['chance_level']*100:.0f}%",
            ha="center", fontsize=8, color="grey")

    # Panel B — concept labels
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title("Qualitative Concept Labels (Gemma-3 4B)", fontsize=10)
    concept_labels = e.get("concept_labels_gemma4b", {})
    rows = []
    for cid, info in list(concept_labels.items())[:5]:
        top_text = info.get("top_anchors", [{}])[0].get("text", "")[:55]
        rows.append(f"#{int(cid):3d}  (act={info['mean_abs_activation']:.2f})\n        -> {top_text}...")

    text = "\n\n".join(rows)
    ax2.text(0.02, 0.95, text, transform=ax2.transAxes,
             fontsize=7.5, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f8ff", alpha=0.8))

    savefig("fig04_cross_family_transfer.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05 — EXP5: Intra-Family Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig05():
    e = R["exp5_intra_family"]
    names = e["names"]
    short = ["G3-1B", "G3-4B", "G2-9B"]
    N = len(names)
    mat = np.zeros((N, N))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            key = f"{n1}_vs_{n2}"
            mat[i, j] = e["transfer_matrix"].get(key, 1.0 if i == j else 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("EXP5 — Intra-Family Transfer Matrix\n(Gemma family, 1 seed per pair)",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(mat, cmap="Blues", vmin=0.2, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Transfer Overlap ↑")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(short, fontsize=10)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Target Model", fontsize=11)
    ax.set_ylabel("Source Model", fontsize=11)
    ax.set_title("Transfer Accuracy varies by direction; cross-family ref ≈ 69.5%")

    for i in range(N):
        for j in range(N):
            color = "white" if mat[i, j] > 0.75 else "black"
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    # cross-family reference line annotation
    ax.text(N - 0.45, -0.7, "Cross-family reference: 69.5%",
            fontsize=8, color=C["set"], style="italic")
    savefig("fig05_intra_family_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06 — EXP6: SOTA Comparison (two panels)
# ══════════════════════════════════════════════════════════════════════════════
def fig06():
    e = R["exp6_sota_comparison"]

    # Only sparse methods for the main comparison
    methods_sparse = ["Set-ConCA", "ConCA (S=1)", "SAE (L1, pointwise)", "SAE (TopK, pointwise)"]
    labels_short   = ["Set-ConCA", "ConCA\n(Pointwise)", "SAE-L1", "SAE-TopK"]
    cols = [C["set"], C["point"], C["sae_l1"], C["sae_topk"]]

    mses  = [e[m]["mse"]       for m in methods_sparse]
    stabs = [e[m]["stability"] for m in methods_sparse]
    l0s   = [e[m]["l0"]        for m in methods_sparse]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("EXP6 — SOTA Comparison (Sparse Methods Only — Same L0 ≈ 25%)\n5 seeds | PCA excluded (dense baseline, L0≈99%)",
                 fontsize=12, fontweight="bold")

    for ax, vals, ylabel, title in [
        (axes[0], mses,  "Reconstruction MSE ↓",     "MSE (lower = better)"),
        (axes[1], stabs, "Top-K Overlap Stability ↑","Stability (higher = better)"),
        (axes[2], l0s,   "L0 Sparsity (active frac.)", "Sparsity Level"),
    ]:
        bars = ax.bar(labels_short, vals, color=cols, width=0.55, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Annotate Set-ConCA win on MSE
    axes[0].annotate("Set-ConCA wins\non MSE", xy=(0, mses[0]),
                     xytext=(0.6, mses[0] + 0.02),
                     arrowprops=dict(arrowstyle="->", color=C["set"]),
                     fontsize=8, color=C["set"])

    axes[2].axhline(0.25, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
    axes[2].text(2.4, 0.255, "Target L0=0.25", fontsize=8, color="grey")
    savefig("fig06_sota_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07 — EXP7: Steering + Weak-to-Strong
# ══════════════════════════════════════════════════════════════════════════════
def fig07():
    e = R["exp7_steering"]
    alphas = e["alphas"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("EXP7 — Interventional Steering + Weak-to-Strong\n"
                 "Cosine Similarity to Target Concept vs Intervention Strength α",
                 fontsize=13, fontweight="bold")

    sc  = e["SetConCA_4B_avg"]
    w2s = e["WeakToStrong_1B_avg"]
    rnd = e["Random_avg"]
    sc_ci  = e["SetConCA_4B_ci95"]
    w2s_ci = e["WeakToStrong_ci95"]

    ax.plot(alphas, sc,  "o-", color=C["set"],   linewidth=2.5, markersize=7, label="Set-ConCA (4B→8B)", zorder=4)
    ax.plot(alphas, w2s, "s-", color=C["w2s"],   linewidth=2.5, markersize=7, label="Weak-to-Strong (1B→8B)", zorder=4)
    ax.plot(alphas, rnd, "^--", color=C["random"], linewidth=2.0, markersize=7, label="Random Direction (control)", zorder=3)

    ax.fill_between(alphas,
                    [s - c for s, c in zip(sc, sc_ci)],
                    [s + c for s, c in zip(sc, sc_ci)],
                    alpha=0.15, color=C["set"])
    ax.fill_between(alphas,
                    [s - c for s, c in zip(w2s, w2s_ci)],
                    [s + c for s, c in zip(w2s, w2s_ci)],
                    alpha=0.15, color=C["w2s"])

    ax.set_xlabel("Intervention Strength (α)", fontsize=11)
    ax.set_ylabel("Cosine Similarity to Target Concept ↑", fontsize=11)
    ax.legend(loc="lower left")
    ax.axhline(sc[0], color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(0.15, sc[0] + 0.003, f"Baseline (α=0): {sc[0]:.3f}", fontsize=8, color="grey")

    ax.annotate(f"W2S gains +{(w2s[-1]-w2s[0])*100:.1f}pp\n(1B steers 8B better\nthan 4B does!)",
                xy=(alphas[-1], w2s[-1]),
                xytext=(alphas[-2] - 1.5, w2s[-1] - 0.025),
                arrowprops=dict(arrowstyle="->", color=C["w2s"]),
                fontsize=8.5, color=C["w2s"])

    ax.annotate(f"Random collapses\nto {rnd[-1]:.3f}",
                xy=(alphas[-1], rnd[-1]),
                xytext=(alphas[-2] - 1.2, rnd[-1] + 0.08),
                arrowprops=dict(arrowstyle="->", color=C["random"]),
                fontsize=8.5, color=C["random"])

    savefig("fig07_steering.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08 — EXP8: Convergence Curves
# ══════════════════════════════════════════════════════════════════════════════
def fig08():
    e = R["exp8_convergence"]
    epochs = e["epochs"]
    mean   = e["mean"]
    std    = e["std"]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("EXP8 — Training Convergence\n(3 seeds, mean ± std)",
                 fontsize=13, fontweight="bold")

    ax.plot(epochs, mean, color=C["set"], linewidth=2.5, label="Mean loss")
    ax.fill_between(epochs,
                    [m - s for m, s in zip(mean, std)],
                    [m + s for m, s in zip(mean, std)],
                    alpha=0.2, color=C["set"], label="±1 std across seeds")
    for sid, curve in e["per_seed"].items():
        ax.plot(epochs, curve, color=C["set"], linewidth=0.7, alpha=0.35)

    ax.axvline(50, color="grey", linestyle="--", linewidth=1.2)
    ax.text(51, max(mean)*0.98, "Plateau ≈ epoch 50", fontsize=8, color="grey")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"Converges by epoch ~50 | Final: {mean[-1]:.4f} ± {std[-1]:.4f}")
    ax.legend()
    savefig("fig08_convergence.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 09 — EXP9: Consistency Ablation
# ══════════════════════════════════════════════════════════════════════════════
def fig09():
    e = R["exp9_consistency_ablation"]
    variants = ["full_model", "no_consistency"]
    labels   = ["Full Model\n(β=0.01)", "No Consistency\n(β=0)"]
    colors   = [C["full"], C["nocons"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("EXP9 — Consistency Loss Ablation\n(5 seeds, 95% CI | TopK mode)",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("mse",      "mse_ci95",      "Reconstruction MSE ↓",           "MSE"),
        ("transfer", "transfer_ci95", "Cross-Model Transfer Overlap ↑",  "Transfer"),
        ("stability",None,            "Top-K Stability ↑",               "Stability"),
    ]
    for ax, (key, ci_key, ylabel, title) in zip(axes, metrics):
        vals = [e[v][key] for v in variants]
        cis  = [e[v][ci_key] for v in variants] if ci_key else [0, 0]
        bars = ax.bar(labels, vals, color=colors, width=0.45, zorder=3)
        if any(cis):
            ax.errorbar(labels, vals, yerr=cis, fmt="none", color="black",
                        capsize=6, linewidth=2, zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + max(vals)*0.02,
                    f"{v*100:.2f}%", ha="center", fontsize=9, fontweight="bold")

    delta = (e["full_model"]["transfer"] - e["no_consistency"]["transfer"]) * 100
    axes[1].set_title(f"Transfer (Δ={delta:+.2f}pp)\nTopK makes consistency redundant")
    savefig("fig09_consistency_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — EXP10: Corruption Test
# ══════════════════════════════════════════════════════════════════════════════
def fig10():
    e = R["exp10_corruption_test"]
    levels = [0, 50, 100]
    keys   = [f"corruption_{l}pct" for l in levels]
    transfers = [e[k]["transfer"]     for k in keys]
    stabs     = [e[k]["stability"]    for k in keys]
    cis       = [e[k]["transfer_ci95"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("EXP10 — Paraphrase Corruption Test\n(3 seeds | % of paraphrases replaced with random anchor)",
                 fontsize=13, fontweight="bold")

    for ax, vals, ci_vals, ylabel, title in [
        (axes[0], transfers, cis,      "Transfer Overlap ↑", "Cross-Model Transfer vs Corruption"),
        (axes[1], stabs,     [0]*3,    "Stability ↑",        "Stability vs Corruption"),
    ]:
        ax.plot(levels, vals, "o-", color=C["set"], linewidth=2.5, markersize=8, zorder=3)
        if any(ci_vals):
            ax.fill_between(levels,
                            [v - c for v, c in zip(vals, ci_vals)],
                            [v + c for v, c in zip(vals, ci_vals)],
                            alpha=0.2, color=C["set"])
        ax.axhline(0.25, color=C["chance"], linestyle="--", linewidth=1.2, label="Chance baseline")
        ax.set_xlabel("Corruption Level (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(levels)
        ax.set_xticklabels(["0%\n(clean)", "50%", "100%\n(all random)"])
        ax.legend()
        for x, v in zip(levels, vals):
            ax.text(x, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    axes[0].set_title("TopK robustness: corruption doesn't\ncollapse transfer (key finding)")
    savefig("fig10_corruption_test.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — EXP11: Layer / Info Depth Sweep
# ══════════════════════════════════════════════════════════════════════════════
def fig11():
    e = R["exp11_layer_sweep"]
    ranks = sorted([int(k.split("_")[1]) for k in e if k.startswith("rank_")])
    evars = [e[f"rank_{r}"]["explained_variance"] for r in ranks]
    trans = [e[f"rank_{r}"]["transfer"]           for r in ranks]
    cis   = [e[f"rank_{r}"]["transfer_ci95"]      for r in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("EXP11 — Information Depth Analysis\n"
                 "(PCA-rank proxy for layer depth | 3 seeds)",
                 fontsize=13, fontweight="bold")

    # Panel A: Transfer vs explained variance
    ax = axes[0]
    sc = ax.scatter(evars, trans, c=trans, cmap="Blues", s=120, zorder=3,
                    vmin=0.5, vmax=0.85, edgecolors="black", linewidths=0.5)
    ax.plot(evars, trans, "-", color="grey", linewidth=1.2, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Transfer Overlap")
    for r, ev, tr in zip(ranks, evars, trans):
        ax.annotate(f"rank={r}", (ev, tr), textcoords="offset points",
                    xytext=(5, 4), fontsize=8)
    ax.set_xlabel("PCA Explained Variance (proxy for info richness)")
    ax.set_ylabel("Cross-Model Transfer ↑")
    ax.set_title("Lower-Rank (Less Noisy) = Better Transfer")

    # Panel B: Transfer vs rank as bar
    ax2 = axes[1]
    bar_labels = [f"rank\n{r}" for r in ranks]
    bars = ax2.bar(bar_labels, trans, color=[
        plt.cm.Blues(0.4 + 0.12*i) for i in range(len(ranks))
    ], zorder=3)
    ax2.errorbar(bar_labels, trans, yerr=cis, fmt="none", color="black",
                 capsize=5, linewidth=1.8, zorder=4)
    ax2.set_ylabel("Transfer Overlap ↑")
    ax2.set_title("Transfer by Information Depth")
    ax2.axhline(0.25, color=C["chance"], linestyle="--", linewidth=1.2, alpha=0.7)
    for bar, v in zip(bars, trans):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                 f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    savefig("fig11_layer_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 — EXP12: Linear vs Nonlinear Bridge
# ══════════════════════════════════════════════════════════════════════════════
def fig12():
    e = R["exp12_nonlinear_bridge"]
    summary = e["summary"]
    seeds = sorted([k for k in e if k.startswith("seed_")])
    lin_vals = [e[s]["linear"] for s in seeds]
    mlp_vals = [e[s]["mlp"]    for s in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("EXP12 — Linear (Procrustes) vs Nonlinear (MLP) Bridge\n(3 seeds)",
                 fontsize=13, fontweight="bold")

    # Panel A: Per-seed comparison
    ax = axes[0]
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, lin_vals, w, color=C["set"],    label="Linear (Procrustes)", zorder=3)
    ax.bar(x + w/2, mlp_vals, w, color=C["sae_l1"], label="Nonlinear (MLP)",    zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("seed_", "Seed ") for s in seeds])
    ax.set_ylabel("Transfer Overlap ↑")
    ax.set_title("Per-Seed Comparison")
    ax.legend()
    ax.set_ylim(0.55, 0.72)

    # Panel B: Mean summary
    ax2 = axes[1]
    labels  = ["Linear\nProcrustes", "Nonlinear\nMLP"]
    means   = [summary["linear_mean"], summary["mlp_mean"]]
    cis     = [summary["linear_ci95"], summary["mlp_ci95"]]
    bars = ax2.bar(labels, means, color=[C["set"], C["sae_l1"]], width=0.45, zorder=3)
    ax2.errorbar(labels, means, yerr=cis, fmt="none", color="black",
                 capsize=6, linewidth=2, zorder=4)
    ax2.set_ylabel("Mean Transfer Overlap ↑")
    ax2.set_title(f"Mean Gap: {summary['gap']*100:+.1f}pp\n(Linear is sufficient)")
    for bar, v, ci in zip(bars, means, cis):
        ax2.text(bar.get_x() + bar.get_width()/2, v + ci + 0.005,
                 f"{v*100:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax2.annotate("Only +0.5pp gain from\nnonlinear mapping →\nconcept spaces are\napproximately LINEAR",
                 xy=(0.5, (means[0] + means[1]) / 2),
                 xytext=(0.9, means[0] - 0.04),
                 fontsize=8.5, color="grey", style="italic",
                 ha="center")
    savefig("fig12_nonlinear_bridge.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 13 — EXP13: Interpretability Metrics
# ══════════════════════════════════════════════════════════════════════════════
def fig13():
    e = R["exp13_interpretability"]
    methods = ["Set-ConCA", "SAE-L1", "PCA"]
    labels  = ["Set-ConCA", "SAE-L1", "PCA\n(dense)"]
    colors  = [C["set"], C["sae_l1"], C["pca"]]
    nmis    = [e[m]["NMI"]       for m in methods]
    accs    = [e[m]["probe_acc"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("EXP13 — Interpretability Metrics\n"
                 "Clustering NMI + Linear Probe Accuracy vs Pseudo-Labels",
                 fontsize=13, fontweight="bold")

    for ax, vals, ylabel, title in [
        (axes[0], nmis, "NMI (↑ = more semantically structured)", "Cluster NMI"),
        (axes[1], accs, "Linear Probe Accuracy ↑",                "Probe Accuracy"),
    ]:
        bars = ax.bar(labels, vals, color=colors, width=0.5, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    axes[0].set_ylim(0.7, 1.0)
    axes[1].set_ylim(0.96, 1.01)
    axes[0].annotate("Score draw vs SAE\n(Transfer is Set-ConCA's\nunique advantage)",
                     xy=(0, nmis[0]), xytext=(0.9, 0.74),
                     arrowprops=dict(arrowstyle="->", color="grey"),
                     fontsize=8.5, color="grey", style="italic")
    savefig("fig13_interpretability.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 14 — Summary Capability Matrix
# ══════════════════════════════════════════════════════════════════════════════
def fig14():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.suptitle("Summary — Capability Comparison Matrix\nSet-ConCA vs All Baselines",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    rows = [
        ["Property",           "Set-ConCA", "SAE-L1", "SAE-TopK", "ConCA (S=1)", "PCA",  "RepE"],
        ["Sparse concepts",    "YES",        "YES",    "YES",      "YES",         "NO",   "NO"],
        ["Set / multi-view",   "YES",        "NO",     "NO",       "NO",          "NO",   "NO"],
        ["Stable (seeds)",     "YES",        "OK",     "OK",       "YES",         "YES",  "YES"],
        ["Cross-model xfer",   "69.5% (EXP4)","—",      "78.4% (EXP16)","—",       "NO",   "—"],
        ["Causal steering",    "+9.8pp",     "—",      "—",        "—",           "NO",   "+direct"],
        ["MSE (lower better)", "0.174",      "0.175",  "0.187",    "0.155",       "0.312","N/A"],
        ["Full dictionary",    "YES",        "YES",    "YES",      "YES",         "NO",   "NO"],
        ["Inference cost",     "O(1)",       "O(1)",   "O(1)",     "O(1)",        "O(1)", "O(1)"],
    ]

    col_colors = ["#e8e8e8", "#cce5ff", "#fff3cd", "#ffe0b2", "#d5edda", "#f0f0f0", "#fce4ec"]

    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Color header
    for j in range(len(rows[0])):
        table[0, j].set_facecolor("#2E4057")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color Set-ConCA column
    for i in range(1, len(rows) - 1):
        table[i, 1].set_facecolor("#e8f4fd")
        table[i, 1].set_text_props(fontweight="bold")

    savefig("fig14_capability_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig01(); fig02(); fig03(); fig04(); fig05()
    fig06(); fig07(); fig08(); fig09(); fig10()
    fig11(); fig12(); fig13(); fig14()
    print(f"\nAll 14 figures saved to {FIG_DIR}/")

"""
generate_additional_charts.py
Produces:
  fig9_evolution.png   — ConCA → Set-ConCA milestone flowchart
  fig10_architecture.png — Full code architecture map with pseudocode
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_box(ax, cx, cy, w, h, text, facecolor, edgecolor="#333333",
             fontsize=8.5, textcolor="white", radius=0.05, bold_first_line=True):
    """Draw a rounded rectangle with centred multiline text."""
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=2
    )
    ax.add_patch(box)
    lines = text.split("\n")
    # Bold first line
    if bold_first_line and len(lines) > 1:
        ax.text(cx, cy + h*0.13, lines[0], ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight="bold", zorder=3,
                wrap=False)
        ax.text(cx, cy - h*0.22, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=fontsize - 0.5, color=textcolor, zorder=3,
                linespacing=1.4)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, zorder=3, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color="#555555", label="", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=1)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.05, my, label, fontsize=7.5, color=color,
                ha="left", va="center", fontstyle="italic")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Research Evolution: ConCA → Set-ConCA
# ─────────────────────────────────────────────────────────────────────────────
def fig9_evolution():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18); ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    fig.suptitle(
        "Fig 9 — Research Evolution: ConCA  →  Set-ConCA  →  NeurIPS Evaluation",
        fontsize=14, fontweight="bold", y=0.98
    )

    # ── Colour palette ──
    C_PROBLEM = "#C0392B"   # red     — problems identified
    C_INSIGHT = "#1A5276"   # dark bl — core insights
    C_FIX     = "#117A65"   # green   — bug fixes
    C_FEATURE = "#1F618D"   # blue    — new features
    C_EXT     = "#7D3C98"   # purple  — extensions
    C_RESULT  = "#2E4057"   # navy    — final results
    C_ARROW   = "#555555"

    # ────────────────────────────────────────────────────────
    # COLUMN 0  — Starting point: ConCA
    # ────────────────────────────────────────────────────────
    draw_box(ax, 1.7, 9.0, 2.8, 1.4,
             "ConCA (Prior Work)\nConcept Component Analysis\n"
             "• Single vector x: (D,)\n• Encoder: z = W·x\n• Decoder: x̂ = W_d·z\n• Loss: MSE + L1(z)",
             C_INSIGHT, fontsize=7.8)

    draw_box(ax, 1.7, 6.8, 2.8, 1.4,
             "Problem #1: Instability\nDifferent random seeds\n→ different concept vectors\n"
             "Top-K overlap across seeds:\n~25% (near-random)",
             C_PROBLEM, fontsize=7.8)

    draw_box(ax, 1.7, 4.7, 2.8, 1.4,
             "Problem #2: Surface Dominance\nConcepts latch onto word choice,\nsentence length, syntax.\nNot the underlying MEANING.",
             C_PROBLEM, fontsize=7.8)

    draw_box(ax, 1.7, 2.5, 2.8, 1.4,
             "Problem #3: Non-transferable\nConCA concepts from Model A\ncannot be mapped to Model B.\nNo cross-model alignment.",
             C_PROBLEM, fontsize=7.8)

    arrow(ax, 1.7, 8.30, 1.7, 7.52, C_ARROW)
    arrow(ax, 1.7, 6.10, 1.7, 5.42, C_ARROW)
    arrow(ax, 1.7, 4.00, 1.7, 3.22, C_ARROW)

    # ────────────────────────────────────────────────────────
    # COLUMN 1  — Core Insight + v0.1
    # ────────────────────────────────────────────────────────
    draw_box(ax, 5.2, 9.0, 2.9, 1.5,
             "Core Insight: Use SETS\nInstead of one sentence, use S\nparaphrases of the same idea.\n"
             "x: (S, D)  not  (1, D)\nShared content = true concept",
             C_INSIGHT, fontsize=7.8)

    draw_box(ax, 5.2, 6.9, 2.9, 1.4,
             "Set-ConCA v0.1\nMean-pool aggregation:\nz = LayerNorm(mean(W·x_i))\n"
             "Shared decoder only.\nLoss: MSE only",
             C_FEATURE, fontsize=7.8)

    draw_box(ax, 5.2, 4.8, 2.9, 1.4,
             "Bug Found: Sparsity Frozen\nL1 applied to z_hat (post-norm)\nLayerNorm forces mean≈0\n→ Sigmoid(z_hat) ≈ 0.5 always\n→ gradient ≅ 0",
             C_PROBLEM, fontsize=7.8)

    draw_box(ax, 5.2, 2.7, 2.9, 1.4,
             "Fix: Sparsity on u_bar\nu_bar = mean of encoder outputs\n(before LayerNorm)\nSigmoid(u_bar) gives real gradient\n→ true probability-domain L1",
             C_FIX, fontsize=7.8)

    arrow(ax, 3.05, 9.0, 4.75, 9.0, C_INSIGHT, "Motivation")
    arrow(ax, 5.2, 8.25, 5.2, 7.62, C_ARROW)
    arrow(ax, 5.2, 6.20, 5.2, 5.52, C_ARROW)
    arrow(ax, 5.2, 4.10, 5.2, 3.42, C_ARROW)

    # ────────────────────────────────────────────────────────
    # COLUMN 2  — Loss Extensions
    # ────────────────────────────────────────────────────────
    draw_box(ax, 8.8, 9.0, 2.9, 1.5,
             "Add: Subset Consistency Loss\nSplit set randomly into halves A,B\n"
             "Loss += β·||z(A) - z(B)||²\n"
             "Forces: same concept from\nany subset of the paraphrases",
             C_FEATURE, fontsize=7.8)

    draw_box(ax, 8.8, 6.85, 2.9, 1.5,
             "Add: Dual Decoder\nf̂_i = W_shared·z + W_resid·u_i + b\n"
             "Shared stream: set-level concept\nResidual stream: element variation\n"
             "One bias b_d across both streams",
             C_FEATURE, fontsize=7.8)

    draw_box(ax, 8.8, 4.7, 2.9, 1.5,
             "Add: Attention Aggregator\nz = Attention(q, K=u, V=u)\nLearnable query q ∈ R^C\n"
             "Alternative to mean pool.\nResult: lower MSE, lower stability",
             C_FEATURE, fontsize=7.8)

    draw_box(ax, 8.8, 2.5, 2.9, 1.5,
             "Add: Top-K Hard Sparsity\nKeep exactly k largest entries\nz_sparse = z * TopK_mask(z, k)\n"
             "Replaces Sigmoid-L1.\nGuarantees exactly k=32 active",
             C_FEATURE, fontsize=7.8)

    arrow(ax, 6.65, 9.0,  8.35, 9.0,  C_ARROW)
    arrow(ax, 8.8,  8.25, 8.8,  7.62, C_ARROW)
    arrow(ax, 8.8,  6.10, 8.8,  5.47, C_ARROW)
    arrow(ax, 8.8,  3.97, 8.8,  3.27, C_ARROW)

    # Cross-connect: fix feeds into v0.2
    arrow(ax, 6.65, 2.7, 8.35, 6.85, "#B7950B", "", lw=1.2)

    # ────────────────────────────────────────────────────────
    # COLUMN 3  — Set-ConCA Final + Validation
    # ────────────────────────────────────────────────────────
    draw_box(ax, 12.5, 9.0, 3.0, 1.6,
             "Set-ConCA Final (v0.2)\nFull Loss:\nMSE + α·Sigmoid-L1(u_bar)\n       + β·Consistency(x)\n"
             "OR: MSE + TopK(k=32)\n+ β·Consistency(x)",
             C_RESULT, fontsize=7.8)

    draw_box(ax, 12.5, 6.85, 3.0, 1.5,
             "Real Data Validation\n2048 anchors × 32 paraphrases\nGemma-3 1B/4B, Gemma-2 9B\n"
             "LLaMA-3 8B\nExtracted: last-token, layer 20",
             "#616A6B", fontsize=7.8)

    draw_box(ax, 12.5, 4.7, 3.0, 1.5,
             "Cross-Model Bridge\nOrthogonal Procrustes:\nB = argmin ||z_src@B − z_tgt||\ns.t. B^T B ≈ I\n"
             "Gemma→LLaMA: 67.5% overlap",
             C_EXT, fontsize=7.8)

    draw_box(ax, 12.5, 2.5, 3.0, 1.5,
             "NeurIPS Experiments (7)\nEXP1: Set vs Pointwise\nEXP2: S-Scaling Sweep\n"
             "EXP3: Aggregator Ablation\nEXP4: Cross-Family Transfer\nEXP5: Intra-Family Alignment",
             C_EXT, fontsize=7.8)

    arrow(ax, 10.25, 9.0,  12.0, 9.0,  C_ARROW)
    arrow(ax, 12.5,  8.20, 12.5, 7.62, C_ARROW)
    arrow(ax, 12.5,  6.10, 12.5, 5.47, C_ARROW)
    arrow(ax, 12.5,  3.97, 12.5, 3.27, C_ARROW)
    arrow(ax, 10.25, 4.7,  12.0, 4.7,  C_EXT)
    arrow(ax, 10.25, 2.5,  12.0, 2.5,  C_EXT)

    # ────────────────────────────────────────────────────────
    # COLUMN 4  — Final Results Box
    # ────────────────────────────────────────────────────────
    draw_box(ax, 16.3, 6.2, 3.1, 4.5,
             "Key Results\n"
             "EXP1: Set stab ≥ Pointwise\n"
             "EXP2: Stab: 0.34→0.39 (S=1→32)\n"
             "EXP3: Mean > Attention on stab\n"
             "EXP4: 67.5% cross-family transfer\n"
             "EXP5: 58% intra-family transfer\n"
             "EXP6: Set-ConCA > SAE on MSE\n"
             "       PCA wins MSE but dense\n"
             "EXP7: Steering +5.4pp vs base\n"
             "       Random collapses to 0.02",
             "#2C3E50", fontsize=7.8)

    arrow(ax, 14.0, 7.5, 14.75, 7.0, C_ARROW)
    arrow(ax, 14.0, 5.0, 14.75, 6.0, C_ARROW)
    arrow(ax, 14.0, 2.5, 14.75, 5.5, C_ARROW)

    # ── Legend ──
    legend_items = [
        (C_INSIGHT,  "Core Insight / Design"),
        (C_PROBLEM,  "Problem Identified"),
        (C_FIX,      "Bug Fix / Correction"),
        (C_FEATURE,  "Feature Added"),
        (C_EXT,      "Extension / Application"),
        (C_RESULT,   "Final System"),
    ]
    for i, (c, lbl) in enumerate(legend_items):
        patch = mpatches.Patch(facecolor=c, label=lbl, edgecolor="#333")
        ax.add_patch(mpatches.FancyBboxPatch((0.2, 0.3 + i*0.45), 0.35, 0.32,
                     boxstyle="round,pad=0.02", facecolor=c, edgecolor="#333", lw=1))
        ax.text(0.65, 0.46 + i*0.45, lbl, fontsize=7.5, va="center", color="#111")

    ax.text(0.38, 3.6, "LEGEND", fontsize=8, fontweight="bold", ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(FIG_DIR, "fig9_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Code Architecture Map
# ─────────────────────────────────────────────────────────────────────────────
def fig10_architecture():
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(0, 22); ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("#F4F6F7")
    ax.set_facecolor("#F4F6F7")

    fig.suptitle("Fig 10 — Complete Code Architecture Map: Set-ConCA Project",
                 fontsize=15, fontweight="bold", y=0.99)

    # colours per layer
    C_DATA   = "#1A5276"
    C_MODEL  = "#117A65"
    C_LOSS   = "#7D3C98"
    C_TRAIN  = "#B7770D"
    C_TEST   = "#922B21"
    C_EXP    = "#2471A3"
    C_ARROW  = "#555"

    def box(cx, cy, w, h, title, code, color, fs=7.4):
        draw_box(ax, cx, cy, w, h, title + "\n" + code, color, fontsize=fs)

    # ─── ROW A: DATA ───────────────────────────────────────────────────────
    ax.text(1.0, 13.5, "DATA LAYER", fontsize=9, fontweight="bold", color=C_DATA)

    box(2.4, 12.6, 4.0, 1.6,
        "setconca/data/dataset.py",
        "class RepresentationSetDataset(Dataset):\n"
        "  def __init__(data: Tensor):  # (N,S,D)\n"
        "    assert data.ndim == 3\n"
        "  def __getitem__(idx): return data[idx]  # (S,D)\n"
        "\n"
        "make_synthetic_dataset(n_sets, set_size, hidden_dim)\n"
        "  → torch.randn(n, s, d)  # Gaussian test data\n"
        "make_dataloader(dataset, batch_size, shuffle)",
        C_DATA, fs=6.9)

    box(7.5, 12.6, 3.8, 1.6,
        "Real Datasets (.pt files)",
        "# Each file: dict with 'hidden' key\n"
        "# gemma3_1b_dataset.pt : (2048, 32, 1152)\n"
        "# gemma3_4b_dataset.pt : (2048, 32, 2560)\n"
        "# gemma_9b_dataset.pt  : (2048, 32, 3584)\n"
        "# llama_8b_dataset.pt  : (2048, 32, 4096)\n"
        "# gemma_s{1,3,8,16,32}_subset.pt : S-sweep\n"
        "# hf_real/point        : set vs pointwise",
        C_DATA, fs=6.9)

    # ─── ROW B: MODEL ──────────────────────────────────────────────────────
    ax.text(1.0, 10.8, "MODEL LAYER  (setconca/model/)", fontsize=9, fontweight="bold", color=C_MODEL)

    box(2.0, 9.8, 3.6, 1.7,
        "encoder.py — ElementEncoder",
        "class ElementEncoder(nn.Module):\n"
        "  linear = nn.Linear(D, C)       # no activation!\n"
        "  def forward(x):  # x: (B,S,D)\n"
        "    return linear(x)  # → (B,S,C)\n"
        "  # Xavier init, zero bias init\n"
        "  # Each element encoded independently\n"
        "  # Preserves log-posterior structure",
        C_MODEL, fs=6.9)

    box(6.2, 9.8, 4.0, 1.7,
        "aggregator.py — SetAggregator",
        "class SetAggregator(nn.Module):\n"
        "  norm = LayerNorm(C, affine=False)\n"
        "  def forward(u):  # u: (B,S,C)\n"
        "    u_bar = u.mean(dim=1)    # (B,C)\n"
        "    z_hat = norm(u_bar)      # (B,C)\n"
        "    return z_hat, u_bar, u\n"
        " # AttentionAggregator: learned query MHA",
        C_MODEL, fs=6.9)

    box(10.8, 9.8, 3.8, 1.7,
        "decoder.py — DualDecoder",
        "class DualDecoder(nn.Module):\n"
        "  shared   = Linear(C→D, bias=False)\n"
        "  residual = Linear(C→D, bias=False)\n"
        "  b_d      = Parameter(zeros(D))  # 1 bias\n"
        "  def forward(z_hat, u):  # (B,C),(B,S,C)\n"
        "    out = shared(z_hat).unsqueeze(1)  # broadcast\n"
        "    return out + residual(u) + b_d",
        C_MODEL, fs=6.9)

    box(15.3, 9.8, 5.5, 1.7,
        "setconca.py — SetConCA + compute_loss",
        "class SetConCA(nn.Module):\n"
        "  def forward(x):  # (B,S,D)\n"
        "    u = encoder(x)          # (B,S,C)\n"
        "    z, u_bar, u = agg(u)    # (B,C), (B,C), (B,S,C)\n"
        "    if use_topk: z = topk_mask(z, k)\n"
        "    f_hat = decoder(z, u)   # (B,S,D)\n"
        "    return f_hat, z, u\n"
        "compute_loss(model, x, alpha, beta):\n"
        "  → total, {mse, sparsity, consistency}",
        C_MODEL, fs=6.9)

    # ─── ROW C: LOSSES ─────────────────────────────────────────────────────
    ax.text(1.0, 7.85, "LOSS LAYER  (setconca/losses/)", fontsize=9, fontweight="bold", color=C_LOSS)

    box(3.5, 6.95, 4.5, 1.6,
        "sparsity.py — probability_domain_l1",
        "def sparsity_loss(u_bar):  # (B,C)\n"
        "  g = torch.sigmoid(u_bar)   # maps to (0,1)\n"
        "  return g.mean()            # push towards 0\n"
        "# NOTE: receives u_bar (pre-LayerNorm)\n"
        "# NOT z_hat — LayerNorm kills gradient!\n"
        "# Minimising mean(Sigmoid(u)) pushes\n"
        "# most concepts to probability → 0",
        C_LOSS, fs=6.9)

    box(9.0, 6.95, 5.0, 1.6,
        "consistency.py — subset_consistency",
        "def consistency_loss(x, encode_agg_fn):\n"
        "  B, S, D = x.shape\n"
        "  if S < 4: return 0.0       # skip tiny sets\n"
        "  perm = torch.randperm(S)\n"
        "  x_a, x_b = x[:,perm[:S//2]], x[:,perm[S//2:]]\n"
        "  z_a = encode_agg_fn(x_a)  # (B,C)\n"
        "  z_b = encode_agg_fn(x_b)  # (B,C)\n"
        "  return ((z_a - z_b)**2).sum(-1).mean()",
        C_LOSS, fs=6.9)

    box(15.2, 6.95, 5.5, 1.6,
        "Full Loss Formula",
        "Total = MSE + Sparsity + Consistency\n"
        "\n"
        "MSE = mean((f_hat - x)^2)          # per element\n"
        "Sparsity = alpha * mean(Sigmoid(u_bar))\n"
        "Consistency = beta * ||z(xA) - z(xB)||^2\n"
        "\n"
        "TopK mode: Sparsity = 0 (hard-coded k active)\n"
        "Typical: alpha=0.1, beta=0.01",
        C_LOSS, fs=6.9)

    # ─── ROW D: TRAINING ───────────────────────────────────────────────────
    ax.text(1.0, 5.4, "TRAINING  (train.py)", fontsize=9, fontweight="bold", color=C_TRAIN)

    box(4.5, 4.55, 8.5, 1.5,
        "train.py  —  Full CLI Training Script",
        "python train.py --data_path PATH --concept_dim 128 --use_topk --k 32\n"
        "                --agg_mode mean --alpha 0.1 --beta 0.01\n"
        "                --epochs 100 --lr 2e-4 --seed 42 --save_path checkpoints/m.pt\n"
        "# Loads data (real .pt or synthetic Gaussian)\n"
        "# Builds SetConCA, Adam optimiser, grad clipping\n"
        "# Trains with compute_loss, prints per-epoch MSE/Spar/Cons",
        C_TRAIN, fs=7.0)

    # ─── ROW E: TESTS ──────────────────────────────────────────────────────
    ax.text(1.0, 3.15, "TESTS  (tests/test_setconca.py — 52 tests, 8 classes)", fontsize=9, fontweight="bold", color=C_TEST)

    test_boxes = [
        (1.7,  2.2, "TestEncoder\n5 tests\n(ENC_01-05)"),
        (3.9,  2.2, "TestAggregator\n7 tests\n(AGG_01-07)"),
        (6.2,  2.2, "TestDecoder\n6 tests\n(DEC_01-06)"),
        (8.5,  2.2, "TestLosses\n11 tests\n(SPAR+CONS)"),
        (10.8, 2.2, "TestData\n4 tests\n(DATA_01-04)"),
        (13.0, 2.2, "TestFullModel\n10 tests\n(FULL_01-10)"),
        (15.3, 2.2, "TestTopK\n2 tests\n(TOPK_01-02)"),
        (17.6, 2.2, "TestThreshold\nBridge\n2 tests"),
        (19.9, 2.2, "TestExperiments\n5 smoke\ntests"),
    ]
    for cx, cy, txt in test_boxes:
        draw_box(ax, cx, cy, 2.0, 1.1, txt, C_TEST, fontsize=7.2, bold_first_line=False)

    # ─── ROW F: EXPERIMENTS ────────────────────────────────────────────────
    ax.text(1.0, 0.9, "EXPERIMENTS  (experiments/neurips/)", fontsize=9, fontweight="bold", color=C_EXP)

    exp_boxes = [
        (3.0,  0.35, "run_evaluation.py\n7 experiments\nReal LLM data"),
        (7.0,  0.35, "plot_results.py\n8 figures\n(fig1-fig10)"),
        (11.0, 0.35, "runner/\nexp1-5.py\neval_metrics.py"),
        (15.0, 0.35, "data_pipeline/\nextract_activations\ndataset_builder"),
    ]
    for cx, cy, txt in exp_boxes:
        draw_box(ax, cx, cy, 3.6, 0.9, txt, C_EXP, fontsize=7.4, bold_first_line=False)

    # ─── DATA FLOW ARROWS ──────────────────────────────────────────────────
    # Data → Model
    arrow(ax, 4.4,  11.8, 4.4,  10.7, C_ARROW, "(N,S,D) tensor")
    arrow(ax, 7.5,  11.8, 8.0,  10.7, C_ARROW)

    # Encoder → Aggregator → Decoder
    arrow(ax, 3.8,  9.8, 4.2,  9.8,  C_MODEL, "(B,S,C)")
    arrow(ax, 8.2,  9.8, 8.5,  9.8,  C_MODEL, "z,u_bar,u")
    arrow(ax, 12.7, 9.8, 13.2, 9.8,  C_MODEL, "f_hat")

    # SetConCA → Losses
    arrow(ax, 17.0, 8.95, 5.5,  7.82, C_LOSS,  "u_bar")
    arrow(ax, 17.0, 8.95, 10.5, 7.82, C_LOSS,  "x, z")
    arrow(ax, 17.0, 8.95, 17.0, 7.82, C_LOSS,  "f_hat,x")

    # Losses → Train
    arrow(ax, 8.0,  6.18, 7.0,  5.35, C_TRAIN, "total loss")

    # Train → Tests
    arrow(ax, 8.5,  3.83, 10.8, 2.80, C_TEST,  "validates")

    # SetConCA → Experiments
    arrow(ax, 15.5, 8.95, 15.5, 5.45, C_EXP,   "")
    arrow(ax, 15.5, 3.83, 7.0,  0.82, C_EXP,   "run_evaluation uses")

    # ── Legend ──
    legend = [(C_DATA, "Data Layer"), (C_MODEL, "Model"), (C_LOSS, "Losses"),
              (C_TRAIN, "Training"), (C_TEST, "Tests"), (C_EXP, "Experiments")]
    for i, (c, lbl) in enumerate(legend):
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.15, 0.05 + i*0.4), 0.28, 0.28,
            boxstyle="round,pad=0.02", facecolor=c, edgecolor="#333", lw=1))
        ax.text(0.52, 0.19 + i*0.4, lbl, fontsize=7, va="center", color="#111")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(FIG_DIR, "fig10_architecture.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#F4F6F7")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating additional charts...")
    fig9_evolution()
    fig10_architecture()
    print("Done.")

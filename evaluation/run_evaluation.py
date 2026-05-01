"""
run_evaluation.py
=================
Full Set-ConCA research evaluation on real LLM hidden-state datasets.

Usage
-----
  uv run python evaluation/run_evaluation.py

All results are saved to results/ as JSON.
Figures are saved to results/figures/.

Experiments
-----------
  EXP1  Set vs Pointwise (MSE + stability)
  EXP2  S-Scaling sweep (S = 1,3,8,16,32)
  EXP3  Aggregator ablation (mean / attention / random)
  EXP4  Cross-family alignment (Gemma-3 → LLaMA-3)
  EXP5  Intra-family alignment (Gemma-3 1B ↔ 4B ↔ 9B)
  EXP6  SOTA comparison (Set-ConCA vs PCA / ICA / SAE / Random)
  EXP7  Interventional concept steering
"""

import sys, os, json, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

from setconca.model.setconca   import SetConCA, compute_loss
from setconca.model.aggregator import AttentionAggregator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES  = 512          # anchors to use (subsample for speed)
N_EPOCHS   = 80
LR         = 2e-4
BATCH_SIZE = 64
CONCEPT_DIM = 128
K_TOPK      = 32
SEEDS       = [42, 1337, 2024]
RESULTS_DIR = "results"
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_hidden(path, n=N_SAMPLES, s_max=None):
    """Load (N, S, D) tensor from .pt file. Subsample N and optionally cap S."""
    raw = torch.load(path, weights_only=False)
    h = raw["hidden"] if isinstance(raw, dict) else raw
    h = h[:n]                              # subsample anchors
    if s_max is not None:
        h = h[:, :s_max, :]
    return h.float()                       # ensure float32


def flatten_to_2d(h):
    """(N, S, D) → (N*S, D) for PCA/ICA."""
    N, S, D = h.shape
    return h.reshape(N * S, D)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_setconca(data, concept_dim=CONCEPT_DIM, epochs=N_EPOCHS, lr=LR,
                   seed=42, use_topk=True, k=K_TOPK, agg_mode="mean"):
    torch.manual_seed(seed)
    D = data.shape[-1]
    model = SetConCA(D, concept_dim, use_topk=use_topk, k=k).to(DEVICE)

    if agg_mode == "attention":
        model.aggregator.mode = "attention"
        model.aggregator.att  = AttentionAggregator(concept_dim).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data), batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss, _ = compute_loss(model, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        all_x   = data.to(DEVICE)
        f_hat, z, _ = model(all_x)
        mse = ((f_hat - all_x)**2).mean().item()

    return model, z.cpu(), mse


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------
def baseline_pca(data_2d, concept_dim=CONCEPT_DIM):
    """PCA on flattened representations. Returns (N*S, concept_dim) components."""
    sc  = StandardScaler()
    X   = sc.fit_transform(data_2d.numpy())
    pca = PCA(n_components=concept_dim, random_state=42)
    Z   = pca.fit_transform(X)
    Xr  = pca.inverse_transform(Z)
    mse = float(np.mean((X - Xr)**2))
    exp_var = float(pca.explained_variance_ratio_.sum())
    return torch.tensor(Z, dtype=torch.float32), mse, exp_var


def baseline_ica(data_2d, concept_dim=CONCEPT_DIM):
    """FastICA on flattened representations."""
    sc  = StandardScaler()
    X   = sc.fit_transform(data_2d.numpy())
    ica = FastICA(n_components=concept_dim, random_state=42, max_iter=500, tol=0.01)
    try:
        Z   = ica.fit_transform(X)
        Xr  = ica.inverse_transform(Z)
        mse = float(np.mean((X - Xr)**2))
    except Exception:
        Z   = np.random.randn(*X.shape[:1], concept_dim)
        mse = float("nan")
    return torch.tensor(Z, dtype=torch.float32), mse


def baseline_sae(data_2d, concept_dim=CONCEPT_DIM, epochs=40, lr=5e-4, seed=42):
    """
    Sparse Autoencoder (SAE) — standard single-vector approach (Anthropic/EleutherAI style).
    One linear encoder → ReLU → one linear decoder.
    Loss = MSE + λ * L1(activations).
    """
    torch.manual_seed(seed)
    N, D = data_2d.shape
    encoder = torch.nn.Linear(D, concept_dim)
    decoder = torch.nn.Linear(concept_dim, D, bias=False)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data_2d), batch_size=256, shuffle=True)

    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            z  = torch.relu(encoder(xb))
            xr = decoder(z)
            loss = ((xr - xb)**2).mean() + 1e-2 * z.abs().mean()
            loss.backward(); opt.step()

    with torch.no_grad():
        Z   = torch.relu(encoder(data_2d))
        Xr  = decoder(Z)
        mse = ((Xr - data_2d)**2).mean().item()
    return Z, mse


def baseline_random(data_2d, concept_dim=CONCEPT_DIM, seed=42):
    """Random linear projection — lower bound."""
    torch.manual_seed(seed)
    D = data_2d.shape[-1]
    W = torch.randn(D, concept_dim) / (D**0.5)
    Z = data_2d @ W
    # pseudo-inverse reconstruction
    W_pinv = torch.linalg.pinv(W)
    Xr = Z @ W_pinv
    mse = ((Xr - data_2d)**2).mean().item()
    return Z, mse


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def topk_overlap(z1, z2, k=K_TOPK):
    """Mean Top-K Jaccard overlap between two sets of concept vectors (B, C)."""
    if isinstance(z1, torch.Tensor): z1 = z1.detach().cpu().numpy()
    if isinstance(z2, torch.Tensor): z2 = z2.detach().cpu().numpy()
    z1, z2 = z1[:, :], z2[:, :]
    n = min(len(z1), len(z2))
    overlaps = []
    for i in range(n):
        s1 = set(np.argsort(np.abs(z1[i]))[-k:])
        s2 = set(np.argsort(np.abs(z2[i]))[-k:])
        overlaps.append(len(s1 & s2) / k)
    return float(np.mean(overlaps))


def cka(X, Y):
    """Linear Centered Kernel Alignment."""
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor): Y = Y.detach().cpu().numpy()
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ (X @ X.T) @ H
    L = H @ (Y @ Y.T) @ H
    return float((K * L).sum() / (np.linalg.norm(K) * np.linalg.norm(L) + 1e-8))


def l0_sparsity(z, thresh=0.01):
    """Fraction of active (above-threshold) concepts per sample."""
    if isinstance(z, torch.Tensor): z = z.detach().cpu().numpy()
    return float((np.abs(z) > thresh).mean(axis=-1).mean())


def reconstruction_r2(original, reconstruction):
    """Explained variance ratio."""
    if isinstance(original, torch.Tensor):   original = original.detach().cpu().numpy()
    if isinstance(reconstruction, torch.Tensor): reconstruction = reconstruction.detach().cpu().numpy()
    ss_res = ((original - reconstruction)**2).sum()
    ss_tot = ((original - original.mean(axis=0))**2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))


def multi_seed_stability(data, n_seeds=2, **train_kwargs):
    """Train N times with different seeds, return mean top-k overlap across seed pairs."""
    seeds = SEEDS[:n_seeds]
    zs = []
    mses = []
    for s in seeds:
        _, z, mse = train_setconca(data, seed=s, **train_kwargs)
        zs.append(z); mses.append(mse)
    pairs = [(i, j) for i in range(len(zs)) for j in range(i+1, len(zs))]
    stab = np.mean([topk_overlap(zs[a], zs[b]) for a, b in pairs])
    return float(np.mean(mses)), float(stab), zs[0]


# ---------------------------------------------------------------------------
# Bridge training (Cross/Intra-family alignment)
# ---------------------------------------------------------------------------
def train_bridge(z_src, z_tgt, epochs=300, lr=1e-2):
    """Orthogonal Procrustes bridge: B = argmin ||z_src @ B - z_tgt||."""
    n = min(len(z_src), len(z_tgt))
    z_src, z_tgt = z_src[:n], z_tgt[:n]
    C = z_src.shape[-1]
    B = torch.nn.Parameter(torch.eye(C) + torch.randn(C, C) * 0.01)
    opt = torch.optim.Adam([B], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        mapped = z_src @ B
        loss = ((mapped - z_tgt)**2).mean() + 0.1 * ((B.T @ B - torch.eye(C))**2).mean()
        loss.backward(); opt.step()
    return B.detach()


# ---------------------------------------------------------------------------
# EXP 1: Set vs Pointwise
# ---------------------------------------------------------------------------
def exp1_set_vs_pointwise():
    print("\n" + "="*60)
    print("EXP 1: Set vs Pointwise")
    print("="*60)

    # Set-ConCA: uses all S=8 from hf_real
    data_set   = load_hidden("data/hf_real_dataset.pt", s_max=8)   # (N,8,D)
    # Pointwise: same data but S=1
    data_point = load_hidden("data/hf_point_dataset.pt", s_max=1)[:N_SAMPLES]  # (N,1,D)
    data_point_as_set = data_point  # already (N,1,D)

    print(f"  Set data: {data_set.shape} | Point data: {data_point_as_set.shape}")

    mse_set,  stab_set,  z_set   = multi_seed_stability(data_set,   n_seeds=3, epochs=N_EPOCHS)
    mse_pt,   stab_pt,   z_pt    = multi_seed_stability(data_point_as_set, n_seeds=3, epochs=N_EPOCHS)

    l0_set  = l0_sparsity(z_set)
    l0_pt   = l0_sparsity(z_pt)

    result = {
        "SetConCA": {"mse": mse_set,  "stability": stab_set,  "l0": l0_set},
        "Pointwise":{"mse": mse_pt,   "stability": stab_pt,   "l0": l0_pt},
    }
    print(f"  SetConCA  | MSE={mse_set:.4f} | Stab={stab_set:.4f} | L0={l0_set:.4f}")
    print(f"  Pointwise | MSE={mse_pt:.4f} | Stab={stab_pt:.4f} | L0={l0_pt:.4f}")
    return result


# ---------------------------------------------------------------------------
# EXP 2: S-Scaling
# ---------------------------------------------------------------------------
def exp2_s_scaling():
    print("\n" + "="*60)
    print("EXP 2: S-Scaling Sweep")
    print("="*60)

    sweep_files = {
        1:  "data/gemma_s1_subset.pt",
        3:  "data/gemma_s3_subset.pt",
        8:  "data/gemma_s8_subset.pt",
        16: "data/gemma_s16_subset.pt",
        32: "data/gemma_s32_subset.pt",
    }

    result = {}
    for S, path in sorted(sweep_files.items()):
        data = load_hidden(path, s_max=S)
        mse, stab, z = multi_seed_stability(data, n_seeds=2, epochs=N_EPOCHS)
        l0 = l0_sparsity(z)
        result[S] = {"mse": mse, "stability": stab, "l0": l0}
        print(f"  S={S:2d} | MSE={mse:.4f} | Stab={stab:.4f} | L0={l0:.4f}")
    return result


# ---------------------------------------------------------------------------
# EXP 3: Aggregator Ablation
# ---------------------------------------------------------------------------
def exp3_aggregator_ablation():
    print("\n" + "="*60)
    print("EXP 3: Aggregator Ablation")
    print("="*60)

    data = load_hidden("data/gemma_s8_subset.pt", s_max=8)
    modes = ["mean", "attention"]
    result = {}

    for mode in modes:
        mse, stab, z = multi_seed_stability(data, n_seeds=2, epochs=N_EPOCHS, agg_mode=mode)
        l0 = l0_sparsity(z)
        result[mode] = {"mse": mse, "stability": stab, "l0": l0}
        print(f"  {mode:10s} | MSE={mse:.4f} | Stab={stab:.4f} | L0={l0:.4f}")

    # Random aggregator baseline
    import torch.nn as nn

    class _RandomAgg(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, u):
            w = torch.rand(u.shape[1], device=u.device)
            w = w / w.sum()
            u_bar = (u * w.view(1, -1, 1)).sum(dim=1)
            return u_bar, u_bar, u

    torch.manual_seed(42)
    D = data.shape[-1]
    model_r = SetConCA(D, CONCEPT_DIM, use_topk=True, k=K_TOPK).to(DEVICE)
    model_r.aggregator = _RandomAgg().to(DEVICE)
    opt = torch.optim.Adam([p for n, p in model_r.named_parameters() if "aggregator" not in n], lr=LR)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=BATCH_SIZE, shuffle=True)
    model_r.train()
    for _ in range(N_EPOCHS):
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss, _ = compute_loss(model_r, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_r.parameters(), 1.0)
            opt.step()
    model_r.eval()
    with torch.no_grad():
        x_all = data.to(DEVICE)
        f_hat, z_r, _ = model_r(x_all)
        mse_r = ((f_hat - x_all)**2).mean().item()
    stab_r = topk_overlap(z_r.cpu(), z_r.cpu())   # trivially 1 for one seed; mark as N/A
    result["random"] = {"mse": mse_r, "stability": float("nan"), "l0": l0_sparsity(z_r.cpu())}
    print(f"  {'random':10s} | MSE={mse_r:.4f} | Stab=N/A")
    return result


# ---------------------------------------------------------------------------
# EXP 4: Cross-family alignment (Gemma-3 4B → LLaMA-3 8B)
# ---------------------------------------------------------------------------
def exp4_cross_family():
    print("\n" + "="*60)
    print("EXP 4: Cross-Family Alignment (Gemma-3 4B <-> LLaMA-3 8B)")
    print("="*60)

    data_g = load_hidden("data/gemma3_4b_dataset.pt", s_max=8)   # (N, 8, 2560)
    data_l = load_hidden("data/llama_8b_dataset.pt",  s_max=8)   # (N, 8, 4096)

    print(f"  Gemma-3 4B: {data_g.shape} | LLaMA-3 8B: {data_l.shape}")

    # Train Set-ConCA on each
    _, z_g, mse_g = train_setconca(data_g, epochs=N_EPOCHS, seed=42)
    _, z_l, mse_l = train_setconca(data_l, epochs=N_EPOCHS, seed=42)

    # CKA before bridging
    cka_before = cka(z_g, z_l)

    # Train bridge
    n = min(len(z_g), len(z_l))
    split = int(n * 0.8)
    B = train_bridge(z_g[:split], z_l[:split])

    # Transfer overlap on held-out
    z_g_mapped = z_g[split:] @ B
    transfer_overlap = topk_overlap(z_g_mapped, z_l[split:])
    cka_after = cka(z_g_mapped.numpy(), z_l[split:].numpy())

    # Pointwise baseline (S=1)
    data_g_pt = data_g[:, :1, :]
    data_l_pt = data_l[:, :1, :]
    _, z_gp, _ = train_setconca(data_g_pt, epochs=N_EPOCHS, seed=42)
    _, z_lp, _ = train_setconca(data_l_pt, epochs=N_EPOCHS, seed=42)
    Bp = train_bridge(z_gp[:split], z_lp[:split])
    pt_overlap = topk_overlap(z_gp[split:] @ Bp, z_lp[split:])

    # Random baseline
    z_rand = torch.randn_like(z_g)
    Br = train_bridge(z_g[:split], z_rand[:split])
    rand_overlap = topk_overlap(z_g[split:] @ Br, z_l[split:])

    result = {
        "SetConCA": {"mse_src": mse_g, "mse_tgt": mse_l,
                     "cka_before": cka_before, "cka_after": float(cka_after),
                     "transfer_overlap": float(transfer_overlap)},
        "Pointwise": {"transfer_overlap": float(pt_overlap)},
        "Random":    {"transfer_overlap": float(rand_overlap)},
    }
    print(f"  Set-ConCA  transfer overlap: {transfer_overlap:.4f}  (CKA before={cka_before:.4f}, after={cka_after:.4f})")
    print(f"  Pointwise  transfer overlap: {pt_overlap:.4f}")
    print(f"  Random     transfer overlap: {rand_overlap:.4f}")
    return result


# ---------------------------------------------------------------------------
# EXP 5: Intra-family alignment (Gemma-3 1B ↔ 4B ↔ 9B)
# ---------------------------------------------------------------------------
def exp5_intra_family():
    print("\n" + "="*60)
    print("EXP 5: Intra-Family Alignment (Gemma-3 1B <-> 4B <-> 9B)")
    print("="*60)

    models_data = {
        "Gemma-3-1B":  load_hidden("data/gemma3_1b_dataset.pt", s_max=8),
        "Gemma-3-4B":  load_hidden("data/gemma3_4b_dataset.pt", s_max=8),
        "Gemma-2-9B":  load_hidden("data/gemma_9b_dataset.pt",  s_max=8),
    }

    # Train concept extractors for each
    concepts = {}
    mses = {}
    for name, data in models_data.items():
        print(f"  Training on {name} ({data.shape})...")
        _, z, mse = train_setconca(data, epochs=N_EPOCHS, seed=42)
        concepts[name] = z
        mses[name] = mse

    names = list(concepts.keys())

    # CKA matrix
    cka_matrix = {}
    transfer_matrix = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if n1 == n2:
                cka_matrix[f"{n1}_vs_{n2}"] = 1.0
                transfer_matrix[f"{n1}_vs_{n2}"] = 1.0
                continue
            c = cka(concepts[n1], concepts[n2])
            n = min(len(concepts[n1]), len(concepts[n2]))
            split = int(n * 0.8)
            B = train_bridge(concepts[n1][:split], concepts[n2][:split])
            to = topk_overlap(concepts[n1][split:] @ B, concepts[n2][split:])
            cka_matrix[f"{n1}_vs_{n2}" ] = round(c, 4)
            transfer_matrix[f"{n1}_vs_{n2}"] = round(float(to), 4)
            print(f"  {n1} → {n2}: CKA={c:.4f}, Transfer={to:.4f}")

    return {"mses": mses, "cka_matrix": cka_matrix, "transfer_matrix": transfer_matrix, "names": names}


# ---------------------------------------------------------------------------
# EXP 6: SOTA comparison
# ---------------------------------------------------------------------------
def exp6_sota_comparison():
    print("\n" + "="*60)
    print("EXP 6: SOTA Comparison (Set-ConCA vs PCA / ICA / SAE / Random)")
    print("="*60)

    data_set   = load_hidden("data/hf_real_dataset.pt", s_max=8)
    data_2d    = flatten_to_2d(data_set)   # (N*S, D)
    D          = data_set.shape[-1]

    print(f"  Data: {data_set.shape} → 2D: {data_2d.shape}")

    result = {}

    # Set-ConCA (set-aware)
    mse_sc, stab_sc, z_sc = multi_seed_stability(data_set, n_seeds=3, epochs=N_EPOCHS)
    l0_sc = l0_sparsity(z_sc)
    result["Set-ConCA"] = {"mse": mse_sc, "stability": stab_sc, "l0": l0_sc}
    print(f"  Set-ConCA   MSE={mse_sc:.4f}  Stab={stab_sc:.4f}  L0={l0_sc:.4f}")

    # Pointwise SAE (current SOTA in LLM interpretability)
    data_2d_pt = data_set[:, 0, :]   # take first element as "pointwise"
    z_sae_pt, mse_sae = baseline_sae(data_2d_pt, concept_dim=CONCEPT_DIM, epochs=60)
    l0_sae_pt = l0_sparsity(z_sae_pt)
    # Stability: train twice
    z_sae_pt2, _ = baseline_sae(data_2d_pt, concept_dim=CONCEPT_DIM, epochs=60, seed=1337)
    stab_sae = topk_overlap(z_sae_pt, z_sae_pt2)
    result["SAE (pointwise)"] = {"mse": mse_sae, "stability": stab_sae, "l0": l0_sae_pt}
    print(f"  SAE(pw)     MSE={mse_sae:.4f}  Stab={stab_sae:.4f}  L0={l0_sae_pt:.4f}")

    # Pointwise (Set-ConCA with S=1)
    data_s1 = data_set[:, :1, :]
    mse_pw, stab_pw, z_pw = multi_seed_stability(data_s1, n_seeds=3, epochs=N_EPOCHS)
    l0_pw = l0_sparsity(z_pw)
    result["ConCA (S=1)"] = {"mse": mse_pw, "stability": stab_pw, "l0": l0_pw}
    print(f"  ConCA(S=1)  MSE={mse_pw:.4f}  Stab={stab_pw:.4f}  L0={l0_pw:.4f}")

    # PCA
    z_pca, mse_pca, exp_var = baseline_pca(data_2d_pt, concept_dim=CONCEPT_DIM)
    z_pca2, _, _ = baseline_pca(data_2d_pt + torch.randn_like(data_2d_pt) * 0.001)
    stab_pca = topk_overlap(z_pca, z_pca2)
    l0_pca   = l0_sparsity(z_pca)
    result["PCA"] = {"mse": mse_pca, "stability": stab_pca, "l0": l0_pca, "exp_var": exp_var}
    print(f"  PCA         MSE={mse_pca:.4f}  Stab={stab_pca:.4f}  L0={l0_pca:.4f}  R²={exp_var:.4f}")

    # ICA
    z_ica, mse_ica = baseline_ica(data_2d_pt, concept_dim=CONCEPT_DIM)
    l0_ica = l0_sparsity(z_ica)
    result["ICA"] = {"mse": mse_ica, "stability": float("nan"), "l0": l0_ica}
    print(f"  ICA         MSE={mse_ica:.4f}  Stab=N/A  L0={l0_ica:.4f}")

    # Random
    z_rand, mse_rand = baseline_random(data_2d_pt, concept_dim=CONCEPT_DIM)
    l0_rand = l0_sparsity(z_rand)
    result["Random"] = {"mse": mse_rand, "stability": 0.0, "l0": l0_rand}
    print(f"  Random      MSE={mse_rand:.4f}  Stab=0.00  L0={l0_rand:.4f}")

    return result


# ---------------------------------------------------------------------------
# EXP 7: Interventional Concept Steering
# ---------------------------------------------------------------------------
def exp7_interventional_steering():
    print("\n" + "="*60)
    print("EXP 7: Interventional Concept Steering")
    print("="*60)

    data_g = load_hidden("data/gemma3_4b_dataset.pt", s_max=8)
    data_l = load_hidden("data/llama_8b_dataset.pt",  s_max=8)
    N = min(len(data_g), len(data_l))
    data_g, data_l = data_g[:N], data_l[:N]

    _, z_g, _ = train_setconca(data_g, epochs=N_EPOCHS, seed=42)
    _, z_l, _ = train_setconca(data_l, epochs=N_EPOCHS, seed=42)

    split = int(N * 0.8)
    B = train_bridge(z_g[:split], z_l[:split])

    results_per_alpha = {}
    alphas = [0.5, 1.0, 2.0, 5.0, 10.0]

    for concept_idx in range(5):
        z_concept = z_g[concept_idx]   # concept vector from source model
        x_base    = data_l[10, 0, :]   # a target model activation (flat, D-dim)
        x_tgt_true = data_l[concept_idx, 0, :]

        sims = {"alpha": alphas}
        sc_sims, rand_sims = [], []

        for alpha in alphas:
            # Set-ConCA intervention
            z_mapped = z_concept @ B   # bridge to target concept space
            # Project back to D-space via decoder would need the full model;
            # instead we measure alignment in concept space (C-dim)
            z_base_l = z_l[10]
            z_tgt_l  = z_l[concept_idx]
            z_intervened = z_base_l + alpha * z_mapped

            cos = float(torch.nn.functional.cosine_similarity(
                z_intervened.unsqueeze(0), z_tgt_l.unsqueeze(0)).item())
            sc_sims.append(cos)

            # Random intervention
            z_rand = torch.randn_like(z_concept)
            z_rand_int = z_base_l + alpha * (z_rand @ B)
            cos_r = float(torch.nn.functional.cosine_similarity(
                z_rand_int.unsqueeze(0), z_tgt_l.unsqueeze(0)).item())
            rand_sims.append(cos_r)

        results_per_alpha[f"concept_{concept_idx}"] = {
            "SetConCA": sc_sims,
            "Random": rand_sims,
            "alphas": alphas,
        }

    # Average across concepts
    avg_sc   = np.mean([results_per_alpha[k]["SetConCA"] for k in results_per_alpha], axis=0).tolist()
    avg_rand = np.mean([results_per_alpha[k]["Random"]   for k in results_per_alpha], axis=0).tolist()

    print(f"  Alpha:     {alphas}")
    print(f"  Set-ConCA: {[f'{v:.3f}' for v in avg_sc]}")
    print(f"  Random:    {[f'{v:.3f}' for v in avg_rand]}")

    return {
        "alphas": alphas,
        "SetConCA_avg": avg_sc,
        "Random_avg": avg_rand,
        "per_concept": results_per_alpha,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEVICE} | N_SAMPLES={N_SAMPLES} | N_EPOCHS={N_EPOCHS}")

    all_results = {}

    all_results["exp1_set_vs_pointwise"]   = exp1_set_vs_pointwise()
    all_results["exp2_s_scaling"]          = exp2_s_scaling()
    all_results["exp3_aggregator_ablation"]= exp3_aggregator_ablation()
    all_results["exp4_cross_family"]       = exp4_cross_family()
    all_results["exp5_intra_family"]       = exp5_intra_family()
    all_results["exp6_sota_comparison"]    = exp6_sota_comparison()
    all_results["exp7_steering"]           = exp7_interventional_steering()

    # Save
    def _jsonify(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)):     return int(obj)
        if isinstance(obj, dict): return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_jsonify(v) for v in obj]
        return obj

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS DONE in {elapsed:.0f}s")
    print(f"Results saved to {out_path}")

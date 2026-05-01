"""
run_evaluation_resume.py
Resumes from saved partial results, runs EXP4-7, and merges.
"""
import sys, os, json, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Force UTF-8 output on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

from setconca.model.setconca   import SetConCA, compute_loss
from setconca.model.aggregator import AttentionAggregator

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES   = 512
N_EPOCHS    = 80
LR          = 2e-4
BATCH_SIZE  = 64
CONCEPT_DIM = 128
K_TOPK      = 32
SEEDS       = [42, 1337, 2024]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_hidden(path, n=N_SAMPLES, s_max=None):
    raw = torch.load(path, weights_only=False)
    h = raw["hidden"] if isinstance(raw, dict) else raw
    h = h[:n]
    if s_max is not None:
        h = h[:, :s_max, :]
    return h.float()


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
        x_all = data.to(DEVICE)
        f_hat, z, _ = model(x_all)
        mse = ((f_hat - x_all)**2).mean().item()
    return model, z.cpu(), mse


def topk_overlap(z1, z2, k=K_TOPK):
    if isinstance(z1, torch.Tensor): z1 = z1.detach().cpu().numpy()
    if isinstance(z2, torch.Tensor): z2 = z2.detach().cpu().numpy()
    n = min(len(z1), len(z2))
    overlaps = []
    for i in range(n):
        s1 = set(np.argsort(np.abs(z1[i]))[-k:])
        s2 = set(np.argsort(np.abs(z2[i]))[-k:])
        overlaps.append(len(s1 & s2) / k)
    return float(np.mean(overlaps))


def cka(X, Y):
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor): Y = Y.detach().cpu().numpy()
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ (X @ X.T) @ H
    L = H @ (Y @ Y.T) @ H
    return float((K * L).sum() / (np.linalg.norm(K) * np.linalg.norm(L) + 1e-8))


def l0_sparsity(z, thresh=0.01):
    if isinstance(z, torch.Tensor): z = z.detach().cpu().numpy()
    return float((np.abs(z) > thresh).mean(axis=-1).mean())


def multi_seed_stability(data, n_seeds=2, **kw):
    zs, mses = [], []
    for s in SEEDS[:n_seeds]:
        _, z, mse = train_setconca(data, seed=s, **kw)
        zs.append(z); mses.append(mse)
    pairs = [(i, j) for i in range(len(zs)) for j in range(i+1, len(zs))]
    stab = float(np.mean([topk_overlap(zs[a], zs[b]) for a, b in pairs]))
    return float(np.mean(mses)), stab, zs[0]


def train_bridge(z_src, z_tgt, epochs=300, lr=1e-2):
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


def baseline_pca(data_2d, concept_dim=CONCEPT_DIM):
    sc  = StandardScaler()
    X   = sc.fit_transform(data_2d.numpy())
    pca = PCA(n_components=concept_dim, random_state=42)
    Z   = pca.fit_transform(X)
    Xr  = pca.inverse_transform(Z)
    mse = float(np.mean((X - Xr)**2))
    exp_var = float(pca.explained_variance_ratio_.sum())
    return torch.tensor(Z, dtype=torch.float32), mse, exp_var


def baseline_ica(data_2d, concept_dim=CONCEPT_DIM):
    sc  = StandardScaler()
    X   = sc.fit_transform(data_2d.numpy())
    ica = FastICA(n_components=concept_dim, random_state=42, max_iter=500, tol=0.01)
    try:
        Z   = ica.fit_transform(X)
        Xr  = ica.inverse_transform(Z)
        mse = float(np.mean((X - Xr)**2))
    except Exception:
        Z   = np.zeros((X.shape[0], concept_dim))
        mse = float("nan")
    return torch.tensor(Z, dtype=torch.float32), mse


def baseline_sae(data_2d, concept_dim=CONCEPT_DIM, epochs=40, lr=5e-4, seed=42):
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
    torch.manual_seed(seed)
    D = data_2d.shape[-1]
    W = torch.randn(D, concept_dim) / (D**0.5)
    Z = data_2d @ W
    W_pinv = torch.linalg.pinv(W)
    Xr = Z @ W_pinv
    mse = ((Xr - data_2d)**2).mean().item()
    return Z, mse


# ============================================================
def exp4_cross_family():
    print("\n" + "=" * 60)
    print("EXP 4: Cross-Family Alignment (Gemma-3 4B <-> LLaMA-3 8B)")
    print("=" * 60)

    data_g = load_hidden("data/gemma3_4b_dataset.pt", s_max=8)
    data_l = load_hidden("data/llama_8b_dataset.pt",  s_max=8)
    print(f"  Gemma-3 4B: {data_g.shape} | LLaMA-3 8B: {data_l.shape}")

    _, z_g, mse_g = train_setconca(data_g, epochs=N_EPOCHS, seed=42)
    _, z_l, mse_l = train_setconca(data_l, epochs=N_EPOCHS, seed=42)

    cka_before = cka(z_g, z_l)
    n = min(len(z_g), len(z_l))
    split = int(n * 0.8)

    B = train_bridge(z_g[:split], z_l[:split])
    z_g_mapped = z_g[split:] @ B
    transfer_overlap = topk_overlap(z_g_mapped, z_l[split:])
    cka_after = cka(z_g_mapped.numpy(), z_l[split:].numpy())

    # Pointwise baseline
    data_g_pt = data_g[:, :1, :]
    data_l_pt = data_l[:, :1, :]
    _, z_gp, _ = train_setconca(data_g_pt, epochs=N_EPOCHS, seed=42)
    _, z_lp, _ = train_setconca(data_l_pt, epochs=N_EPOCHS, seed=42)
    Bp = train_bridge(z_gp[:split], z_lp[:split])
    pt_overlap = topk_overlap(z_gp[split:] @ Bp, z_lp[split:])

    # Random
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


def exp5_intra_family():
    print("\n" + "=" * 60)
    print("EXP 5: Intra-Family Alignment (Gemma-3 1B <-> 4B <-> 9B)")
    print("=" * 60)

    models_data = {
        "Gemma-3-1B":  load_hidden("data/gemma3_1b_dataset.pt", s_max=8),
        "Gemma-3-4B":  load_hidden("data/gemma3_4b_dataset.pt", s_max=8),
        "Gemma-2-9B":  load_hidden("data/gemma_9b_dataset.pt",  s_max=8),
    }

    concepts, mses = {}, {}
    for name, data in models_data.items():
        print(f"  Training on {name} ({data.shape})...")
        _, z, mse = train_setconca(data, epochs=N_EPOCHS, seed=42)
        concepts[name] = z
        mses[name] = mse

    names = list(concepts.keys())
    cka_matrix, transfer_matrix = {}, {}

    for n1 in names:
        for n2 in names:
            key = f"{n1}_vs_{n2}"
            if n1 == n2:
                cka_matrix[key] = 1.0
                transfer_matrix[key] = 1.0
                continue
            c = cka(concepts[n1], concepts[n2])
            n = min(len(concepts[n1]), len(concepts[n2]))
            split = int(n * 0.8)
            B = train_bridge(concepts[n1][:split], concepts[n2][:split])
            to = topk_overlap(concepts[n1][split:] @ B, concepts[n2][split:])
            cka_matrix[key]      = round(c, 4)
            transfer_matrix[key] = round(float(to), 4)
            print(f"  {n1} --> {n2}: CKA={c:.4f}, Transfer={to:.4f}")

    return {"mses": mses, "cka_matrix": cka_matrix, "transfer_matrix": transfer_matrix, "names": names}


def exp6_sota_comparison():
    print("\n" + "=" * 60)
    print("EXP 6: SOTA Comparison (Set-ConCA vs PCA / ICA / SAE / Random)")
    print("=" * 60)

    data_set   = load_hidden("data/hf_real_dataset.pt", s_max=8)
    data_2d_pt = data_set[:, 0, :]
    print(f"  Data: {data_set.shape}  pointwise: {data_2d_pt.shape}")

    result = {}

    mse_sc, stab_sc, z_sc = multi_seed_stability(data_set, n_seeds=3, epochs=N_EPOCHS)
    l0_sc = l0_sparsity(z_sc)
    result["Set-ConCA"] = {"mse": mse_sc, "stability": stab_sc, "l0": l0_sc}
    print(f"  Set-ConCA   MSE={mse_sc:.4f}  Stab={stab_sc:.4f}  L0={l0_sc:.4f}")

    data_s1 = data_set[:, :1, :]
    mse_pw, stab_pw, z_pw = multi_seed_stability(data_s1, n_seeds=3, epochs=N_EPOCHS)
    l0_pw = l0_sparsity(z_pw)
    result["ConCA (S=1)"] = {"mse": mse_pw, "stability": stab_pw, "l0": l0_pw}
    print(f"  ConCA(S=1)  MSE={mse_pw:.4f}  Stab={stab_pw:.4f}  L0={l0_pw:.4f}")

    z_sae_pt, mse_sae = baseline_sae(data_2d_pt, epochs=60)
    l0_sae = l0_sparsity(z_sae_pt)
    z_sae2,  _        = baseline_sae(data_2d_pt, epochs=60, seed=1337)
    stab_sae = topk_overlap(z_sae_pt, z_sae2)
    result["SAE (pointwise)"] = {"mse": mse_sae, "stability": stab_sae, "l0": l0_sae}
    print(f"  SAE(pw)     MSE={mse_sae:.4f}  Stab={stab_sae:.4f}  L0={l0_sae:.4f}")

    z_pca, mse_pca, exp_var = baseline_pca(data_2d_pt)
    z_pca2, _, _ = baseline_pca(data_2d_pt + torch.randn_like(data_2d_pt) * 1e-3)
    stab_pca = topk_overlap(z_pca, z_pca2)
    l0_pca   = l0_sparsity(z_pca)
    result["PCA"] = {"mse": mse_pca, "stability": stab_pca, "l0": l0_pca, "exp_var": exp_var}
    print(f"  PCA         MSE={mse_pca:.4f}  Stab={stab_pca:.4f}  L0={l0_pca:.4f}  R2={exp_var:.4f}")

    z_ica, mse_ica = baseline_ica(data_2d_pt)
    l0_ica = l0_sparsity(z_ica)
    result["ICA"] = {"mse": mse_ica, "stability": None, "l0": l0_ica}
    print(f"  ICA         MSE={mse_ica:.4f}  Stab=N/A  L0={l0_ica:.4f}")

    z_rand, mse_rand = baseline_random(data_2d_pt)
    l0_rand = l0_sparsity(z_rand)
    result["Random"] = {"mse": mse_rand, "stability": 0.0, "l0": l0_rand}
    print(f"  Random      MSE={mse_rand:.4f}  Stab=0.00  L0={l0_rand:.4f}")

    return result


def exp7_steering():
    print("\n" + "=" * 60)
    print("EXP 7: Interventional Concept Steering")
    print("=" * 60)

    data_g = load_hidden("data/gemma3_4b_dataset.pt", s_max=8)
    data_l = load_hidden("data/llama_8b_dataset.pt",  s_max=8)
    N = min(len(data_g), len(data_l))
    data_g, data_l = data_g[:N], data_l[:N]

    _, z_g, _ = train_setconca(data_g, epochs=N_EPOCHS, seed=42)
    _, z_l, _ = train_setconca(data_l, epochs=N_EPOCHS, seed=42)
    split = int(N * 0.8)
    B = train_bridge(z_g[:split], z_l[:split])

    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    sc_sims_all, rand_sims_all = [], []
    for concept_idx in range(8):
        z_concept = z_g[concept_idx]
        z_base_l  = z_l[10]
        z_tgt_l   = z_l[concept_idx]

        sc_sims, rand_sims = [], []
        for alpha in alphas:
            z_mapped     = z_concept @ B
            z_intervened = z_base_l + alpha * z_mapped
            cos = float(torch.nn.functional.cosine_similarity(
                z_intervened.unsqueeze(0), z_tgt_l.unsqueeze(0)).item())
            sc_sims.append(cos)

            z_rand  = torch.randn_like(z_concept)
            z_ri    = z_base_l + alpha * (z_rand @ B)
            cos_r   = float(torch.nn.functional.cosine_similarity(
                z_ri.unsqueeze(0), z_tgt_l.unsqueeze(0)).item())
            rand_sims.append(cos_r)

        sc_sims_all.append(sc_sims)
        rand_sims_all.append(rand_sims)

    avg_sc   = np.mean(sc_sims_all,   axis=0).tolist()
    avg_rand = np.mean(rand_sims_all, axis=0).tolist()

    print(f"  Alphas:     {alphas}")
    print(f"  Set-ConCA:  {[round(v, 4) for v in avg_sc]}")
    print(f"  Random:     {[round(v, 4) for v in avg_rand]}")

    return {"alphas": alphas, "SetConCA_avg": avg_sc, "Random_avg": avg_rand}


# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEVICE}")

    # Load partial results from run 1
    partial_path = os.path.join(RESULTS_DIR, "results_partial.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        print("Loaded partial results (EXP1-3)")
    else:
        all_results = {}

    all_results["exp4_cross_family"]       = exp4_cross_family()
    all_results["exp5_intra_family"]       = exp5_intra_family()
    all_results["exp6_sota_comparison"]    = exp6_sota_comparison()
    all_results["exp7_steering"]           = exp7_steering()

    def _j(o):
        if isinstance(o, (np.float32, np.float64, np.floating)): return float(o)
        if isinstance(o, (np.int32, np.int64, np.integer)):      return int(o)
        if isinstance(o, dict): return {k: _j(v) for k, v in o.items()}
        if isinstance(o, list): return [_j(v) for v in o]
        return o

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_j(all_results), f, indent=2)

    print(f"\nDone in {time.time()-t0:.0f}s  -->  {out_path}")

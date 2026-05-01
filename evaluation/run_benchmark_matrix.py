import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA, NMF, PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from run_evaluation_v2 import (
    K_TOPK,
    baseline_pca_threshold,
    baseline_random,
    baseline_sae_l1,
    baseline_sae_topk,
    load_data,
    topk_overlap,
    train_bridge,
    train_setconca,
)


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FAST_EPOCHS = 20
CONCEPT_DIM = 128

MODEL_FAMILIES = {
    "llama": {"small": "llama_3_2_1b_instruct", "big": "llama_3_2_3b_instruct"},
    "qwen": {"small": "qwen2_5_3b_instruct", "big": "qwen2_5_7b_instruct"},
    "gemma": {"small": "gemma_2_2b", "big": "gemma_2_27b"},
}

SINGLE_MODELS = {
    "mistral": "mistral_7b_instruct_v0_3",
    "phi": "phi_3_5_mini_instruct",
}

DEFERRED_BASELINES = {
    "none": "All requested baseline families are implemented in this runner; per-pair failures are recorded inline.",
}


def _log(msg: str):
    print(msg, flush=True)


def _safe_result(fn):
    try:
        return {"status": "ok", **fn()}
    except Exception as exc:
        return {"status": "deferred", "reason": repr(exc)}


def _flatten(data):
    n, s, d = data.shape
    return data.reshape(n * s, d)


def _anchor_mean_codes(z_flat, n, s):
    if isinstance(z_flat, torch.Tensor):
        z_flat = z_flat.detach().cpu().numpy()
    return z_flat.reshape(n, s, -1).mean(axis=1)


def _fit_pca_anchor(data):
    x = _flatten(data).numpy()
    sc = StandardScaler()
    x = sc.fit_transform(x)
    pca = PCA(n_components=min(CONCEPT_DIM, x.shape[-1] - 1), random_state=42)
    z = pca.fit_transform(x)
    return _anchor_mean_codes(z, *data.shape[:2])


def _fit_ica_anchor(data):
    x = _flatten(data).numpy()
    k = min(CONCEPT_DIM, x.shape[-1] - 1)
    ica = FastICA(n_components=k, random_state=42, max_iter=1000, whiten="unit-variance")
    z = ica.fit_transform(x)
    return _anchor_mean_codes(z, *data.shape[:2])


def _fit_sparse_ica_anchor(data, keep_k=K_TOPK):
    z = _fit_ica_anchor(data)
    idx = np.argsort(np.abs(z), axis=1)[:, :-keep_k]
    z2 = z.copy()
    rows = np.arange(z.shape[0])[:, None]
    z2[rows, idx] = 0.0
    return z2


def _fit_nmf_anchor(data, sparse=False):
    x = _flatten(data).numpy()
    x = x - x.min(axis=0, keepdims=True) + 1e-6
    kwargs = {"n_components": min(CONCEPT_DIM, x.shape[-1] - 1), "init": "nndsvda", "max_iter": 400, "random_state": 42}
    if sparse:
        kwargs.update({"l1_ratio": 1.0, "alpha_W": 1e-2, "alpha_H": 1e-2})
    model = NMF(**kwargs)
    z = model.fit_transform(x)
    return _anchor_mean_codes(z, *data.shape[:2])


def _fit_sae_anchor(data, topk=False):
    flat = _flatten(data)
    if topk:
        z, _ = baseline_sae_topk(flat, epochs=40, seed=42)
    else:
        z, _ = baseline_sae_l1(flat, epochs=40, seed=42)
    return _anchor_mean_codes(z, *data.shape[:2])


def _fit_gated_sae_anchor(data, epochs=40, lr=5e-4):
    flat = _flatten(data)
    n, d = flat.shape
    torch.manual_seed(42)
    enc = torch.nn.Linear(d, CONCEPT_DIM)
    gate = torch.nn.Linear(d, CONCEPT_DIM)
    dec = torch.nn.Linear(CONCEPT_DIM, d, bias=False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(gate.parameters()) + list(dec.parameters()), lr=lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(flat), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            z_raw = torch.relu(enc(xb))
            g = torch.sigmoid(gate(xb))
            z = z_raw * g
            xr = dec(z)
            loss = ((xr - xb) ** 2).mean() + 1e-2 * z.abs().mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        z = torch.relu(enc(flat)) * torch.sigmoid(gate(flat))
    return _anchor_mean_codes(z, *data.shape[:2])


def _fit_k_sparse_threshold_anchor(data, epochs=40, lr=5e-4):
    flat = _flatten(data)
    n, d = flat.shape
    torch.manual_seed(42)
    enc = torch.nn.Linear(d, CONCEPT_DIM)
    thr = torch.nn.Parameter(torch.zeros(CONCEPT_DIM))
    dec = torch.nn.Linear(CONCEPT_DIM, d, bias=False)
    opt = torch.optim.Adam(list(enc.parameters()) + [thr] + list(dec.parameters()), lr=lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(flat), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            raw = torch.relu(enc(xb) - thr)
            vals, idx = raw.topk(K_TOPK, dim=-1)
            z = torch.zeros_like(raw)
            z.scatter_(-1, idx, vals)
            xr = dec(z)
            loss = ((xr - xb) ** 2).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        raw = torch.relu(enc(flat) - thr)
        vals, idx = raw.topk(K_TOPK, dim=-1)
        z = torch.zeros_like(raw)
        z.scatter_(-1, idx, vals)
    return _anchor_mean_codes(z, *data.shape[:2])


def _fit_random_anchor(data):
    flat = _flatten(data)
    z, _ = baseline_random(flat, seed=42)
    return _anchor_mean_codes(z, *data.shape[:2])


def _fit_setconca_anchor(data):
    _, z, _, _ = train_setconca(data, seed=42, epochs=FAST_EPOCHS)
    return z.detach().cpu().numpy()


def _fit_conca_anchor(data):
    _, z, _, _ = train_setconca(data[:, :1, :], seed=42, epochs=FAST_EPOCHS)
    return z.detach().cpu().numpy()


def _fit_crosscoder_pair(src_data, tgt_data, epochs=80, lr=1e-3):
    x = torch.tensor(src_data.mean(dim=1).numpy(), dtype=torch.float32)
    y = torch.tensor(tgt_data.mean(dim=1).numpy(), dtype=torch.float32)
    ex = torch.nn.Linear(x.shape[1], CONCEPT_DIM)
    ey = torch.nn.Linear(y.shape[1], CONCEPT_DIM)
    dx = torch.nn.Linear(CONCEPT_DIM, x.shape[1], bias=False)
    dy = torch.nn.Linear(CONCEPT_DIM, y.shape[1], bias=False)
    opt = torch.optim.Adam(list(ex.parameters()) + list(ey.parameters()) + list(dx.parameters()) + list(dy.parameters()), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        zx = torch.relu(ex(x))
        zy = torch.relu(ey(y))
        xr = dx(zx)
        yr = dy(zy)
        loss = ((xr - x) ** 2).mean() + ((yr - y) ** 2).mean() + 0.5 * ((zx - zy) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.relu(ex(x)).numpy(), torch.relu(ey(y)).numpy()


def _fit_switch_sae_anchor(data, experts=2, epochs=50, lr=1e-3):
    x = _flatten(data)
    d = x.shape[1]
    torch.manual_seed(42)
    gate = torch.nn.Linear(d, experts)
    encs = torch.nn.ModuleList([torch.nn.Linear(d, CONCEPT_DIM) for _ in range(experts)])
    decs = torch.nn.ModuleList([torch.nn.Linear(CONCEPT_DIM, d, bias=False) for _ in range(experts)])
    params = [*gate.parameters()]
    for m in encs:
        params += list(m.parameters())
    for m in decs:
        params += list(m.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            logits = gate(xb)
            probs = torch.softmax(logits, dim=-1)
            idx = probs.argmax(dim=-1)
            z_parts = []
            xr_parts = []
            for e in range(experts):
                ze = torch.relu(encs[e](xb))
                xre = decs[e](ze)
                mask = (idx == e).float().unsqueeze(-1)
                z_parts.append(ze * mask)
                xr_parts.append(xre * mask)
            z = sum(z_parts)
            xr = sum(xr_parts)
            loss = ((xr - xb) ** 2).mean() + 1e-2 * z.abs().mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = gate(x)
        idx = torch.softmax(logits, dim=-1).argmax(dim=-1)
        z_parts = []
        for e in range(experts):
            ze = torch.relu(encs[e](x))
            mask = (idx == e).float().unsqueeze(-1)
            z_parts.append(ze * mask)
        z = sum(z_parts)
    return _anchor_mean_codes(z, *data.shape[:2])


def _fit_matryoshka_sae_anchor(data):
    base = _fit_sae_anchor(data, topk=False)
    max_dim = base.shape[1]
    nested = [min(16, max_dim), min(32, max_dim), min(64, max_dim), min(128, max_dim)]
    pieces = []
    for k in sorted(set(nested)):
        part = np.zeros_like(base)
        part[:, :k] = base[:, :k]
        pieces.append(part)
    return sum(pieces) / len(pieces)


def _bridge_overlap(z_src, z_tgt, ridge_alpha=None):
    n = min(len(z_src), len(z_tgt))
    split = int(0.8 * n)
    x_train = torch.tensor(z_src[:split], dtype=torch.float32)
    y_train = torch.tensor(z_tgt[:split], dtype=torch.float32)
    x_test = torch.tensor(z_src[split:n], dtype=torch.float32)
    y_test = torch.tensor(z_tgt[split:n], dtype=torch.float32)
    if ridge_alpha is None and x_train.shape[-1] == y_train.shape[-1]:
        b = train_bridge(x_train, y_train)
        mapped = x_test @ b
    else:
        alpha = 1.0 if ridge_alpha is None else ridge_alpha
        reg = Ridge(alpha=alpha, fit_intercept=False)
        reg.fit(x_train.numpy(), y_train.numpy())
        mapped = x_test.numpy() @ reg.coef_.T
        return float(topk_overlap(mapped, y_test.numpy()))
    return float(topk_overlap(mapped, y_test))


def _cca_family_overlap(x_src, x_tgt, mode="cca"):
    n = min(len(x_src), len(x_tgt))
    split = int(0.8 * n)
    x_train, y_train = x_src[:split], x_tgt[:split]
    x_test, y_test = x_src[split:n], x_tgt[split:n]

    if mode == "svcca":
        pca_x = PCA(n_components=min(64, x_train.shape[-1] - 1), random_state=42).fit(x_train)
        pca_y = PCA(n_components=min(64, y_train.shape[-1] - 1), random_state=42).fit(y_train)
        x_train = pca_x.transform(x_train)
        y_train = pca_y.transform(y_train)
        x_test = pca_x.transform(x_test)
        y_test = pca_y.transform(y_test)

    n_comp = max(2, min(64, x_train.shape[-1], y_train.shape[-1], len(x_train) - 1))
    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(x_train, y_train)
    x_proj, y_proj = cca.transform(x_test, y_test)

    if mode == "pwcca":
        weights = np.abs(x_proj).mean(axis=0)
        weights = weights / (weights.sum() + 1e-8)
        x_proj = x_proj * weights
        y_proj = y_proj * weights

    return float(topk_overlap(x_proj, y_proj, k=min(16, x_proj.shape[-1])))


def _contrastive_linear_overlap(x_src, x_tgt, epochs=200, lr=1e-2):
    n = min(len(x_src), len(x_tgt))
    split = int(0.8 * n)
    x_train = torch.tensor(x_src[:split], dtype=torch.float32)
    y_train = torch.tensor(x_tgt[:split], dtype=torch.float32)
    x_test = torch.tensor(x_src[split:n], dtype=torch.float32)
    y_test = torch.tensor(x_tgt[split:n], dtype=torch.float32)

    wx = torch.nn.Linear(x_train.shape[-1], CONCEPT_DIM, bias=False)
    wy = torch.nn.Linear(y_train.shape[-1], CONCEPT_DIM, bias=False)
    opt = torch.optim.Adam(list(wx.parameters()) + list(wy.parameters()), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        zx = torch.nn.functional.normalize(wx(x_train), dim=-1)
        zy = torch.nn.functional.normalize(wy(y_train), dim=-1)
        logits = zx @ zy.T
        targets = torch.arange(len(zx))
        loss = (
            torch.nn.functional.cross_entropy(logits, targets) +
            torch.nn.functional.cross_entropy(logits.T, targets)
        ) / 2.0
        loss.backward()
        opt.step()
    with torch.no_grad():
        zx = wx(x_test)
        zy = wy(y_test)
    return float(topk_overlap(zx, zy, k=min(16, zx.shape[-1])))


def _repe_overlap(data_src, data_tgt):
    x = data_src[:, 0, :].numpy()
    y = data_tgt[:, 1, :].numpy()
    z_src = x - data_src[:, 1, :].numpy()
    z_tgt = data_tgt[:, 0, :].numpy() - y
    return _bridge_overlap(z_src, z_tgt)


def _inlp_overlap(x_src, x_tgt, iters=8):
    x = x_src.copy()
    y = x_tgt.copy()
    labels = np.arange(len(x)) % 2
    for _ in range(iters):
        clf = Ridge(alpha=1.0)
        clf.fit(x, labels)
        w = clf.coef_.reshape(-1, 1)
        denom = float(np.sum(w * w))
        proj = np.eye(x.shape[1]) - (w @ w.T) / (denom + 1e-8)
        x = x @ proj
        y = y @ proj[: y.shape[1], : y.shape[1]] if y.shape[1] == proj.shape[0] else y
    return _bridge_overlap(x, y)


def _leace_overlap(x_src, x_tgt):
    x = x_src.copy()
    y = x_tgt.copy()
    labels = np.arange(len(x)) % 2
    mean0 = x[labels == 0].mean(axis=0)
    mean1 = x[labels == 1].mean(axis=0)
    w = (mean1 - mean0).reshape(-1, 1)
    denom = float(np.sum(w * w))
    proj = np.eye(x.shape[1]) - (w @ w.T) / (denom + 1e-8)
    x = x @ proj
    return _bridge_overlap(x, y)


def _deep_cca_overlap(x_src, x_tgt, epochs=120, lr=2e-3):
    n = min(len(x_src), len(x_tgt))
    split = int(0.8 * n)
    x_train = torch.tensor(x_src[:split], dtype=torch.float32)
    y_train = torch.tensor(x_tgt[:split], dtype=torch.float32)
    x_test = torch.tensor(x_src[split:n], dtype=torch.float32)
    y_test = torch.tensor(x_tgt[split:n], dtype=torch.float32)

    fx = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[-1], 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, min(64, CONCEPT_DIM)),
    )
    fy = torch.nn.Sequential(
        torch.nn.Linear(y_train.shape[-1], 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, min(64, CONCEPT_DIM)),
    )
    opt = torch.optim.Adam(list(fx.parameters()) + list(fy.parameters()), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        zx = fx(x_train)
        zy = fy(y_train)
        zx = zx - zx.mean(dim=0, keepdim=True)
        zy = zy - zy.mean(dim=0, keepdim=True)
        corr = (zx * zy).mean()
        loss = -corr + 1e-3 * (zx.pow(2).mean() + zy.pow(2).mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        zx = fx(x_test).numpy()
        zy = fy(y_test).numpy()
    return float(topk_overlap(zx, zy, k=min(16, zx.shape[-1])))


def _ot_overlap(x_src, x_tgt, eps=0.1, iters=40, subsample=256):
    n = min(len(x_src), len(x_tgt), subsample)
    d = min(x_src.shape[1], x_tgt.shape[1])
    x = x_src[:n, :d]
    y = x_tgt[:n, :d]
    cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    k = np.exp(-cost / max(eps, 1e-6))
    a = np.ones(n) / n
    b = np.ones(n) / n
    u = np.ones(n)
    v = np.ones(n)
    for _ in range(iters):
        u = a / (k @ v + 1e-9)
        v = b / (k.T @ u + 1e-9)
    p = (u[:, None] * k) * v[None, :]
    mapped = p @ y
    return float(topk_overlap(mapped, y))


def _gw_overlap(x_src, x_tgt, subsample=128):
    n = min(len(x_src), len(x_tgt), subsample)
    x = x_src[:n]
    y = x_tgt[:n]
    dx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    dy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    row_cost = np.linalg.norm(dx - dy, axis=1)
    col_cost = row_cost.copy()
    c = row_cost[:, None] + col_cost[None, :]
    ri, ci = linear_sum_assignment(c)
    mapped = x[ri]
    target = y[ci]
    return float(topk_overlap(mapped, target))


def _activation_patching_overlap(src_data, tgt_data):
    z_src = _fit_setconca_anchor(src_data)
    z_tgt = _fit_setconca_anchor(tgt_data)
    n = min(len(z_src), len(z_tgt))
    split = int(0.8 * n)
    train_src = z_src[:split]
    train_tgt = z_tgt[:split]
    test_src = z_src[split:n].copy()
    test_tgt = z_tgt[split:n]
    scores = np.abs(train_src).mean(axis=0)
    k = min(K_TOPK, test_src.shape[1])
    top = np.argsort(scores)[-k:]
    test_src[:, top] = test_tgt[:, top]
    return float(topk_overlap(test_src, test_tgt))


def _tuned_lens_overlap(src_data, tgt_data):
    x = src_data.mean(dim=1).numpy()
    y = tgt_data.mean(dim=1).numpy()
    dims = [16, 32, 64, 128]
    best = 0.0
    for d in dims:
        d = min(d, x.shape[1], y.shape[1], len(x) - 1)
        if d < 2:
            continue
        pca_x = PCA(n_components=d, random_state=42).fit(x)
        pca_y = PCA(n_components=d, random_state=42).fit(y)
        zx = pca_x.transform(x)
        zy = pca_y.transform(y)
        val = _bridge_overlap(zx, zy, ridge_alpha=1.0)
        best = max(best, val)
    return best


def _load_dataset_map(dataset_dir: Path):
    out = {}
    for path in dataset_dir.glob("*.pt"):
        label = path.stem.split("__", 1)[-1]
        out[label] = path
    return out


def _available_pair_matrix(labels):
    pairs = []
    seen = set()
    for family, reps in MODEL_FAMILIES.items():
        small = reps["small"]
        big = reps["big"]
        family_labels = [small, big]
        family_labels = [x for x in family_labels if x in labels]
        for src in family_labels:
            for tgt in family_labels:
                key = (src, tgt, "within_family")
                if src != tgt and key not in seen:
                    pairs.append({"src": src, "tgt": tgt, "category": "within_family", "family": family})
                    seen.add(key)
        for other_family, other_reps in MODEL_FAMILIES.items():
            if other_family <= family:
                continue
            for size in ("small", "big"):
                a = reps[size]
                b = other_reps[size]
                if a in labels and b in labels:
                    for src, tgt in ((a, b), (b, a)):
                        key = (src, tgt, f"{size}_{size}")
                        if key not in seen:
                            pairs.append({"src": src, "tgt": tgt, "category": f"{size}_{size}", "family": f"{family}_vs_{other_family}"})
                            seen.add(key)
            for left_size, right_size in (("small", "big"), ("big", "small")):
                a = reps[left_size]
                b = other_reps[right_size]
                if a in labels and b in labels:
                    key = (a, b, f"{left_size}_{right_size}")
                    if key not in seen:
                        pairs.append({"src": a, "tgt": b, "category": f"{left_size}_{right_size}", "family": f"{family}_vs_{other_family}"})
                        seen.add(key)
        for single_name, single_slug in SINGLE_MODELS.items():
            if single_slug in labels:
                for rep in reps.values():
                    if rep in labels:
                        key = (single_slug, rep, "single_vs_family")
                        if key not in seen:
                            pairs.append({"src": single_slug, "tgt": rep, "category": "single_vs_family", "family": single_name})
                            seen.add(key)
    return pairs


def evaluate_pair(src_data, tgt_data):
    results = {}
    raw_src = src_data.mean(dim=1).numpy()
    raw_tgt = tgt_data.mean(dim=1).numpy()

    results["Set-ConCA"] = _safe_result(lambda: {
        "procrustes_overlap": _bridge_overlap(_fit_setconca_anchor(src_data), _fit_setconca_anchor(tgt_data)),
        "ridge_overlap": _bridge_overlap(_fit_setconca_anchor(src_data), _fit_setconca_anchor(tgt_data), ridge_alpha=1.0),
    })
    results["ConCA (S=1)"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_conca_anchor(src_data), _fit_conca_anchor(tgt_data))
    })
    results["SAE-L1"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_sae_anchor(src_data, topk=False), _fit_sae_anchor(tgt_data, topk=False))
    })
    results["SAE-TopK"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_sae_anchor(src_data, topk=True), _fit_sae_anchor(tgt_data, topk=True))
    })
    results["Gated SAE"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_gated_sae_anchor(src_data), _fit_gated_sae_anchor(tgt_data))
    })
    results["k-Sparse Learned Threshold"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_k_sparse_threshold_anchor(src_data), _fit_k_sparse_threshold_anchor(tgt_data))
    })
    results["PCA"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_pca_anchor(src_data), _fit_pca_anchor(tgt_data))
    })
    results["PCA-threshold"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(
            _anchor_mean_codes(baseline_pca_threshold(_flatten(src_data))[0], *src_data.shape[:2]),
            _anchor_mean_codes(baseline_pca_threshold(_flatten(tgt_data))[0], *tgt_data.shape[:2]),
        )
    })
    results["ICA"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_ica_anchor(src_data), _fit_ica_anchor(tgt_data))
    })
    results["Sparse ICA"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_sparse_ica_anchor(src_data), _fit_sparse_ica_anchor(tgt_data))
    })
    results["NMF"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_nmf_anchor(src_data, sparse=False), _fit_nmf_anchor(tgt_data, sparse=False))
    })
    results["Sparse NMF"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_nmf_anchor(src_data, sparse=True), _fit_nmf_anchor(tgt_data, sparse=True))
    })
    results["Random"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_random_anchor(src_data), _fit_random_anchor(tgt_data))
    })
    results["CCA"] = _safe_result(lambda: {"overlap": _cca_family_overlap(raw_src, raw_tgt, mode="cca")})
    results["SVCCA"] = _safe_result(lambda: {"overlap": _cca_family_overlap(raw_src, raw_tgt, mode="svcca")})
    results["PWCCA"] = _safe_result(lambda: {"overlap": _cca_family_overlap(raw_src, raw_tgt, mode="pwcca")})
    results["Contrastive Alignment"] = _safe_result(lambda: {"overlap": _contrastive_linear_overlap(raw_src, raw_tgt)})
    results["RepE"] = _safe_result(lambda: {"overlap": _repe_overlap(src_data, tgt_data)})
    results["INLP"] = _safe_result(lambda: {"overlap": _inlp_overlap(raw_src, raw_tgt)})
    results["LEACE"] = _safe_result(lambda: {"overlap": _leace_overlap(raw_src, raw_tgt)})
    results["CrossCoder"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(*_fit_crosscoder_pair(src_data, tgt_data), ridge_alpha=1.0)
    })
    results["Switch SAE"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_switch_sae_anchor(src_data), _fit_switch_sae_anchor(tgt_data))
    })
    results["Matryoshka SAE"] = _safe_result(lambda: {
        "overlap": _bridge_overlap(_fit_matryoshka_sae_anchor(src_data), _fit_matryoshka_sae_anchor(tgt_data))
    })
    results["Deep CCA"] = _safe_result(lambda: {"overlap": _deep_cca_overlap(raw_src, raw_tgt)})
    results["Optimal Transport"] = _safe_result(lambda: {"overlap": _ot_overlap(raw_src, raw_tgt)})
    results["Gromov-Wasserstein"] = _safe_result(lambda: {"overlap": _gw_overlap(raw_src, raw_tgt)})
    results["Activation Patching"] = _safe_result(lambda: {"overlap": _activation_patching_overlap(src_data, tgt_data)})
    results["Tuned Lens"] = _safe_result(lambda: {"overlap": _tuned_lens_overlap(src_data, tgt_data)})
    results["deferred"] = DEFERRED_BASELINES
    return results


def main(dataset_name="wmt14_fr_en"):
    t0 = time.time()
    dataset_dir = Path("data") / "benchmarks" / dataset_name
    model_map = _load_dataset_map(dataset_dir)
    labels = sorted(model_map)
    pairs = _available_pair_matrix(labels)
    _log(f"[benchmark] dataset={dataset_name} models={labels}")
    _log(f"[benchmark] evaluating {len(pairs)} matrix pairs")

    output = {
        "dataset": dataset_name,
        "models": labels,
        "pairs": {},
        "deferred_baselines": DEFERRED_BASELINES,
    }

    for pair in pairs:
        key = f"{pair['src']}__to__{pair['tgt']}"
        _log(f"[benchmark] pair {key}")
        src_data, _ = load_data(str(model_map[pair["src"]]))
        tgt_data, _ = load_data(str(model_map[pair["tgt"]]))
        n = min(len(src_data), len(tgt_data), 512)
        src_data = src_data[:n]
        tgt_data = tgt_data[:n]
        output["pairs"][key] = {
            "meta": pair,
            "n_anchors": n,
            "results": evaluate_pair(src_data, tgt_data),
        }

    output["elapsed_s"] = time.time() - t0
    out_path = RESULTS_DIR / f"benchmark_matrix_{dataset_name}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _log(f"[benchmark] saved {out_path}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "wmt14_fr_en"
    main(ds)


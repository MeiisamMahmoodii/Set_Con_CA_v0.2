"""
run_extended_alignment.py
=========================
Extended analyses for:
  - additional SOTA-like baselines (NMF/ICA/Ridge/CCA)
  - layerwise alignment matrix search
  - early/mid/late steering comparison
  - unequal-depth (60% position) mapping checks
  - transfer asymmetry diagnostics
  - cross-language (EN/FR) checks when paired data exists

Usage:
  uv run python evaluation/run_extended_alignment.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA, NMF
from sklearn.linear_model import Ridge

from run_evaluation_v2 import (
    K_TOPK,
    N_SEEDS,
    SEEDS,
    ci95,
    load_data,
    topk_overlap,
    train_bridge,
    train_setconca,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FAST_EPOCHS = 20
FAST_SEEDS = SEEDS[:3]
HEAVY_BASELINE_N = 512

# Force line-buffered stdout so progress appears immediately in terminal.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _log(msg):
    print(msg, flush=True)


def _dbg(run_id, hypothesis_id, location, message, data):
    # region agent log
    try:
        payload = {
            "sessionId": "527570",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with Path("debug-527570.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # endregion


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _flatten_first_paraphrase(h):
    # (N, S, D) -> (N, D), use shared first element as pointwise proxy
    return _to_numpy(h[:, 0, :])


def ridge_bridge(z_src, z_tgt, alpha=1.0):
    n = min(len(z_src), len(z_tgt))
    x = _to_numpy(z_src[:n])
    y = _to_numpy(z_tgt[:n])
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(x, y)
    return reg.coef_.T


def cca_overlap(x_src, x_tgt, n_comp=64):
    n = min(len(x_src), len(x_tgt))
    x = _to_numpy(x_src[:n])
    y = _to_numpy(x_tgt[:n])
    k = max(2, min(n_comp, x.shape[-1], y.shape[-1], n - 1))
    cca = CCA(n_components=k, max_iter=1000)
    x_c, y_c = cca.fit_transform(x, y)
    return float(topk_overlap(x_c, y_c, k=min(16, k)))


def nmf_codes(x, n_comp=128):
    # NMF requires non-negative inputs.
    x_np = _to_numpy(x)
    x_pos = x_np - x_np.min(axis=0, keepdims=True) + 1e-6
    k = min(n_comp, x_pos.shape[-1] - 1)
    model = NMF(n_components=k, init="nndsvda", max_iter=500, random_state=42)
    z = model.fit_transform(x_pos)
    return z


def ica_codes(x, n_comp=128):
    x_np = _to_numpy(x)
    k = min(n_comp, x_np.shape[-1] - 1)
    ica = FastICA(n_components=k, random_state=42, max_iter=1000, whiten="unit-variance")
    return ica.fit_transform(x_np)


def _make_pseudo_layers(h):
    """
    Fallback when true per-layer activations are unavailable.
    Build early/mid/late pseudo-layer views by splitting D.
    """
    n, s, d = h.shape
    cuts = [0, d // 3, 2 * d // 3, d]
    out = {}
    names = ["early", "mid", "late"]
    for i, name in enumerate(names):
        a, b = cuts[i], cuts[i + 1]
        out[name] = h[:, :, a:b].clone()
    return out


def _relative_depth_index(names, frac):
    idx = int(round((len(names) - 1) * frac))
    return max(0, min(len(names) - 1, idx))


def exp_sota_extensions():
    _log("[exp_sota_extensions] Loading Gemma/LLaMA datasets...")
    data_g, _ = load_data("data/gemma3_4b_dataset.pt", s_max=8)
    data_l, _ = load_data("data/llama_8b_dataset.pt", s_max=8)
    n = min(len(data_g), len(data_l))
    data_g, data_l = data_g[:n], data_l[:n]
    split = int(0.8 * n)

    # Set-ConCA latent references
    _log("[exp_sota_extensions] Training Set-ConCA references...")
    _, z_g_set, _, _ = train_setconca(data_g, seed=42, epochs=FAST_EPOCHS)
    _, z_l_set, _, _ = train_setconca(data_l, seed=42, epochs=FAST_EPOCHS)

    # Bridge baselines
    _log("[exp_sota_extensions] Computing Procrustes/Ridge bridges...")
    b_lin = train_bridge(z_g_set[:split], z_l_set[:split])
    z_lin = z_g_set[split:] @ b_lin
    ov_lin = float(topk_overlap(z_lin, z_l_set[split:]))

    b_ridge = ridge_bridge(z_g_set[:split], z_l_set[:split], alpha=1.0)
    ov_ridge = float(topk_overlap(_to_numpy(z_g_set[split:]) @ b_ridge, _to_numpy(z_l_set[split:])))

    # Pointwise unsupervised baselines
    _log("[exp_sota_extensions] Running NMF/ICA/CCA baselines...")
    g_2d = _flatten_first_paraphrase(data_g)[:HEAVY_BASELINE_N]
    l_2d = _flatten_first_paraphrase(data_l)[:HEAVY_BASELINE_N]
    z_g_nmf = nmf_codes(g_2d, n_comp=128)
    z_l_nmf = nmf_codes(l_2d, n_comp=128)
    ov_nmf = float(topk_overlap(z_g_nmf, z_l_nmf, k=16))

    z_g_ica = ica_codes(g_2d, n_comp=128)
    z_l_ica = ica_codes(l_2d, n_comp=128)
    ov_ica = float(topk_overlap(z_g_ica, z_l_ica, k=16))

    ov_cca = cca_overlap(g_2d, l_2d, n_comp=64)

    return {
        "setconca_procrustes_overlap": ov_lin,
        "setconca_ridge_overlap": ov_ridge,
        "nmf_overlap": ov_nmf,
        "ica_overlap": ov_ica,
        "cca_overlap": ov_cca,
        "note": "NMF/ICA/CCA baselines are pointwise latent-space references, not set-trained methods.",
    }


def exp_layerwise_alignment_and_steering():
    _log("[exp_layerwise] Loading datasets and building pseudo-layers...")
    data_g, _ = load_data("data/gemma3_4b_dataset.pt", s_max=8)
    data_l, _ = load_data("data/llama_8b_dataset.pt", s_max=8)
    n = min(len(data_g), len(data_l))
    data_g, data_l = data_g[:n], data_l[:n]
    split = int(0.8 * n)

    # Fallback to pseudo-layer views if true per-layer activations are not present.
    layers_g = _make_pseudo_layers(data_g)
    layers_l = _make_pseudo_layers(data_l)
    names_g = list(layers_g.keys())
    names_l = list(layers_l.keys())

    matrix = {}
    best_pair = None
    best_score = -1.0
    for ng in names_g:
        _log(f"[exp_layerwise] Evaluating source layer bucket: {ng}")
        for nl in names_l:
            _, z_g, _, _ = train_setconca(layers_g[ng], seed=42, epochs=FAST_EPOCHS)
            _, z_l, _, _ = train_setconca(layers_l[nl], seed=42, epochs=FAST_EPOCHS)
            b = train_bridge(z_g[:split], z_l[:split])
            score = float(topk_overlap(z_g[split:] @ b, z_l[split:]))
            matrix[f"{ng}_to_{nl}"] = score
            if score > best_score:
                best_score = score
                best_pair = (ng, nl)

    # Early/mid/late steering comparison (latent-direction similarity gain proxy).
    _log("[exp_layerwise] Running steering alpha sweeps...")
    alphas = [0.0, 1.0, 2.0, 5.0]
    steering = {}
    for ng in names_g:
        _, z_g, _, _ = train_setconca(layers_g[ng], seed=42, epochs=FAST_EPOCHS)
        _, z_l, _, _ = train_setconca(layers_l[ng], seed=42, epochs=FAST_EPOCHS)
        b = train_bridge(z_g[:split], z_l[:split])
        z_base = z_l[10]
        z_tgt = z_l[20]
        sims = []
        for a in alphas:
            z_int = z_base + a * (z_g[20] @ b)
            sim = float(torch.nn.functional.cosine_similarity(z_int.unsqueeze(0), z_tgt.unsqueeze(0)).item())
            sims.append(sim)
        steering[ng] = {"alphas": alphas, "sims": sims, "gain_a5": sims[-1] - sims[0]}

    # Unequal layer-count mapping via relative depth 60%.
    idx_g_60 = _relative_depth_index(names_g, 0.6)
    idx_l_60 = _relative_depth_index(names_l, 0.6)
    name_g_60 = names_g[idx_g_60]
    name_l_60 = names_l[idx_l_60]
    _, z_g60, _, _ = train_setconca(layers_g[name_g_60], seed=42, epochs=FAST_EPOCHS)
    _, z_l60, _, _ = train_setconca(layers_l[name_l_60], seed=42, epochs=FAST_EPOCHS)
    b60 = train_bridge(z_g60[:split], z_l60[:split])
    score60 = float(topk_overlap(z_g60[split:] @ b60, z_l60[split:]))

    return {
        "layer_pair_overlap_matrix": matrix,
        "best_layer_pair": {"source": best_pair[0], "target": best_pair[1], "overlap": best_score},
        "steering_early_mid_late": steering,
        "relative_depth_60pct": {
            "source_layer": name_g_60,
            "target_layer": name_l_60,
            "overlap": score60,
        },
        "note": "Layerwise results use pseudo-layer views from hidden-dimension partition because only one extracted model layer is available in current data artifacts.",
    }


def exp_asymmetry_diagnostics():
    _log("[exp_asymmetry] Loading small/mid/big datasets...")
    data_small, _ = load_data("data/gemma3_1b_dataset.pt", s_max=8)
    data_mid, _ = load_data("data/gemma3_4b_dataset.pt", s_max=8)
    data_big, _ = load_data("data/llama_8b_dataset.pt", s_max=8)

    n = min(len(data_small), len(data_mid), len(data_big))
    data_small, data_mid, data_big = data_small[:n], data_mid[:n], data_big[:n]
    split = int(0.8 * n)

    pairs = [
        ("small_to_mid", data_small, data_mid),
        ("mid_to_small", data_mid, data_small),
        ("mid_to_big", data_mid, data_big),
        ("big_to_mid", data_big, data_mid),
    ]
    out = {}
    for name, src, tgt in pairs:
        _log(f"[exp_asymmetry] Running pair: {name}")
        vals = []
        for s in FAST_SEEDS:
            _log(f"[exp_asymmetry]   seed={s}")
            _, z_src, _, _ = train_setconca(src, seed=s, epochs=FAST_EPOCHS)
            _, z_tgt, _, _ = train_setconca(tgt, seed=s, epochs=FAST_EPOCHS)
            b = train_bridge(z_src[:split], z_tgt[:split])
            vals.append(float(topk_overlap(z_src[split:] @ b, z_tgt[split:])))
        out[name] = {"mean": float(np.mean(vals)), "ci95": ci95(vals)}

    out["interpretation_hint"] = (
        "Asymmetry can reflect receiver capacity limits, architecture/training-recipe differences, "
        "and information bottleneck mismatch."
    )
    return out


def exp_cross_language_if_available():
    _log("[exp_cross_language] Checking EN/FR paired datasets...")
    # Expected optional artifacts for EN/FR paired sets.
    candidates = [
        ("data/enfr_gemma3_4b_dataset.pt", "data/enfr_llama_8b_dataset.pt"),
        ("data/gemma3_4b_enfr_dataset.pt", "data/llama_8b_enfr_dataset.pt"),
    ]
    pair = None
    # region agent log
    _dbg(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="evaluation/run_extended_alignment.py:exp_cross_language_if_available:candidates",
        message="Checking legacy EN/FR candidate file pairs under data/.",
        data={"candidates": candidates},
    )
    # endregion
    for a, b in candidates:
        # region agent log
        _dbg(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="evaluation/run_extended_alignment.py:exp_cross_language_if_available:existence_check",
            message="Candidate pair existence check result.",
            data={"a": a, "a_exists": Path(a).exists(), "b": b, "b_exists": Path(b).exists()},
        )
        # endregion
        if Path(a).exists() and Path(b).exists():
            pair = (a, b)
            break

    pair_source = "legacy_data_root"
    if pair is None:
        bench_dir = Path("data") / "benchmarks" / "wmt14_fr_en"
        benchmark_candidates = [
            (
                bench_dir / "wmt14_fr_en__gemma_2_2b.pt",
                bench_dir / "wmt14_fr_en__llama_3_2_3b_instruct.pt",
            ),
            (
                bench_dir / "wmt14_fr_en__qwen2_5_3b_instruct.pt",
                bench_dir / "wmt14_fr_en__mistral_7b_instruct_v0_3.pt",
            ),
        ]
        for a, b in benchmark_candidates:
            if a.exists() and b.exists():
                pair = (str(a), str(b))
                pair_source = "benchmark_wmt14_fr_en"
                # region agent log
                _dbg(
                    run_id="post-fix",
                    hypothesis_id="H5",
                    location="evaluation/run_extended_alignment.py:exp_cross_language_if_available:fallback_pick",
                    message="Using benchmark EN/FR tensors as cross-language fallback pair.",
                    data={"source": str(a), "target": str(b)},
                )
                # endregion
                break

    if pair is None:
        bench_dir = Path("data") / "benchmarks" / "wmt14_fr_en"
        bench_pt_files = sorted(str(p) for p in bench_dir.glob("*.pt"))
        # region agent log
        _dbg(
            run_id="pre-fix",
            hypothesis_id="H3",
            location="evaluation/run_extended_alignment.py:exp_cross_language_if_available:benchmark_visibility",
            message="Cross-language skip happened; checking benchmark tensor availability.",
            data={
                "cwd": os.getcwd(),
                "bench_dir": str(bench_dir),
                "bench_dir_exists": bench_dir.exists(),
                "bench_pt_files_count": len(bench_pt_files),
                "bench_pt_files_sample": bench_pt_files[:6],
            },
        )
        # endregion
        _log("[exp_cross_language] EN/FR datasets not found; skipping.")
        return {
            "status": "skipped",
            "reason": "No EN/FR paired activation datasets found in data/.",
            "expected_paths": candidates,
        }

    path_g, path_l = pair
    data_g, _ = load_data(path_g, s_max=8)
    data_l, _ = load_data(path_l, s_max=8)
    n = min(len(data_g), len(data_l))
    data_g, data_l = data_g[:n], data_l[:n]
    split = int(0.8 * n)

    _, z_g, _, _ = train_setconca(data_g, seed=42, epochs=FAST_EPOCHS)
    _, z_l, _, _ = train_setconca(data_l, seed=42, epochs=FAST_EPOCHS)
    b = train_bridge(z_g[:split], z_l[:split])
    ov = float(topk_overlap(z_g[split:] @ b, z_l[split:]))
    # region agent log
    _dbg(
        run_id="post-fix",
        hypothesis_id="H6",
        location="evaluation/run_extended_alignment.py:exp_cross_language_if_available:overlap",
        message="Cross-language EN/FR overlap computed.",
        data={"pair_source": pair_source, "source_path": path_g, "target_path": path_l, "overlap": ov, "n": n},
    )
    # endregion
    return {
        "status": "ok",
        "pair_source": pair_source,
        "dataset_paths": {"source": path_g, "target": path_l},
        "en_fr_overlap": ov,
    }


def _format_terminal_report(results, elapsed_s):
    sota = results["sota_extensions"]
    layer = results["layerwise_alignment_and_steering"]
    asym = results["asymmetry_diagnostics"]
    xlang = results["cross_language_en_fr"]

    lines = [
        "",
        "=" * 72,
        "SetConCA Extended Alignment Terminal Report",
        "=" * 72,
        f"Elapsed: {elapsed_s:.1f}s",
        f"Run profile: FAST_EPOCHS={FAST_EPOCHS}, FAST_SEEDS={list(FAST_SEEDS)}",
        f"Heavy baselines sample size: N={HEAVY_BASELINE_N}",
        "",
        "[SOTA Extensions]",
        f"- SetConCA Procrustes overlap: {sota['setconca_procrustes_overlap']:.4f}",
        f"- SetConCA Ridge overlap:      {sota['setconca_ridge_overlap']:.4f}",
        f"- NMF overlap (pointwise):     {sota['nmf_overlap']:.4f}",
        f"- ICA overlap (pointwise):     {sota['ica_overlap']:.4f}",
        f"- CCA overlap (pointwise):     {sota['cca_overlap']:.4f}",
        "",
        "[Layerwise Alignment]",
        f"- Best layer pair: {layer['best_layer_pair']['source']} -> "
        f"{layer['best_layer_pair']['target']} "
        f"(overlap={layer['best_layer_pair']['overlap']:.4f})",
        f"- Relative depth 60%: {layer['relative_depth_60pct']['source_layer']} -> "
        f"{layer['relative_depth_60pct']['target_layer']} "
        f"(overlap={layer['relative_depth_60pct']['overlap']:.4f})",
        "",
        "[Steering Gains by Layer Bucket]",
    ]
    for name, obj in layer["steering_early_mid_late"].items():
        lines.append(f"- {name}: gain@alpha5 = {obj['gain_a5']:+.4f}")

    lines.extend(
        [
            "",
            "[Transfer Asymmetry]",
            f"- small_to_mid: {asym['small_to_mid']['mean']:.4f} +/- {asym['small_to_mid']['ci95']:.4f}",
            f"- mid_to_small: {asym['mid_to_small']['mean']:.4f} +/- {asym['mid_to_small']['ci95']:.4f}",
            f"- mid_to_big:   {asym['mid_to_big']['mean']:.4f} +/- {asym['mid_to_big']['ci95']:.4f}",
            f"- big_to_mid:   {asym['big_to_mid']['mean']:.4f} +/- {asym['big_to_mid']['ci95']:.4f}",
            "",
            "[Cross-Language EN/FR]",
        ]
    )
    if xlang["status"] == "ok":
        lines.append(f"- EN/FR overlap: {xlang['en_fr_overlap']:.4f}")
        lines.append(
            f"- Datasets: {xlang['dataset_paths']['source']} | {xlang['dataset_paths']['target']}"
        )
    else:
        lines.append(f"- Status: {xlang['status']} ({xlang['reason']})")

    lines.extend(
        [
            "",
            f"JSON output: {RESULTS_DIR / 'extended_alignment_results.json'}",
            f"Text report: {RESULTS_DIR / 'extended_alignment_terminal_report.txt'}",
            "=" * 72,
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.time()
    _log("Running SOTA extension block...")
    sota_extensions = exp_sota_extensions()
    _log("Running layerwise alignment + steering block...")
    layerwise = exp_layerwise_alignment_and_steering()
    _log("Running asymmetry diagnostics block...")
    asymmetry = exp_asymmetry_diagnostics()
    _log("Running cross-language EN/FR block...")
    xlang = exp_cross_language_if_available()

    results = {
        "sota_extensions": sota_extensions,
        "layerwise_alignment_and_steering": layerwise,
        "asymmetry_diagnostics": asymmetry,
        "cross_language_en_fr": xlang,
    }

    out_path = RESULTS_DIR / "extended_alignment_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    terminal_report = _format_terminal_report(results, elapsed)
    report_path = RESULTS_DIR / "extended_alignment_terminal_report.txt"
    report_path.write_text(terminal_report, encoding="utf-8")

    print(terminal_report)

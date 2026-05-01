#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "final_bundle" / "figures"


def method_means(matrix: dict) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for pair in matrix["pairs"].values():
        for method, payload in pair["results"].items():
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                continue
            vals = [v for k, v in payload.items() if k != "status" and isinstance(v, (int, float))]
            if vals:
                out.setdefault(method, []).append(sum(vals) / len(vals))
    return {k: float(sum(v) / len(v)) for k, v in out.items()}


def barh_chart(data: dict[str, float], title: str, path: Path, top_n: int = 10) -> None:
    s = pd.Series(data).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=s.values, y=s.index, orient="h", palette="viridis")
    plt.title(title)
    plt.xlabel("Mean overlap")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    wmt = json.loads((RESULTS / "benchmark_matrix_wmt14_fr_en.json").read_text(encoding="utf-8"))
    opus = json.loads((RESULTS / "benchmark_matrix_opus100_multi_en.json").read_text(encoding="utf-8"))
    rv2 = json.loads((RESULTS / "results_v2.json").read_text(encoding="utf-8"))

    wmt_means = method_means(wmt)
    opus_means = method_means(opus)

    barh_chart(wmt_means, "WMT14 Top Methods (Smaller View)", OUT / "small_wmt14_top_methods.png", top_n=10)
    barh_chart(opus_means, "OPUS100 Top Methods (Smaller View)", OUT / "small_opus100_top_methods.png", top_n=10)

    key = ["Set-ConCA", "ConCA (S=1)", "SAE-TopK", "CrossCoder", "Contrastive Alignment", "PCA", "CCA", "SVCCA", "PWCCA"]
    rows = []
    for m in key:
        if m in wmt_means:
            rows.append({"Method": m, "Dataset": "WMT14", "Mean": wmt_means[m]})
        if m in opus_means:
            rows.append({"Method": m, "Dataset": "OPUS100", "Mean": opus_means[m]})
    df = pd.DataFrame(rows)
    if not df.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="Method", y="Mean", hue="Dataset")
        plt.xticks(rotation=35, ha="right")
        plt.title("Key Methods: WMT14 vs OPUS100")
        plt.tight_layout()
        plt.savefig(OUT / "small_key_methods_dual_dataset.png", dpi=180)
        plt.close()

    exp6 = rv2["exp6_sota_comparison"]
    sparse_methods = {}
    for m in ["Set-ConCA", "ConCA (S=1)", "SAE (L1, pointwise)", "SAE (TopK, pointwise)"]:
        if m in exp6 and isinstance(exp6[m], dict) and "mse" in exp6[m]:
            sparse_methods[m] = exp6[m]["mse"]
    if sparse_methods:
        s = pd.Series(sparse_methods).sort_values()
        plt.figure(figsize=(7, 4))
        sns.barplot(x=s.index, y=s.values, palette="mako")
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("MSE (lower is better)")
        plt.title("Sparse Method MSE Comparison (Smaller View)")
        plt.tight_layout()
        plt.savefig(OUT / "small_sparse_mse_comparison.png", dpi=180)
        plt.close()

    exp = rv2["exp16_topk_pointwise_vs_set"]
    plt.figure(figsize=(6, 4))
    x = ["Pointwise SAE-TopK", "Set-ConCA"]
    y = [exp["pointwise"]["mean"], exp["set"]["mean"]]
    e = [exp["pointwise"]["ci95"], exp["set"]["ci95"]]
    plt.bar(x, y, yerr=e, color=["#7f8c8d", "#2980b9"])
    plt.ylim(0, 1.0)
    plt.ylabel("Transfer overlap")
    plt.title("EXP16 Focus Chart (Smaller View)")
    plt.tight_layout()
    plt.savefig(OUT / "small_exp16_focus.png", dpi=180)
    plt.close()

    print(f"Wrote small charts to {OUT}")


if __name__ == "__main__":
    main()


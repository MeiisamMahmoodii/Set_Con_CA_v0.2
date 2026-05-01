from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_MD = RESULTS / "REPORT.md"
EXEC_SUMMARY_MD = RESULTS / "EXECUTIVE_SUMMARY.md"


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _method_mean_from_matrix(matrix: dict, method: str) -> float:
    """
    Matches the averaging logic used by `scripts/build_report.py`.
    For each pair: if payload is dict and payload['status']=='ok', compute mean of all numeric
    fields (excluding 'status'), then average those per-pair means across pairs.
    """
    out_vals: list[float] = []
    for _, pair in matrix.get("pairs", {}).items():
        payload = pair.get("results", {}).get(method)
        if not isinstance(payload, dict):
            continue
        if payload.get("status") != "ok":
            continue
        numeric = [
            float(v)
            for k, v in payload.items()
            if k != "status" and isinstance(v, (int, float))
        ]
        if numeric:
            out_vals.append(sum(numeric) / len(numeric))
    if not out_vals:
        raise KeyError(f"method={method} not found with status ok in this matrix")
    return float(sum(out_vals) / len(out_vals))


def _parse_exec_summary_value(pattern: str, text: str) -> float:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Could not find pattern: {pattern}")
    return float(m.group(1))


def _parse_exec_summary_two_values(pattern: str, text: str) -> tuple[float, float]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Could not find pattern: {pattern}")
    return float(m.group(1)), float(m.group(2))


@dataclass
class Check:
    name: str
    computed: float
    reported: float
    tolerance: float = 1e-3

    def ok(self) -> bool:
        return abs(self.computed - self.reported) <= self.tolerance


def main() -> int:
    results_v2 = _load_json(RESULTS / "results_v2.json")
    extended = _load_json(RESULTS / "extended_alignment_results.json")
    wmt = _load_json(RESULTS / "benchmark_matrix_wmt14_fr_en.json")
    opus_path = RESULTS / "benchmark_matrix_opus100_multi_en.json"
    have_opus = opus_path.exists()
    opus = _load_json(opus_path) if have_opus else {"pairs": {}}
    if not have_opus:
        print("[warn] benchmark_matrix_opus100_multi_en.json missing; skipping OPUS verification checks.")

    exec_text = EXEC_SUMMARY_MD.read_text(encoding="utf-8", errors="replace")

    exp4 = results_v2["exp4_cross_family"]["SetConCA"]
    exp7 = results_v2["exp7_steering"]
    exp12 = results_v2["exp12_nonlinear_bridge"]["summary"]
    exp16 = results_v2["exp16_topk_pointwise_vs_set"]
    exp6 = results_v2["exp6_sota_comparison"]
    exp9 = results_v2["exp9_consistency_ablation"]

    # Multilingual means (average across directed pairs, using method mean logic).
    set_wmt = _method_mean_from_matrix(wmt, "Set-ConCA")
    set_opus = _method_mean_from_matrix(opus, "Set-ConCA") if have_opus else float("nan")
    conca1_wmt = _method_mean_from_matrix(wmt, "ConCA (S=1)")
    conca1_opus = _method_mean_from_matrix(opus, "ConCA (S=1)") if have_opus else float("nan")

    # Extended alignment diagnostics.
    sota = extended["sota_extensions"]
    layer = extended["layerwise_alignment_and_steering"]

    checks: list[Check] = []

    # Exec summary numbers are in percents for most transfer/overlap claims.
    # exp4: reported "69.5% +/- 0.6pp" => use transfer_g_to_l and CI
    exp4_mean = float(exp4["transfer_g_to_l"])
    exp4_ci = float(exp4["transfer_g_to_l_ci95"])
    chance = float(results_v2["exp4_cross_family"]["chance_level"])

    # Cross-family transfer line: **69.5% +/- 0.6pp** vs **25% chance**
    rep_cross = _parse_exec_summary_value(r"reaches \*\*([0-9.]+)% \+/- [0-9.]+pp\*\*", exec_text)
    # Another parse for CI and chance by separate patterns.
    rep_cross_ci = _parse_exec_summary_value(r"reaches \*\*[0-9.]+% \+/- ([0-9.]+)pp\*\*", exec_text)
    rep_chance = _parse_exec_summary_value(r"vs \*\*([0-9.]+)% chance\*\*", exec_text)
    # Report values are rounded (typically to 0.1). Allow a small tolerance for that rounding.
    checks.append(Check("exp4_transfer_mean_pct", exp4_mean * 100, rep_cross, tolerance=0.06))
    checks.append(Check("exp4_transfer_ci_pctpp", exp4_ci * 100, rep_cross_ci, tolerance=0.06))
    checks.append(Check("exp4_chance_pct", chance * 100, rep_chance, tolerance=0.5e-1))

    # Causal steering: baseline and gains
    # Line contains "+9.8pp" and "+10.7pp"
    rep_gain_4b = _parse_exec_summary_value(r"Set-ConCA gains \*\*\+([0-9.]+)pp\*\*", exec_text)
    rep_gain_w2s = _parse_exec_summary_value(r"weak-to-strong gains \*\*\+([0-9.]+)pp\*\*", exec_text)
    checks.append(Check("exp7_gain_4b_pp", float(exp7["gain_at_alpha10_4B"]) * 100, rep_gain_4b, tolerance=0.06))
    checks.append(Check("exp7_gain_w2s_pp", float(exp7["gain_at_alpha10_w2s"]) * 100, rep_gain_w2s, tolerance=0.06))

    # Linear bridge: 69.3% vs 64.2%
    rep_lin = _parse_exec_summary_value(r"linear bridge reaches \*\*([0-9.]+)%\*\*", exec_text)
    rep_mlp = _parse_exec_summary_value(r"nonlinear MLP falls to \*\*([0-9.]+)%\*\*", exec_text)
    checks.append(Check("exp12_linear_pct", float(exp12["linear_mean"]) * 100, rep_lin, tolerance=0.06))
    checks.append(Check("exp12_mlp_pct", float(exp12["mlp_mean"]) * 100, rep_mlp, tolerance=0.06))

    # Sparse reconstruction tradeoff on MSE (not percents)
    rep_mse_set, rep_mse_sae = _parse_exec_summary_two_values(
        r"Sparse reconstruction trade-off:.*?\(\*\*([0-9.]+)\*\* vs \*\*([0-9.]+)\*\*\)",
        exec_text,
    )
    checks.append(Check("exp6_mse_set", float(exp6["Set-ConCA"]["mse"]), rep_mse_set, tolerance=5e-4))
    checks.append(
        Check(
            "exp6_mse_sae_topk",
            float(exp6["SAE (TopK, pointwise)"]["mse"]),
            rep_mse_sae,
            tolerance=5e-4,
        )
    )

    # Pointwise raw transfer: "78.4% vs 69.5%"
    rep_raw_pt, rep_raw_set = _parse_exec_summary_two_values(
        r"Pointwise raw transfer:.*?overlap \(\*\*([0-9.]+)%\*\* vs \*\*([0-9.]+)%\*\*\)",
        exec_text,
    )
    checks.append(Check("exp16_pointwise_pct", float(exp16["pointwise"]["mean"]) * 100, rep_raw_pt, tolerance=0.06))
    checks.append(Check("exp16_set_pct", float(exp16["set"]["mean"]) * 100, rep_raw_set, tolerance=0.06))

    # Multilingual means (direct overlap, not percents)
    rep_set_wmt = _parse_exec_summary_value(r"Set-ConCA averages \*\*([0-9.]+)\*\* \(WMT14\)", exec_text)
    checks.append(Check("multilingual_set_wmt", set_wmt, rep_set_wmt))
    rep_conca1_wmt, rep_conca1_opus = _parse_exec_summary_two_values(
        r"ConCA\(S=1\) is \*\*([0-9.]+)\s*/\s*([0-9.]+)\*\*",
        exec_text,
    )
    checks.append(Check("multilingual_conca1_wmt", conca1_wmt, rep_conca1_wmt))
    if have_opus:
        rep_set_opus = _parse_exec_summary_value(r"and \*\*([0-9.]+)\*\* \(OPUS100\)", exec_text)
        checks.append(Check("multilingual_set_opus", set_opus, rep_set_opus, tolerance=1e-4))
        checks.append(Check("multilingual_conca1_opus", conca1_opus, rep_conca1_opus))

    # Extended diagnostics: use a few anchor numbers.
    rep_procrustes = _parse_exec_summary_value(r"Procrustes \*\*([0-9.]+)\*\*", exec_text)
    rep_ridge = _parse_exec_summary_value(r"Ridge \*\*([0-9.]+)\*\*", exec_text)
    rep_cca = _parse_exec_summary_value(r"CCA \*\*([0-9.]+)\*\*", exec_text)
    checks.append(Check("extended_procrustes", float(sota["setconca_procrustes_overlap"]), rep_procrustes))
    checks.append(Check("extended_ridge", float(sota["setconca_ridge_overlap"]), rep_ridge))
    checks.append(Check("extended_cca", float(sota["cca_overlap"]), rep_cca))

    # Run checks
    failed = [c for c in checks if not c.ok()]
    if failed:
        print("Report verification FAILED. Mismatches:")
        for c in failed:
            print(
                f"- {c.name}: computed={c.computed} reported={c.reported} "
                f"(abs diff {abs(c.computed - c.reported):.6g}, tol {c.tolerance})"
            )
        return 1

    print("Report verification PASSED: EXECUTIVE_SUMMARY.md matches computed artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


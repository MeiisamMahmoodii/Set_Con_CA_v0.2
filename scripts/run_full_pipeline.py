from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_MD = RESULTS_DIR / "REPORT.md"


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    _log(f"\n[run] {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=True,
    )


def ensure_required_data(required_files: list[str], *, hard_fail: bool) -> None:
    missing: list[str] = []
    for rel in required_files:
        p = PROJECT_ROOT / "data" / rel
        if not p.exists():
            missing.append(str(rel))

    if missing:
        msg = "Missing required data artifacts:\n- " + "\n- ".join(missing)
        if hard_fail:
            raise FileNotFoundError(msg)
        _log("[warn] " + msg)


def benchmark_data_ok(dataset_name: str) -> bool:
    bench_dir = PROJECT_ROOT / "data" / "benchmarks" / dataset_name
    if not bench_dir.exists():
        return False
    # Heuristic: at least one extracted hidden-state shard exists.
    any_pt = any(p.suffix == ".pt" for p in bench_dir.glob("*.pt"))
    status_json = (bench_dir / "extraction_status.json").exists()
    return any_pt and status_json


def maybe_build_multilingual_benchmarks(*, force: bool) -> None:
    # These are the exact dataset names consumed by `run_benchmark_matrix.py`.
    datasets = ["wmt14_fr_en", "opus100_multi_en"]

    need_any = force is True or any(not benchmark_data_ok(ds) for ds in datasets)
    if not need_any:
        _log("[data] Multilingual benchmark tensors already exist; skipping build.")
        return

    _log("[data] Building multilingual benchmark datasets (WMT14 + OPUS100)...")
    script = PROJECT_ROOT / "evaluation" / "build_multilingual_benchmarks.py"
    run_cmd([sys.executable, str(script)])


def run_tests(*, extra_pytest_args: list[str]) -> None:
    _log("[tests] Running pytest...")
    cmd = [sys.executable, "-m", "pytest", "-q", *extra_pytest_args]
    run_cmd(cmd)


def run_evaluation_v2(*, force: bool) -> None:
    # `run_evaluation_v2.py` always writes results/results_v2.json and figures.
    # Force just influences whether we delete previous outputs.
    if force:
        (RESULTS_DIR / "results_v2.json").unlink(missing_ok=True)
        (FIGURES_DIR / "fig01_set_vs_pointwise.png").unlink(missing_ok=True)

    _log("[eval] Running main evaluation suite (run_evaluation_v2.py)...")
    script = PROJECT_ROOT / "evaluation" / "run_evaluation_v2.py"
    run_cmd([sys.executable, str(script)])


def run_extended_alignment(*, force: bool) -> None:
    if force:
        (RESULTS_DIR / "extended_alignment_results.json").unlink(missing_ok=True)
        (RESULTS_DIR / "extended_alignment_terminal_report.txt").unlink(missing_ok=True)

    _log("[eval] Running extended alignment diagnostics (run_extended_alignment.py)...")
    script = PROJECT_ROOT / "evaluation" / "run_extended_alignment.py"
    run_cmd([sys.executable, str(script)])


def run_benchmark_matrices(dataset_names: list[str], *, force: bool) -> None:
    for ds in dataset_names:
        out_path = RESULTS_DIR / f"benchmark_matrix_{ds}.json"
        if force:
            out_path.unlink(missing_ok=True)

        # Skip if it already exists and user didn't request force.
        if out_path.exists() and not force:
            _log(f"[bench] Skipping matrix {ds} (already exists).")
            continue

        _log(f"[bench] Running benchmark matrix: {ds}...")
        script = PROJECT_ROOT / "evaluation" / "run_benchmark_matrix.py"
        run_cmd([sys.executable, str(script), ds])


def generate_figures(*, force: bool) -> None:
    # These scripts regenerate results/figures/*.png from results_v2.json.
    scripts = [
        PROJECT_ROOT / "evaluation" / "plot_results.py",
        PROJECT_ROOT / "evaluation" / "generate_additional_charts.py",
        PROJECT_ROOT / "evaluation" / "generate_report_enhancement_charts.py",
    ]

    if force:
        _log("[fig] Force-regenerating: clearing existing PNGs.")
        if FIGURES_DIR.exists():
            for p in FIGURES_DIR.glob("*.png"):
                p.unlink(missing_ok=True)

    _log("[fig] Regenerating figures...")
    for s in scripts:
        run_cmd([sys.executable, str(s)])


def maybe_build_report(*, force: bool) -> None:
    if not (RESULTS_DIR / "results_v2.json").exists():
        _log("[report] Skipping docs/report generation (results_v2.json missing).")
        return

    deep = PROJECT_ROOT / "docs" / "report" / "generated" / "deep_dive"
    if deep.exists() and not force:
        _log("[report] Generated deep-dive already exists; skipping (use --force-report to rebuild).")
        return

    _log("[report] Building docs/report (markdown + figures from JSON)...")
    script = PROJECT_ROOT / "scripts" / "build_report.py"
    run_cmd([sys.executable, str(script)])


def validate_report_assets() -> None:
    if not REPORT_MD.exists():
        _log("[report] WARNING: results/REPORT.md not found; skipping asset validation.")
        return

    text = REPORT_MD.read_text(encoding="utf-8", errors="replace")

    # Extract png references that commonly appear as:
    # - ![Alt](./figures/figXX_....png)
    # - ./figures/figXX_....png (plain)
    pngs = set(re.findall(r"(?:\./figures/|figures/)([^)\s]+?\.png)", text))
    missing = [p for p in sorted(pngs) if not (FIGURES_DIR / p).exists()]

    if missing:
        _log("[report] WARNING: REPORT.md references missing figures:")
        for m in missing:
            _log(f"  - {m}")
    else:
        _log("[report] Asset validation passed: all referenced PNGs exist.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full Set-ConCA pipeline (single command).")
    p.add_argument(
        "--datasets",
        type=str,
        default="wmt14_fr_en,opus100_multi_en",
        help="Comma-separated datasets for run_benchmark_matrix.py.",
    )
    p.add_argument("--skip-tests", action="store_true", help="Skip pytest.")
    p.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmark matrices.")
    p.add_argument("--skip-figures", action="store_true", help="Skip figure generation.")
    p.add_argument(
        "--skip-report",
        "--skip-obsidian",
        dest="skip_report",
        action="store_true",
        help="Skip docs/report markdown generation (alias: --skip-obsidian).",
    )
    p.add_argument("--force", action="store_true", help="Force rebuild (evaluation/benchmarks/figures).")
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute eval + benchmarks + figures + docs/report, but do NOT rebuild multilingual activations.",
    )
    p.add_argument(
        "--force-report",
        "--force-obsidian",
        dest="force_report_only",
        action="store_true",
        help="Force rebuild of docs/report generated tree (alias: --force-obsidian).",
    )
    p.add_argument(
        "--rebuild-activations",
        action="store_true",
        help="Rebuild multilingual activation tensors (only meaningful with --recompute).",
    )
    p.add_argument(
        "--pytest-args",
        type=str,
        default="",
        help="Extra args appended to pytest command (e.g. '-k claim').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]

    ts = _now_ts()
    log_path = RESULTS_DIR / f"pipeline_run_{ts}.log"
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    log_path.write_text(f"Pipeline run start: {ts}\n", encoding="utf-8")

    # Make python subprocesses stream output in real-time.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    _log(f"Set-ConCA full pipeline @ {ts}")
    _log(f"Project root: {PROJECT_ROOT}")
    _log(f"Datasets: {dataset_names}")

    # Per-step timing/status for final terminal report.
    steps: dict[str, dict[str, object]] = {}
    overall_t0 = time.time()

    # Compute which stages should be forced.
    # - --force keeps old behavior: it forces everything, including activation tensors.
    # - --recompute is the common "full rerun" mode: recompute results from existing activations.
    force_activations = bool(args.force or args.rebuild_activations)
    force_eval = bool(args.force or args.recompute)
    force_extended = bool(args.force or args.recompute)
    force_benchmarks = bool(args.force or args.recompute)
    force_figures = bool(args.force or args.recompute)
    force_report = bool(args.force or args.recompute or args.force_report_only)

    # Step 1: Data preparation.
    t0 = time.time()
    ensure_required_data(
        required_files=[
            "hf_real_dataset.pt",
            "hf_point_dataset.pt",
            "gemma3_1b_dataset.pt",
            "gemma3_4b_dataset.pt",
            "gemma_9b_dataset.pt",
            "llama_8b_dataset.pt",
        ],
        hard_fail=False,
    )
    maybe_build_multilingual_benchmarks(force=force_activations)
    steps["data_prep"] = {
        "status": "ok",
        "elapsed_s": round(time.time() - t0, 1),
        "note": "Validated local tensors and multilingual benchmarks.",
    }

    # Step 2: Tests.
    t0 = time.time()
    if not args.skip_tests:
        run_tests(extra_pytest_args=args.pytest_args.split() if args.pytest_args.strip() else [])
        steps["tests"] = {
            "status": "ok",
            "elapsed_s": round(time.time() - t0, 1),
            "note": "Pytest suite completed (see terminal for summary).",
        }
    else:
        _log("[tests] Skipped (--skip-tests).")
        steps["tests"] = {
            "status": "skipped",
            "elapsed_s": 0.0,
            "note": "Skipped by user flag.",
        }

    # Step 3: Training + evaluation.
    t0 = time.time()
    if force_eval or not (RESULTS_DIR / "results_v2.json").exists():
        run_evaluation_v2(force=force_eval)
        steps["evaluation_v2"] = {
            "status": "ok",
            "elapsed_s": round(time.time() - t0, 1),
            "note": "Wrote results_v2.json and core figures.",
        }
    else:
        _log("[eval] Skipping run_evaluation_v2.py (results_v2.json already exists).")
        steps["evaluation_v2"] = {
            "status": "cached",
            "elapsed_s": 0.0,
            "note": "Reused existing results_v2.json.",
        }

    # Step 4: Extended diagnostics.
    t0 = time.time()
    if force_extended or not (RESULTS_DIR / "extended_alignment_results.json").exists():
        run_extended_alignment(force=force_extended)
        steps["extended_alignment"] = {
            "status": "ok",
            "elapsed_s": round(time.time() - t0, 1),
            "note": "Wrote extended_alignment_results.json and terminal report.",
        }
    else:
        _log("[eval] Skipping run_extended_alignment.py (already exists).")
        steps["extended_alignment"] = {
            "status": "cached",
            "elapsed_s": 0.0,
            "note": "Reused existing extended_alignment_results.json.",
        }

    # Step 5: Benchmark matrices.
    t0 = time.time()
    if not args.skip_benchmarks:
        run_benchmark_matrices(dataset_names, force=force_benchmarks)
        steps["benchmarks"] = {
            "status": "ok",
            "elapsed_s": round(time.time() - t0, 1),
            "note": "Wrote benchmark_matrix_*.json for multilingual datasets.",
        }
    else:
        _log("[bench] Skipped (--skip-benchmarks).")
        steps["benchmarks"] = {
            "status": "skipped",
            "elapsed_s": 0.0,
            "note": "Skipped by user flag.",
        }

    # Step 6: Figures + docs/report generated pages.
    t0 = time.time()
    if not args.skip_figures:
        generate_figures(force=force_figures)
        fig_status = "ok"
        fig_note = "Regenerated results/figures/*.png."
    else:
        _log("[fig] Skipped (--skip-figures).")
        fig_status = "skipped"
        fig_note = "Skipped by user flag."
    fig_elapsed = round(time.time() - t0, 1)

    t_obs = time.time()
    if not args.skip_report:
        maybe_build_report(force=force_report)
        rep_status = "ok"
        rep_note = "Updated docs/report/generated/ from JSON."
    else:
        _log("[report] Skipped (--skip-report).")
        rep_status = "skipped"
        rep_note = "Skipped by user flag."
    rep_elapsed = round(time.time() - t_obs, 1)

    steps["figures"] = {
        "status": fig_status,
        "elapsed_s": fig_elapsed,
        "note": fig_note,
    }
    steps["report_md"] = {
        "status": rep_status,
        "elapsed_s": rep_elapsed,
        "note": rep_note,
    }

    # Final report asset sanity check.
    validate_report_assets()

    overall_elapsed = round(time.time() - overall_t0, 1)
    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + f"Pipeline run complete in {overall_elapsed}s.\n",
        encoding="utf-8",
    )

    # Compact terminal summary.
    _log("\n" + "=" * 72)
    _log("Set-ConCA Full Pipeline — Terminal Report")
    _log("=" * 72)
    _log(f"Total elapsed: {overall_elapsed:.1f}s")
    _log("")
    for name, info in steps.items():
        status = info.get("status", "unknown")
        elapsed = info.get("elapsed_s", 0.0)
        note = info.get("note", "")
        _log(f"- {name:18s} | {status:8s} | {elapsed:6.1f}s | {note}")
    _log("")
    _log(f"Core artifacts:")
    _log(f"- results_v2.json          -> {RESULTS_DIR / 'results_v2.json'}")
    _log(f"- extended_alignment JSON  -> {RESULTS_DIR / 'extended_alignment_results.json'}")
    _log(f"- multilingual matrices    -> {RESULTS_DIR / 'benchmark_matrix_wmt14_fr_en.json'}")
    _log(f"                             {RESULTS_DIR / 'benchmark_matrix_opus100_multi_en.json'}")
    _log(f"- main report              -> {REPORT_MD}")
    _log(f"- figures                  -> {FIGURES_DIR}")
    _log(f"- docs/report (generated)  -> {PROJECT_ROOT / 'docs' / 'report'}")
    _log("=" * 72 + "\n")


if __name__ == "__main__":
    main()


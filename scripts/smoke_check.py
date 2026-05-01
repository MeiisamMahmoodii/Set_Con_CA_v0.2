#!/usr/bin/env python3
"""Fast sanity check: required activation tensors + pytest.

Run from repo root:
  uv run python scripts/smoke_check.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

# Keep in sync with scripts/run_full_pipeline.py ensure_required_data()
REQUIRED_DATA_FILES = [
    "hf_real_dataset.pt",
    "hf_point_dataset.pt",
    "gemma3_1b_dataset.pt",
    "gemma3_4b_dataset.pt",
    "gemma_9b_dataset.pt",
    "llama_8b_dataset.pt",
]


def main() -> int:
    missing = [rel for rel in REQUIRED_DATA_FILES if not (DATA / rel).exists()]

    print("Set-ConCA smoke check")
    print("=" * 50)
    print(f"Project root: {PROJECT_ROOT}")
    print("\n[1] Activation tensors (main eval)")
    if missing:
        print("  MISSING (full pipeline will skip or warn):")
        for m in missing:
            print(f"    - data/{m}")
    else:
        print("  OK: all required tensors present under data/")

    print("\n[2] pytest -q")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        print("\nSmoke check FAILED: pytest errors.")
        return r.returncode

    if missing:
        print("\nSmoke check PARTIAL: pytest passed; restore missing data/*.pt for full GPU runs.")
        return 2

    print("\nSmoke check OK (data + tests).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

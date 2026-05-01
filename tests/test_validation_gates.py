import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.runner.eval_metrics import bootstrap_ci, cka, topk_overlap


def test_METRIC_01_topk_overlap_is_symmetric_and_bounded():
    z1 = np.array([0.1, 0.4, 0.3, 0.2, -0.1, -0.2], dtype=np.float32)
    z2 = np.array([0.2, 0.5, 0.1, 0.0, -0.3, -0.4], dtype=np.float32)
    ov12 = topk_overlap(z1, z2, k=3)
    ov21 = topk_overlap(z2, z1, k=3)
    assert 0.0 <= ov12 <= 1.0
    assert ov12 == pytest.approx(ov21, abs=1e-12)


def test_METRIC_02_topk_overlap_batch_is_bounded():
    rng = np.random.default_rng(42)
    z1 = rng.normal(size=(16, 64)).astype(np.float32)
    z2 = rng.normal(size=(16, 64)).astype(np.float32)
    ov = topk_overlap(z1, z2, k=8)
    assert 0.0 <= ov <= 1.0


def test_METRIC_02b_topk_overlap_uses_absolute_activation():
    z1 = np.array([-10.0, 2.0, 1.0, 0.5], dtype=np.float32)
    z2 = np.array([-9.0, 3.0, 0.1, 0.2], dtype=np.float32)
    # If abs-TopK is used, both pick indices {0,1} -> perfect overlap.
    assert topk_overlap(z1, z2, k=2) == pytest.approx(1.0, abs=1e-12)


def test_METRIC_02c_topk_overlap_caps_k_to_feature_dim():
    z1 = np.array([0.2, -0.4, 0.1], dtype=np.float32)
    z2 = np.array([0.1, -0.5, 0.3], dtype=np.float32)
    ov = topk_overlap(z1, z2, k=99)
    assert 0.0 <= ov <= 1.0


def test_METRIC_03_cka_is_symmetric():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(32, 16)).astype(np.float32)
    y = rng.normal(size=(32, 16)).astype(np.float32)
    cxy = cka(x, y)
    cyx = cka(y, x)
    assert np.isfinite(cxy)
    assert np.isfinite(cyx)
    assert cxy == pytest.approx(cyx, abs=1e-8)


def test_METRIC_04_bootstrap_ci_returns_valid_interval():
    values = np.array([0.2, 0.4, 0.5, 0.7, 0.9], dtype=np.float64)
    lo, hi = bootstrap_ci(values, n=200)
    assert lo <= hi
    assert lo <= float(values.mean()) <= hi


def _load_results():
    path = Path("results/results_v2.json")
    if not path.exists():
        pytest.skip("results/results_v2.json not present yet")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_CLAIM_01_exp9_framing_matches_transfer_direction():
    data = _load_results()
    exp9 = data["exp9_consistency_ablation"]
    delta = exp9["full_model"]["transfer"] - exp9["no_consistency"]["transfer"]
    framing = exp9.get("framing", "").lower()
    # If consistency does not help, framing must not claim it "drives" gains.
    if delta <= 0.0:
        assert "drives" not in framing
        assert "demonstrates that the consistency loss" not in framing


def test_CLAIM_02_exp10_near_chance_only_if_numeric_supports_it():
    data = _load_results()
    exp10 = data["exp10_corruption_test"]
    chance = data["exp4_cross_family"]["chance_level"]
    t100 = exp10["corruption_100pct"]["transfer"]
    framing = exp10.get("framing", "").lower()
    if "near chance" in framing:
        assert t100 <= chance + 0.1


def test_CLAIM_03_exp14_improvement_text_matches_values():
    data = _load_results()
    transfer_pca32 = data["exp14_pca32_transfer"]["transfer_mean"]
    transfer_full = data["exp4_cross_family"]["SetConCA"]["transfer_g_to_l"]
    framing = data["exp14_pca32_transfer"].get("framing", "").lower()
    if "improv" in framing:
        assert transfer_pca32 > transfer_full


def test_CLAIM_04_exp16_primary_driver_statement_matches_diff():
    data = _load_results()
    exp16 = data["exp16_topk_pointwise_vs_set"]
    diff = exp16["diff"]
    framing = exp16.get("framing", "").lower()
    if "primary driver" in framing:
        assert diff > 0.0

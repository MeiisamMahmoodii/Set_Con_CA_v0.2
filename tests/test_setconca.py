"""
tests/test_setconca.py – Consolidated Set-ConCA test suite
===========================================================
Categories
----------
TestEncoder          – ElementEncoder unit tests (ENC_01 – ENC_05)
TestAggregator       – SetAggregator unit tests  (AGG_01 – AGG_07)
TestDecoder          – DualDecoder unit tests    (DEC_01 – DEC_06)
TestLosses           – Sparsity + Consistency    (SPAR_01-05, CONS_01-06)
TestData             – Dataset / DataLoader      (DATA_01 – DATA_04)
TestFullModel        – End-to-end SetConCA       (FULL_01 – FULL_10)
TestTopK             – Hard Top-K mode           (TOPK_01 – TOPK_02)
TestThresholdBridge  – S-sweep sanity + bridge   (THRESHOLD_01, BRIDGE_01)
TestExperiments      – Smoke tests for all 5 evaluation experiment runners
"""

import os
import torch
import pytest
import numpy as np

from setconca.model.encoder    import ElementEncoder
from setconca.model.aggregator import SetAggregator, AttentionAggregator
from setconca.model.decoder    import DualDecoder
from setconca.model.setconca   import SetConCA, compute_loss
from setconca.losses.sparsity  import sparsity_loss
from setconca.losses.consistency import consistency_loss
from setconca.data.dataset     import RepresentationSetDataset, make_synthetic_dataset, make_dataloader

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
B, S, D, C = 4, 8, 64, 128


# ===========================================================================
# TestEncoder
# ===========================================================================
class TestEncoder:
    def test_ENC_01_output_shape(self):
        enc = ElementEncoder(D, C)
        u = enc(torch.randn(B, S, D))
        assert u.shape == (B, S, C)

    def test_ENC_02_linearity(self):
        enc = ElementEncoder(D, C)
        enc.eval()
        x1, x2 = torch.randn(1, 1, D), torch.randn(1, 1, D)
        a, b = 2.5, -1.3
        enc.linear.bias.data.zero_()
        assert torch.allclose(enc(a*x1 + b*x2), a*enc(x1) + b*enc(x2), atol=1e-5)

    def test_ENC_03_no_relu_applied(self):
        enc = ElementEncoder(D, C)
        with torch.no_grad():
            enc.linear.weight.fill_(-1.0)
            enc.linear.bias.zero_()
        u = enc(torch.ones(1, 1, D) * 0.5)
        assert (u < 0).any(), "No negative values — ReLU may have been applied"

    def test_ENC_04_unbounded_output(self):
        enc = ElementEncoder(D, C)
        u = enc(torch.randn(B, S, D) * 100)
        assert u.abs().max() > 10, "Suspiciously small range — check for clipping"

    def test_ENC_05_permutation_independence(self):
        enc = ElementEncoder(D, C)
        x, perm = torch.randn(1, S, D), torch.randperm(S)
        assert torch.allclose(enc(x)[:, perm, :], enc(x[:, perm, :]), atol=1e-5)


# ===========================================================================
# TestAggregator
# ===========================================================================
class TestAggregator:
    def test_AGG_01_output_shapes(self):
        agg = SetAggregator(C)
        z_hat, u_bar, u_out = agg(torch.randn(B, S, C))
        assert z_hat.shape == (B, C)
        assert u_bar.shape == (B, C)
        assert u_out.shape == (B, S, C)

    def test_AGG_02_permutation_invariance(self):
        agg = SetAggregator(C)
        agg.eval()
        u, perm = torch.randn(1, S, C), torch.randperm(S)
        z1, _, _ = agg(u)
        z2, _, _ = agg(u[:, perm, :])
        assert torch.allclose(z1, z2, atol=1e-5), "Not permutation invariant"

    def test_AGG_03_u_not_modified(self):
        agg = SetAggregator(C)
        u = torch.randn(B, S, C)
        _, _, u_out = agg(u)
        assert torch.allclose(u, u_out, atol=1e-6), "u was modified by aggregator"

    def test_AGG_04_layernorm_affine_free(self):
        agg = SetAggregator(C)
        assert len(list(agg.norm.parameters())) == 0, "LayerNorm has affine params"

    def test_AGG_05_z_hat_normalized(self):
        agg = SetAggregator(C)
        agg.eval()
        z_hat, _, _ = agg(torch.randn(B, S, C) * 50)
        assert z_hat.mean(dim=-1).abs().max() < 0.1
        assert (z_hat.std(dim=-1) - 1.0).abs().max() < 0.15

    def test_AGG_06_dropout_disabled_at_eval(self):
        agg = SetAggregator(C, dropout_p=0.9)
        agg.eval()
        u = torch.randn(B, S, C)
        z1, _, _ = agg(u)
        z2, _, _ = agg(u)
        assert torch.allclose(z1, z2), "Dropout active in eval mode"

    def test_AGG_07_u_bar_is_pre_norm(self):
        agg = SetAggregator(C)
        agg.eval()
        u = torch.ones(B, S, C) * 5.0
        _, u_bar, _ = agg(u)
        assert u_bar.mean().abs() > 1.0, "u_bar appears normalised (should be raw)"
        z_hat, _, _ = agg(u)
        assert z_hat.mean().abs() < 0.1, "z_hat should be zero-mean after LayerNorm"


# ===========================================================================
# TestDecoder
# ===========================================================================
class TestDecoder:
    def test_DEC_01_output_shape(self):
        dec = DualDecoder(C, D)
        assert dec(torch.randn(B, C), torch.randn(B, S, C)).shape == (B, S, D)

    def test_DEC_02_single_bias_parameter(self):
        dec = DualDecoder(C, D)
        named = dict(dec.named_parameters())
        assert "b_d" in named
        assert "shared.bias" not in named
        assert "residual.bias" not in named
        assert named["b_d"].shape == (D,)

    def test_DEC_03_shared_stream_broadcasts(self):
        dec = DualDecoder(C, D)
        dec.residual.weight.data.zero_()
        dec.b_d.data.zero_()
        out = dec(torch.randn(B, C), torch.zeros(B, S, C))
        for i in range(1, S):
            assert torch.allclose(out[:, 0, :], out[:, i, :], atol=1e-5)

    def test_DEC_04_residual_varies_per_element(self):
        dec = DualDecoder(C, D)
        dec.shared.weight.data.zero_()
        dec.b_d.data.zero_()
        out = dec(torch.zeros(B, C), torch.randn(B, S, C))
        assert not torch.allclose(out[:, 0, :], out[:, 1, :], atol=1e-3)

    def test_DEC_05_bias_shape(self):
        assert DualDecoder(C, D).b_d.shape == (D,)

    def test_DEC_06_gradient_flows_to_both_streams(self):
        dec = DualDecoder(C, D)
        z = torch.randn(B, C, requires_grad=True)
        u = torch.randn(B, S, C, requires_grad=True)
        dec(z, u).mean().backward()
        assert z.grad is not None
        assert u.grad is not None


# ===========================================================================
# TestLosses
# ===========================================================================
class TestLosses:
    # --- Sparsity ---
    def test_SPAR_01_output_range(self):
        loss = sparsity_loss(torch.randn(8, 128))
        assert 0 < loss.item() < 1

    def test_SPAR_02_near_zero_for_very_negative_input(self):
        loss = sparsity_loss(torch.full((4, 64), -50.0))
        assert loss.item() < 0.01

    def test_SPAR_03_bounded_on_large_values(self):
        loss = sparsity_loss(torch.randn(4, 64) * 1000)
        assert loss.item() < 1.1

    def test_SPAR_04_zero_input_gives_half(self):
        loss = sparsity_loss(torch.zeros(32, 128))
        assert abs(loss.item() - 0.5) < 0.01

    def test_SPAR_05_gradient_flows(self):
        z = torch.randn(4, 64, requires_grad=True)
        sparsity_loss(z).backward()
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()

    # --- Consistency ---
    def _enc_agg_fn(self):
        enc, agg = ElementEncoder(32, 64), SetAggregator(64)
        def fn(x):
            z, _, _ = agg(enc(x))
            return z
        return fn

    def test_CONS_01_output_is_scalar(self):
        fn = self._enc_agg_fn()
        assert consistency_loss(torch.randn(4, 8, 32), fn).shape == ()

    def test_CONS_02_skips_small_sets(self):
        fn = self._enc_agg_fn()
        assert consistency_loss(torch.randn(4, 3, 32), fn).item() == 0.0

    def test_CONS_03_nonnegative(self):
        fn = self._enc_agg_fn()
        x = torch.randn(4, 8, 32)
        for _ in range(5):
            assert consistency_loss(x, fn).item() >= 0

    def test_CONS_04_gradient_flows(self):
        enc, agg = ElementEncoder(32, 64), SetAggregator(64)
        fn = lambda x: agg(enc(x))[0]
        consistency_loss(torch.randn(4, 8, 32), fn).backward()
        assert enc.linear.weight.grad is not None

    def test_CONS_05_constant_set_gives_near_zero(self):
        enc, agg = ElementEncoder(32, 64), SetAggregator(64)
        enc.eval(); agg.eval()
        fn = lambda x: agg(enc(x))[0]
        v = torch.randn(1, 1, 32).expand(4, 8, 32).contiguous()
        assert consistency_loss(v, fn).item() < 1e-4

    def test_CONS_06_random_splits_each_call(self):
        fn = self._enc_agg_fn()
        x = torch.randn(4, 8, 32)
        losses = {round(consistency_loss(x, fn).item(), 6) for _ in range(10)}
        assert len(losses) > 1


# ===========================================================================
# TestData
# ===========================================================================
class TestData:
    def test_DATA_01_output_shape(self):
        ds = make_synthetic_dataset(64, 8, 32)
        batch = next(iter(make_dataloader(ds, batch_size=16)))
        assert batch.shape == (16, 8, 32)

    def test_DATA_02_no_preprocessing(self):
        raw = torch.randn(4, 3, 8) * 100
        ds = RepresentationSetDataset(raw)
        assert torch.allclose(ds[0], raw[0])

    def test_DATA_03_rejects_2d_input(self):
        with pytest.raises(AssertionError):
            RepresentationSetDataset(torch.randn(64, 32))

    def test_DATA_04_deterministic_without_shuffle(self):
        ds = make_synthetic_dataset(32, 4, 16)
        dl = make_dataloader(ds, batch_size=8, shuffle=False)
        for b1, b2 in zip(list(dl), list(dl)):
            assert torch.allclose(b1, b2)


# ===========================================================================
# TestFullModel
# ===========================================================================
class TestFullModel:
    def test_FULL_01_forward_shapes(self):
        model = SetConCA(D, C)
        f_hat, z_hat, u = model(torch.randn(B, S, D))
        assert f_hat.shape == (B, S, D)
        assert z_hat.shape == (B, C)
        assert u.shape == (B, S, C)

    def test_FULL_02_loss_components_present(self):
        model = SetConCA(D, C)
        total, parts = compute_loss(model, torch.randn(B, S, D))
        assert {"mse", "sparsity", "consistency"}.issubset(set(parts))
        core = parts["mse"] + parts["sparsity"] + parts["consistency"]
        assert abs(total.item() - core.item()) < 1e-5

    def test_FULL_03_loss_decreases(self):
        model = SetConCA(D, C)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(B, S, D)
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss, _ = compute_loss(model, x)
            loss.backward(); opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_FULL_04_no_nan(self):
        model = SetConCA(D, C)
        for _ in range(10):
            assert not torch.isnan(compute_loss(model, torch.randn(B, S, D))[0])

    def test_FULL_05_gradients_reach_all_params(self):
        model = SetConCA(D, C)
        loss, _ = compute_loss(model, torch.randn(B, S, D))
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_FULL_06_parameter_count(self):
        model = SetConCA(D, C)
        total = sum(p.numel() for p in model.parameters())
        expected = D*C + C + C*D + C*D + D  # W_e, b_e, W_shared, W_resid, b_d
        assert total == expected, f"Got {total}, expected {expected}"

    def test_FULL_07_alpha_zero_disables_sparsity(self):
        model = SetConCA(D, C)
        _, parts = compute_loss(model, torch.randn(B, S, D), alpha=0.0)
        assert parts["sparsity"].item() == 0.0

    def test_FULL_08_beta_zero_disables_consistency(self):
        model = SetConCA(D, C)
        _, parts = compute_loss(model, torch.randn(B, S, D), beta=0.0)
        assert parts["consistency"].item() == 0.0

    def test_FULL_09_sparsity_not_constant_during_training(self):
        """Regression: frozen sparsity bug — sparsity must see u_bar, not z_hat."""
        torch.manual_seed(0)
        model = SetConCA(D, C)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randn(B, S, D)
        vals = []
        for _ in range(30):
            opt.zero_grad()
            loss, parts = compute_loss(model, x, alpha=1.0, beta=0.0)
            loss.backward(); opt.step()
            vals.append(round(parts["sparsity"].item(), 6))
        assert len(set(vals)) > 5, (
            f"Sparsity is nearly constant ({len(set(vals))} unique values). "
            "Check sparsity_loss receives u_bar, not z_hat."
        )

    def test_FULL_10_sparsity_uses_u_bar(self):
        """If encoder bias = +10, u_bar is large positive → Sigmoid >> 0.5."""
        model = SetConCA(D, C)
        with torch.no_grad():
            model.encoder.linear.bias.fill_(10.0)
        _, parts = compute_loss(model, torch.randn(B, S, D), alpha=1.0)
        assert parts["sparsity"].item() > 0.8, (
            f"Sparsity={parts['sparsity'].item():.4f} — expected >0.8 with bias=+10. "
            "Likely still using z_hat instead of u_bar."
        )


# ===========================================================================
# TestTopK
# ===========================================================================
class TestTopK:
    def test_TOPK_01_exactly_k_nonzero(self):
        torch.manual_seed(0)
        k = 17
        model = SetConCA(hidden_dim=D, concept_dim=C, use_topk=True, k=k)
        _, z_hat, _ = model(torch.randn(B, S, D))
        counts = (z_hat != 0).sum(dim=-1)
        assert torch.all(counts == k), f"Expected {k} active latents per sample, got {counts.tolist()}"

    def test_TOPK_02_disables_sigmoid_sparsity(self):
        torch.manual_seed(1)
        model = SetConCA(hidden_dim=D, concept_dim=C, use_topk=True, k=16)
        _, parts = compute_loss(model, torch.randn(B, S, D), alpha=0.5, beta=0.1)
        assert parts["sparsity"].item() == 0.0


# ===========================================================================
# TestThresholdBridge
# ===========================================================================
class TestThresholdBridge:
    def _quick_train(self, s):
        torch.manual_seed(42 + s)
        d, c = 48, 96
        base  = torch.randn(160, 1, d) * 1.25 + torch.randn(160, s, d) * 0.6
        eval_ = torch.randn(64,  1, d) * 1.25 + torch.randn(64,  s, d) * 0.6
        model = SetConCA(d, c, use_topk=False)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(8):
            opt.zero_grad()
            loss, _ = compute_loss(model, base, alpha=0.1, beta=0.01)
            loss.backward(); opt.step()
        with torch.no_grad():
            _, parts = compute_loss(model, eval_, alpha=0.1, beta=0.01)
        return parts["mse"].item(), parts["consistency"].item()

    def test_THRESHOLD_01_finite_metrics_across_set_sizes(self):
        rows = [self._quick_train(s) for s in (3, 8, 16)]
        mse_vals = set()
        for mse, cons in rows:
            assert torch.isfinite(torch.tensor(mse))
            assert torch.isfinite(torch.tensor(cons))
            mse_vals.add(round(mse, 6))
        assert len(mse_vals) > 1, "MSE identical across S values — degenerate sweep"

    def test_BRIDGE_01_correlated_beats_random(self):
        """Top-K overlap of correlated pairs must beat random baseline."""
        torch.manual_seed(7)
        n, dim, k = 64, 256, 20
        x = torch.randn(n, dim)
        y = x + 0.05 * torch.randn(n, dim)

        def _topk_overlap(a, b, k):
            overlaps = []
            for i in range(len(a)):
                ia = set(a[i].topk(k).indices.tolist())
                ib = set(b[i].topk(k).indices.tolist())
                overlaps.append(len(ia & ib) / k)
            return float(np.mean(overlaps))

        corr_score = _topk_overlap(x, y, k)
        rand_score = _topk_overlap(x, y[torch.randperm(n)], k)
        assert corr_score > rand_score + 0.05


# ===========================================================================
# TestExperiments  – smoke tests for all 5 NeurIPS experiment runners
# ===========================================================================
class TestExperiments:
    """
    Smoke tests: run each experiment on tiny synthetic data.
    They verify the runner executes without errors and returns expected keys.
    """

    _data_small = torch.randn(20, 8, 64)

    def _rotated(self, data):
        q, _ = torch.linalg.qr(torch.randn(data.shape[-1], data.shape[-1]))
        return (data @ q) + torch.randn_like(data) * 0.1

    def test_EXP_01_set_vs_pointwise(self):
        from evaluation.runner.exp1_set_vs_pointwise import run_exp1
        res = run_exp1(self._data_small, seeds=[42, 1337])
        assert "SetConCA"  in res
        assert "Pointwise" in res
        for key in ("mse", "stability"):
            assert key in res["SetConCA"]
            assert key in res["Pointwise"]

    def test_EXP_02_set_size_scaling(self):
        from evaluation.runner.exp2_set_size_scaling import run_exp2
        res = run_exp2(self._data_small, sweep=[1, 3, 8], seeds=[42, 1337])
        for s in (1, 3, 8):
            assert s in res
            assert "mse" in res[s] and "stability" in res[s]

    def test_EXP_03_aggregator_ablation(self):
        from evaluation.runner.exp3_aggregator_ablation import run_exp3
        res = run_exp3(self._data_small, seeds=[42, 1337])
        for mode in ("attention", "mean", "random"):
            assert mode in res

    def test_EXP_04_cross_model_transfer(self):
        from evaluation.runner.exp4_cross_model_transfer import run_exp4
        res = run_exp4(self._data_small, self._rotated(self._data_small))
        for key in ("setconca_overlap", "pointwise_overlap", "random_overlap", "cka"):
            assert key in res

    def test_EXP_05_interventional_steering(self):
        from evaluation.runner.exp5_interventional_steering import run_exp5
        res = run_exp5(self._data_small, self._rotated(self._data_small))
        for key in ("base", "setconca", "random"):
            assert key in res

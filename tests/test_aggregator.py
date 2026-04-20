import torch, pytest
from setconca.model.aggregator import SetAggregator

B, S, C = 4, 8, 128

def test_AGG_01_output_shapes():
    agg = SetAggregator(C)
    u = torch.randn(B, S, C)
    z_hat, u_bar, u_out = agg(u)
    assert z_hat.shape == (B, C)
    assert u_bar.shape == (B, C)
    assert u_out.shape == (B, S, C)

def test_AGG_02_permutation_invariance():
    """z_hat must be identical for any permutation of set elements."""
    agg = SetAggregator(C)
    agg.eval()
    u = torch.randn(1, S, C)
    perm = torch.randperm(S)
    z1, _, _ = agg(u)
    z2, _, _ = agg(u[:, perm, :])
    assert torch.allclose(z1, z2, atol=1e-5), 'Not permutation invariant'

def test_AGG_03_norm_applied_to_pooled_not_elements():
    """u_out must equal u_in exactly — norm must not touch individual u_i."""
    agg = SetAggregator(C)
    u = torch.randn(B, S, C)
    _, _, u_out = agg(u)
    assert torch.allclose(u, u_out, atol=1e-6), 'u was modified by aggregator'

def test_AGG_04_layernorm_is_affine_free():
    """LayerNorm must have no learned scale or bias parameters."""
    agg = SetAggregator(C)
    params = list(agg.norm.parameters())
    assert len(params) == 0, f'LayerNorm has affine params: {params}'

def test_AGG_05_z_hat_is_normalized():
    """z_hat should have near-zero mean and near-unit variance per sample."""
    agg = SetAggregator(C)
    agg.eval()
    u = torch.randn(B, S, C) * 50  # large scale shouldn't matter
    z_hat, _, _ = agg(u)
    means = z_hat.mean(dim=-1).abs()
    stds = z_hat.std(dim=-1)
    assert means.max() < 0.1, f'z_hat not zero-centered: max mean = {means.max()}'
    assert (stds - 1.0).abs().max() < 0.15, f'z_hat not unit std: {stds}'

def test_AGG_06_dropout_zero_at_eval():
    """Dropout must be disabled in eval mode."""
    agg = SetAggregator(C, dropout_p=0.9)
    agg.eval()
    u = torch.randn(B, S, C)
    z1, _, _ = agg(u)
    z2, _, _ = agg(u)
    assert torch.allclose(z1, z2), 'Dropout active in eval mode'

def test_AGG_07_u_bar_is_pre_norm():
    """u_bar must NOT be zero-mean: it is the pre-LayerNorm log-posterior estimate."""
    agg = SetAggregator(C)
    agg.eval()
    # Encoder output with a strong positive bias -> u_bar should be positive-mean
    u = torch.ones(B, S, C) * 5.0   # all positive
    _, u_bar, _ = agg(u)
    # u_bar should be ~5.0 (mean of ones * 5), not zero-centred
    assert u_bar.mean().abs() > 1.0, 'u_bar appears to be normalised (should be raw)'
    # z_hat (LayerNorm output) should still be zero-mean
    z_hat, _, _ = agg(u)
    assert z_hat.mean().abs() < 0.1, 'z_hat should be zero-mean after LayerNorm'

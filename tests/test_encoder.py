import torch, pytest
from setconca.model.encoder import ElementEncoder

B, S, D, C = 4, 8, 64, 128

def test_ENC_01_output_shape():
    enc = ElementEncoder(D, C)
    x = torch.randn(B, S, D)
    u = enc(x)
    assert u.shape == (B, S, C), f'Got {u.shape}'

def test_ENC_02_linearity():
    """u must be a linear function of x — f(ax+by) == af(x)+bf(y)."""
    enc = ElementEncoder(D, C)
    enc.eval()
    x1, x2 = torch.randn(1, 1, D), torch.randn(1, 1, D)
    a, b = 2.5, -1.3
    combo = enc(a * x1 + b * x2)
    linear_combo = a * enc(x1) + b * enc(x2)
    # subtract bias*a + bias*b which double-counts; recheck with zero bias
    enc.linear.bias.data.zero_()
    combo = enc(a * x1 + b * x2)
    linear_combo = a * enc(x1) + b * enc(x2)
    assert torch.allclose(combo, linear_combo, atol=1e-5)

def test_ENC_03_no_relu_applied():
    """Output must contain negative values for negative inputs."""
    enc = ElementEncoder(D, C)
    with torch.no_grad():
        enc.linear.weight.fill_(-1.0)  # force negative outputs
        enc.linear.bias.zero_()
    x = torch.ones(1, 1, D) * 0.5
    u = enc(x)
    assert (u < 0).any(), 'No negative values — ReLU may have been applied'

def test_ENC_04_unbounded_output():
    """Output range must be unbounded (not clipped to [0,1] etc)."""
    enc = ElementEncoder(D, C)
    x = torch.randn(B, S, D) * 100
    u = enc(x)
    assert u.abs().max() > 10, 'Suspiciously small range — check for clipping'

def test_ENC_05_permutation_independence():
    """Encoding each element is independent — permuting set permutes output."""
    enc = ElementEncoder(D, C)
    x = torch.randn(1, S, D)
    perm = torch.randperm(S)
    u = enc(x)
    u_perm = enc(x[:, perm, :])
    assert torch.allclose(u[:, perm, :], u_perm, atol=1e-5)

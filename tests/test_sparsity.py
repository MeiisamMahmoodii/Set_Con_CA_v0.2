import torch, pytest
from setconca.losses.sparsity import sparsity_loss

def test_SPAR_01_output_range():
    """Loss must be in (0,1) — Sigmoid output range."""
    z = torch.randn(8, 128)
    loss = sparsity_loss(z)
    assert 0 < loss.item() < 1

def test_SPAR_02_applied_after_sigmoid():
    """Very negative z_hat -> near-zero loss (Sigmoid(z) -> 0)."""
    z = torch.full((4, 64), -50.0)
    loss = sparsity_loss(z)
    assert loss.item() < 0.01, 'Loss should be near 0 for very negative z_hat'

def test_SPAR_03_not_on_raw_values():
    """Raw L1 on large z would be huge; Sigmoid-L1 should remain bounded."""
    z = torch.randn(4, 64) * 1000
    loss = sparsity_loss(z)
    assert loss.item() < 1.1, 'Loss exceeded Sigmoid range — raw z used?'

def test_SPAR_04_zero_z_gives_half():
    """Sigmoid(0) = 0.5, so zero z_hat should give loss ~0.5."""
    z = torch.zeros(32, 128)
    loss = sparsity_loss(z)
    assert abs(loss.item() - 0.5) < 0.01

def test_SPAR_05_gradient_flows():
    z = torch.randn(4, 64, requires_grad=True)
    loss = sparsity_loss(z)
    loss.backward()
    assert z.grad is not None
    assert not torch.isnan(z.grad).any()

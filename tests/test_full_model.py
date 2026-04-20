import torch, pytest
from setconca.model.setconca import SetConCA, compute_loss

B, S, D, C = 4, 8, 64, 128

def test_FULL_01_forward_shapes():
    model = SetConCA(D, C)
    x = torch.randn(B, S, D)
    f_hat, z_hat, u = model(x)
    assert f_hat.shape == (B, S, D)
    assert z_hat.shape == (B, C)
    assert u.shape == (B, S, C)

def test_FULL_02_loss_components_are_separate():
    model = SetConCA(D, C)
    x = torch.randn(B, S, D)
    total, parts = compute_loss(model, x)
    assert 'mse' in parts and 'sparsity' in parts and 'consistency' in parts
    # total should equal sum of parts
    assert abs(total.item() - sum(v.item() for v in parts.values())) < 1e-5

def test_FULL_03_loss_decreases_with_training():
    """Loss should go down over 50 gradient steps on synthetic data."""
    model = SetConCA(D, C)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(B, S, D)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss, _ = compute_loss(model, x)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f'Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}'

def test_FULL_04_no_nan_in_loss():
    model = SetConCA(D, C)
    for _ in range(10):
        x = torch.randn(B, S, D)
        loss, _ = compute_loss(model, x)
        assert not torch.isnan(loss), 'NaN loss detected'

def test_FULL_05_gradients_reach_all_parameters():
    model = SetConCA(D, C)
    x = torch.randn(B, S, D)
    loss, _ = compute_loss(model, x)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f'No gradient for {name}'

def test_FULL_06_parameter_count():
    """Verify exact parameter count matches spec."""
    model = SetConCA(D, C)
    total = sum(p.numel() for p in model.parameters())
    expected = D*C + C + C*D + C*D + D  # W_e,b_e,W_shared,W_resid,b_d
    assert total == expected, f'Got {total}, expected {expected}'

def test_FULL_07_alpha_zero_disables_sparsity():
    model = SetConCA(D, C)
    x = torch.randn(B, S, D)
    _, p1 = compute_loss(model, x, alpha=0.0)
    assert p1['sparsity'].item() == 0.0

def test_FULL_08_beta_zero_disables_consistency():
    model = SetConCA(D, C)
    x = torch.randn(B, S, D)
    _, p1 = compute_loss(model, x, beta=0.0)
    assert p1['consistency'].item() == 0.0

def test_FULL_09_sparsity_is_not_constant_during_training():
    """
    Regression test for the 'frozen sparsity' bug.

    When sparsity was applied to z_hat (LayerNorm output), Sigmoid was always
    ~0.5 and the loss was stuck at ~alpha*0.5 for all 100 epochs.
    Now that sparsity is applied to u_bar (pre-norm), the encoder can shift
    u_bar away from zero, so the sparsity term must visibly change as the
    encoder weights change under gradient descent.
    """
    torch.manual_seed(0)
    model = SetConCA(D, C)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(B, S, D)

    sparsity_values = []
    for _ in range(30):
        opt.zero_grad()
        loss, parts = compute_loss(model, x, alpha=1.0, beta=0.0)  # high alpha, isolate sparsity
        loss.backward()
        opt.step()
        sparsity_values.append(round(parts['sparsity'].item(), 6))

    unique = len(set(sparsity_values))
    assert unique > 5, (
        f'Sparsity loss is nearly constant ({unique} unique values in 30 steps). '
        f'Check that sparsity_loss receives u_bar, not z_hat.\n'
        f'Values: {sparsity_values}'
    )

def test_FULL_10_sparsity_applied_to_u_bar_not_z_hat():
    """
    Directly verify that sparsity sees pre-norm values.
    If we force encoder bias to +10, u_bar mean will be large positive,
    Sigmoid(u_bar) >> 0.5, and sparsity >> alpha*0.5.
    If it were applied to z_hat (LayerNorm output), it would still be ~0.5.
    """
    model = SetConCA(D, C)
    with torch.no_grad():
        model.encoder.linear.bias.fill_(10.0)   # force large positive u_bar
    x = torch.randn(B, S, D)
    _, parts = compute_loss(model, x, alpha=1.0)
    # Sigmoid(10) ~ 1.0, so sparsity should be >> 0.5 * alpha=1.0
    assert parts['sparsity'].item() > 0.8, (
        f"Sparsity={parts['sparsity'].item():.4f} — expected > 0.8 with bias=+10. "
        "Likely still using z_hat (LayerNorm output) instead of u_bar."
    )

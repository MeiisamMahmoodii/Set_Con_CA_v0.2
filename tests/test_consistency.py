import torch, pytest
from setconca.model.encoder import ElementEncoder
from setconca.model.aggregator import SetAggregator
from setconca.losses.consistency import consistency_loss

B, S, D, C = 4, 8, 32, 64

def make_encode_agg(enc, agg):
    def fn(x):
        u = enc(x)
        z, _, _ = agg(u)   # (z_hat, u_bar, u) — discard u_bar and u
        return z
    return fn

def test_CONS_01_output_is_scalar():
    enc, agg = ElementEncoder(D, C), SetAggregator(C)
    fn = make_encode_agg(enc, agg)
    x = torch.randn(B, S, D)
    loss = consistency_loss(x, fn)
    assert loss.shape == ()

def test_CONS_02_skips_for_small_sets():
    enc, agg = ElementEncoder(D, C), SetAggregator(C)
    fn = make_encode_agg(enc, agg)
    x = torch.randn(B, 3, D)  # set_size < 4
    loss = consistency_loss(x, fn)
    assert loss.item() == 0.0

def test_CONS_03_loss_is_nonnegative():
    enc, agg = ElementEncoder(D, C), SetAggregator(C)
    fn = make_encode_agg(enc, agg)
    x = torch.randn(B, S, D)
    for _ in range(5):
        assert consistency_loss(x, fn).item() >= 0

def test_CONS_04_gradient_flows_to_encoder():
    """Encoder weights must receive gradients via consistency loss."""
    enc, agg = ElementEncoder(D, C), SetAggregator(C)
    fn = make_encode_agg(enc, agg)
    x = torch.randn(B, S, D)
    loss = consistency_loss(x, fn)
    loss.backward()
    assert enc.linear.weight.grad is not None
    assert not torch.isnan(enc.linear.weight.grad).any()

def test_CONS_05_identical_sets_give_low_loss():
    """If both halves are identical (all same vector), loss should be 0."""
    enc, agg = ElementEncoder(D, C), SetAggregator(C)
    enc.eval(); agg.eval()
    fn = make_encode_agg(enc, agg)
    # Constant set — any split gives same mean, same z_hat
    v = torch.randn(1, 1, D).expand(B, S, D).contiguous()
    loss = consistency_loss(v, fn)
    assert loss.item() < 1e-4, f'Constant set gave loss {loss.item()}'

def test_CONS_06_different_random_splits_each_call():
    """Loss should vary across calls due to random splits."""
    enc, agg = ElementEncoder(D, C), SetAggregator(C)
    fn = make_encode_agg(enc, agg)
    x = torch.randn(B, S, D)
    losses = [consistency_loss(x, fn).item() for _ in range(10)]
    # Not all the same (would indicate deterministic split)
    assert len(set(round(l, 6) for l in losses)) > 1

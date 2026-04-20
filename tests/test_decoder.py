import torch, pytest
from setconca.model.decoder import DualDecoder

B, S, C, D = 4, 8, 128, 64

def test_DEC_01_output_shape():
    dec = DualDecoder(C, D)
    z_hat = torch.randn(B, C)
    u = torch.randn(B, S, C)
    out = dec(z_hat, u)
    assert out.shape == (B, S, D), f'Got {out.shape}'

def test_DEC_02_single_bias_parameter():
    """Must have exactly one b_d, not two."""
    dec = DualDecoder(C, D)
    named = dict(dec.named_parameters())
    assert 'b_d' in named
    assert 'shared.bias' not in named
    assert 'residual.bias' not in named
    assert named['b_d'].shape == (D,)

def test_DEC_03_shared_stream_broadcasts():
    """Shared component must be identical across set dimension."""
    dec = DualDecoder(C, D)
    dec.residual.weight.data.zero_()  # zero residual to isolate shared
    dec.b_d.data.zero_()
    z_hat = torch.randn(B, C)
    u = torch.zeros(B, S, C)
    out = dec(z_hat, u)
    # all S slices should be identical
    for i in range(1, S):
        assert torch.allclose(out[:, 0, :], out[:, i, :], atol=1e-5)

def test_DEC_04_residual_stream_varies_per_element():
    """Residual stream makes output differ across set elements."""
    dec = DualDecoder(C, D)
    dec.shared.weight.data.zero_()
    dec.b_d.data.zero_()
    z_hat = torch.zeros(B, C)
    u = torch.randn(B, S, C)  # different per element
    out = dec(z_hat, u)
    assert not torch.allclose(out[:, 0, :], out[:, 1, :], atol=1e-3)

def test_DEC_05_bias_shape():
    dec = DualDecoder(C, D)
    assert dec.b_d.shape == (D,)

def test_DEC_06_gradient_flows_to_both_streams():
    dec = DualDecoder(C, D)
    z_hat = torch.randn(B, C, requires_grad=True)
    u = torch.randn(B, S, C, requires_grad=True)
    out = dec(z_hat, u)
    loss = out.mean()
    loss.backward()
    assert z_hat.grad is not None
    assert u.grad is not None

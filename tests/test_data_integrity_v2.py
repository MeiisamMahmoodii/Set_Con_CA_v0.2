import torch, pytest
from setconca.data.dataset import RepresentationSetDataset, make_synthetic_dataset, make_dataloader

def test_DATA_01_output_shape():
    """Datalor must yield (B, S, D) tensors."""
    ds = make_synthetic_dataset(n_sets=64, set_size=8, hidden_dim=32)
    dl = make_dataloader(ds, batch_size=16)
    batch = next(iter(dl))
    assert batch.shape == (16, 8, 32), f'Got {batch.shape}'

def test_DATA_02_no_preprocessing():
    """Values must be unchanged — no normalization applied."""
    raw = torch.randn(4, 3, 8) * 100  # extreme range to detect scaling
    ds = RepresentationSetDataset(raw)
    assert torch.allclose(ds[0], raw[0]), 'Data was modified in dataset'

def test_DATA_03_rejects_2d_input():
    """Must raise on (N, D) input — sets required."""
    with pytest.raises(AssertionError):
        RepresentationSetDataset(torch.randn(64, 32))

def test_DATA_04_deterministic_without_shuffle():
    """Without shuffle, two passes over loader give same order."""
    ds = make_synthetic_dataset(32, 4, 16)
    dl = make_dataloader(ds, batch_size=8, shuffle=False)
    pass1 = [b.clone() for b in dl]
    pass2 = [b.clone() for b in dl]
    for b1, b2 in zip(pass1, pass2):
        assert torch.allclose(b1, b2)

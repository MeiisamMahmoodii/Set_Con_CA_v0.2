import torch
from torch.utils.data import Dataset, DataLoader

class RepresentationSetDataset(Dataset):
    """
    Each item is a set of hidden states of shape (set_size, hidden_dim).
    Sets represent paraphrases, trajectory steps, or local neighbourhoods.
    """
    def __init__(self, data: torch.Tensor):
        # data shape: (N, set_size, hidden_dim)
        assert data.ndim == 3, 'Expected (N, set_size, hidden_dim)'
        self.data = data  # stored as-is, no preprocessing

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]  # (set_size, hidden_dim)

def make_synthetic_dataset(n_sets=512, set_size=8, hidden_dim=64):
    """Synthetic Gaussian data for unit tests."""
    data = torch.randn(n_sets, set_size, hidden_dim)
    return RepresentationSetDataset(data)

def make_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

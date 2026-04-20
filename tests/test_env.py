import torch, numpy, einops, pytest
import os

def test_torch_version():
    assert int(torch.__version__.split('.')[0]) >= 2

def test_cuda_or_cpu():
    # must run on at least CPU; log whether CUDA available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 4, 8).to(device)
    assert x.shape == (2, 4, 8)

def test_directory_structure():
    for d in ['model','losses','data','tests','docs']:
        assert os.path.isdir(f'setconca/{d}'), f'Missing: setconca/{d}'

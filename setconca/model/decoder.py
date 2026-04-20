import torch
import torch.nn as nn

class DualDecoder(nn.Module):
    """
    Shared + Residual decoder.
    f_hat(x_i) = W_shared * z_hat + W_residual * u_i + b_d
    Single bias b_d shared across both streams.
    """
    def __init__(self, concept_dim: int, hidden_dim: int):
        super().__init__()
        # No bias in either Linear — single b_d added manually
        self.shared = nn.Linear(concept_dim, hidden_dim, bias=False)
        self.residual = nn.Linear(concept_dim, hidden_dim, bias=False)
        self.b_d = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.shared.weight)
        nn.init.xavier_uniform_(self.residual.weight)

    def forward(self, z_hat: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # z_hat: (B, C), u: (B, S, C)
        shared_out = self.shared(z_hat).unsqueeze(1)  # (B,1,D) broadcast
        resid_out = self.residual(u)                   # (B,S,D)
        return shared_out + resid_out + self.b_d       # (B,S,D)

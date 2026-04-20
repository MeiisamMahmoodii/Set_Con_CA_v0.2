import torch
import torch.nn as nn
from typing import Tuple

class AttentionAggregator(nn.Module):
    """
    Learned attention-based aggregation of set elements.
    Uses a learnable query to calculate importance weights for each element.
    """
    def __init__(self, concept_dim: int, n_heads: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, concept_dim))
        self.mha = nn.MultiheadAttention(concept_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(concept_dim, elementwise_affine=False)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # u: (B, S, C)
        # Broadcast query to (B, 1, C)
        query = self.query.expand(u.size(0), -1, -1)
        # z, _ = mha(query, key, value)
        z, _ = self.mha(query, u, u)
        u_bar = z.squeeze(1) # (B, C)
        z_hat = self.norm(u_bar)
        return z_hat, u_bar


class SetAggregator(nn.Module):
    def __init__(self, concept_dim: int, dropout_p: float = 0.0, mode: str = 'mean'):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(concept_dim, elementwise_affine=False)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        
        if mode == 'attention':
            self.att = AttentionAggregator(concept_dim)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # u: (B, S, C)
        if self.mode == 'attention':
            z_hat, u_bar = self.att(u)
        else:
            u_bar = u.mean(dim=1)          # mean pool -> (B, C)
            z_hat = self.norm(u_bar)
            
        z_hat = self.dropout(z_hat)
        return z_hat, u_bar, u

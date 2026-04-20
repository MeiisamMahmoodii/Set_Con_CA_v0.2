import torch
import torch.nn as nn
from .encoder import ElementEncoder
from .aggregator import SetAggregator
from .decoder import DualDecoder
from ..losses.sparsity import sparsity_loss
from ..losses.consistency import consistency_loss

class SetConCA(nn.Module):
    def __init__(self, hidden_dim, concept_dim, dropout_p=0.0, use_topk=False, k=32):
        super().__init__()
        self.encoder = ElementEncoder(hidden_dim, concept_dim)
        self.aggregator = SetAggregator(concept_dim, dropout_p)
        self.decoder = DualDecoder(concept_dim, hidden_dim)
        self.use_topk = use_topk
        self.k = k

    def encode_and_aggregate(self, x):
        u = self.encoder(x)
        z_hat, u_bar, u = self.aggregator(u)
        
        if self.use_topk:
            # Apply Top-K Sparsity
            topk_vals, topk_indices = torch.topk(z_hat, self.k, dim=-1)
            mask = torch.zeros_like(z_hat)
            mask.scatter_(-1, topk_indices, 1.0)
            z_hat = z_hat * mask
            
        return z_hat, u_bar, u

    def forward(self, x):
        z_hat, u_bar, u = self.encode_and_aggregate(x)  # (B,C), (B,C), (B,S,C)
        f_hat = self.decoder(z_hat, u)                   # (B,S,D)
        return f_hat, z_hat, u


def compute_loss(model, x, alpha=1e-3, beta=1e-2):
    # Step through internals explicitly to access u_bar for sparsity.
    u = model.encoder(x)
    z_hat, u_bar, u_res = model.aggregator(u)
    
    if getattr(model, 'use_topk', False):
        # Top-K Sparsity: Directly zero out non-top-k indices of z_hat
        topk_vals, topk_indices = torch.topk(z_hat, model.k, dim=-1)
        mask = torch.zeros_like(z_hat)
        mask.scatter_(-1, topk_indices, 1.0)
        z_hat_sparse = z_hat * mask
        f_hat = model.decoder(z_hat_sparse, u_res)
        
        # MSE: mean over batch and set elements
        mse = ((f_hat - x) ** 2).mean(dim=-1).mean(dim=-1).mean()
        
        # In fixed-k Top-K, we don't need the L1/Sigmoid sparsity loss (it's hard-coded k)
        spar = torch.tensor(0.0, device=x.device)
    else:
        # Standard Sigmoid + L1 Path
        f_hat = model.decoder(z_hat, u_res)
        mse = ((f_hat - x) ** 2).mean(dim=-1).mean(dim=-1).mean()
        spar = alpha * sparsity_loss(u_bar)

    # Consistency: mean over batch
    def _enc_agg(x_sub):
        u_sub = model.encoder(x_sub)
        z_sub, _, _ = model.aggregator(u_sub)
        if getattr(model, 'use_topk', False):
             _, topk_indices = torch.topk(z_sub, model.k, dim=-1)
             mask = torch.zeros_like(z_sub)
             mask.scatter_(-1, topk_indices, 1.0)
             z_sub = z_sub * mask
        return z_sub
        
    cons = beta * consistency_loss(x, _enc_agg)

    total = mse + spar + cons
    return total, {'mse': mse, 'sparsity': spar, 'consistency': cons}

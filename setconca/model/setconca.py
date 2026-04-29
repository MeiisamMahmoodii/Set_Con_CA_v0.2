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


def compute_loss(model, x, alpha=1e-3, beta=1e-2, gamma=10.0, lambda_res=0.1):
    # Step through internals explicitly to access u_bar for sparsity.
    u = model.encoder(x)
    z_hat, u_bar, u_res = model.aggregator(u)
    
    if getattr(model, 'use_topk', False):
        # Top-K Sparsity: Directly zero out non-top-k indices of z_hat
        topk_vals, topk_indices = torch.topk(z_hat, model.k, dim=-1)
        mask = torch.zeros_like(z_hat)
        mask.scatter_(-1, topk_indices, 1.0)
        z_hat_sparse = z_hat * mask
        
        # 1. Full Reconstruction (Shared + Residual)
        f_hat_full = model.decoder(z_hat_sparse, u_res)
        mse_full = ((f_hat_full - x) ** 2).mean()
        
        # 2. Semantic Exclusive Reconstruction (Shared ONLY)
        # Force the concept vector Z to carry the weight of reconstruction.
        f_hat_shared = model.decoder.shared(z_hat_sparse).unsqueeze(1) + model.decoder.b_d
        mse_shared = ((f_hat_shared - x) ** 2).mean()
        
        # Weigh shared loss much higher than full/residual loss
        mse = gamma * mse_shared + lambda_res * mse_full
        spar = torch.tensor(0.0, device=x.device)
    else:
        # Standard Sigmoid + L1 Path
        f_hat_full = model.decoder(z_hat, u_res)
        mse_full = ((f_hat_full - x) ** 2).mean()
        
        f_hat_shared = model.decoder.shared(z_hat).unsqueeze(1) + model.decoder.b_d
        mse_shared = ((f_hat_shared - x) ** 2).mean()
        
        mse = gamma * mse_shared + lambda_res * mse_full
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
    return total, {'mse': mse, 'mse_full': mse_full, 'mse_shared': mse_shared, 'sparsity': spar, 'consistency': cons}

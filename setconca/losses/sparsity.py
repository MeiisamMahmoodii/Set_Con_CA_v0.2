import torch
import torch.nn.functional as F

def sparsity_loss(u_bar: torch.Tensor) -> torch.Tensor:
    """
    L1 sparsity on probability-mapped latents (paper Sec. 4.4).

    Input: u_bar = (1/m) sum_j W_e f(x_j) + b_e  — the pre-LayerNorm
    mean-pooled encoding, which approximates [log p(z_i | X)] (Eq. 20).

    g(u_bar) = Sigmoid(u_bar) maps to (0, 1), acting as a smooth surrogate
    for the concept activation probability. L1 on non-negatives equals their
    mean, which pushes activations toward 0 (concept absent).

    NOTE: Must receive u_bar (pre-norm), NOT z_hat (post-LayerNorm).
    LayerNorm forces zero-mean, which locks Sigmoid output to ~0.5 and
    kills the gradient signal entirely.
    """
    g = torch.sigmoid(u_bar)   # (B, C) -> values in (0, 1)
    return g.mean()            # L1 of non-negatives = mean; minimising pushes toward 0

# Alias to make intent explicit at call sites:
def probability_domain_l1(u_bar: torch.Tensor) -> torch.Tensor:
    return sparsity_loss(u_bar)

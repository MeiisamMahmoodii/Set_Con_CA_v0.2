import torch
from typing import Callable, Tuple

def consistency_loss(
    x: torch.Tensor,
    encode_and_aggregate: Callable  # fn(x_subset) -> z_hat
) -> torch.Tensor:
    """
    Subset consistency regularization.
    Splits x into two disjoint halves, runs both through shared encoder,
    returns squared L2 distance between set-level latents.
    """
    B, S, D = x.shape
    if S < 4:
        return torch.tensor(0.0, device=x.device, requires_grad=False)

    k = S // 2
    perm = torch.randperm(S, device=x.device)
    idx_a, idx_b = perm[:k], perm[k:]

    x_a = x[:, idx_a, :]  # (B, k, D)
    x_b = x[:, idx_b, :]  # (B, S-k, D)

    # Both share encoder weights — same function object
    z_a = encode_and_aggregate(x_a)  # (B, C)
    z_b = encode_and_aggregate(x_b)  # (B, C)

    return ((z_a - z_b) ** 2).sum(dim=-1).mean()  # mean over batch

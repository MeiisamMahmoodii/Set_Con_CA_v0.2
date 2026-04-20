import torch
import torch.nn as nn

class ElementEncoder(nn.Module):
    """
    Linear encoder applied independently to each set element.
    u_i = W_e * f(x_i) + b_e
    No activation — preserves log-posterior interpretation.
    """
    def __init__(self, hidden_dim: int, concept_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, concept_dim, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> u: (B, S, C)
        # nn.Linear broadcasts over leading dims — no reshape needed
        return self.linear(x)  # purely linear, no activation

import torch
import torch.nn as nn
import numpy as np
from setconca.model.setconca import SetConCA, compute_loss
from .eval_metrics import topk_overlap

# Note: We avoid fundamentally altering `setconca.model.aggregator.SetAggregator` permanently
# by using subclassing / mocking for the ablation test.
class AblatedAggregator(nn.Module):
    def __init__(self, original_aggregator, mode="mean"):
        super().__init__()
        self.original = original_aggregator
        self.mode = mode
        
    def forward(self, u):
        B, S, C = u.shape
        if self.mode == "mean":
            # Simple average
            u_bar = u.mean(dim=1)
            z_hat = u_bar
            return z_hat, u_bar, u
        elif self.mode == "random":
            # Random fixed weights matching current S (to handle subset consistency passes)
            current_S = u.shape[1]
            # Use fixed seed or stable hash so weights for subset are consistent,
            # or just generate new random weights per pass since it's an ablation of "random"
            rand_w = torch.rand(current_S, device=u.device)
            rand_w = rand_w / rand_w.sum()
            u_bar = (u * rand_w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
            z_hat = u_bar
            return z_hat, u_bar, u
        else:
            # Fallback to original attention
            return self.original(u)

def train_model(model, optimizer, data, epochs=50):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss, _ = compute_loss(model, data)
        loss.backward()
        optimizer.step()
    return model

def run_exp3(data, seeds=[42, 1337]):
    """
    Experiment 3: Aggregator Ablation
    """
    print("--- Running Experiment 3: Aggregator Ablation ---")
    D = data.shape[-1]
    results = {}
    
    modes = ["attention", "mean", "random"]
    
    for mode in modes:
        mode_res = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = SetConCA(hidden_dim=D, concept_dim=64, use_topk=True)
            
            # Sub into the pipeline
            model.aggregator = AblatedAggregator(model.aggregator, mode=mode)
            
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_model(model, opt, data)
            
            model.eval()
            with torch.no_grad():
                f_hat, z, _ = model(data)
                mse = ((f_hat - data)**2).mean().item()
            mode_res.append({"mse": mse, "z": z})
            
        stab = topk_overlap(mode_res[0]["z"].numpy(), mode_res[1]["z"].numpy())
        avg_mse = np.mean([r["mse"] for r in mode_res])
        results[mode] = {"stability": stab, "mse": avg_mse}
        print(f"Aggregator [{mode}] | Stability: {stab:.4f} | MSE: {avg_mse:.4f}")
        
    return results

if __name__ == "__main__":
    test_data = torch.randn(10, 8, 128)
    run_exp3(test_data)

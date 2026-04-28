import torch
import numpy as np
from setconca.model.setconca import SetConCA, compute_loss
from .eval_metrics import topk_overlap

def train_model(model, optimizer, data, epochs=50):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss, _ = compute_loss(model, data)
        loss.backward()
        optimizer.step()
    return model

def run_exp2(data, sweep=[1, 3, 8, 16, 32], seeds=[42, 1337]):
    """
    Experiment 2: S-Scaling
    data should have at least max(sweep) items in the set dimension.
    """
    print("--- Running Experiment 2: Set Size Scaling Sweep ---")
    D = data.shape[-1]
    
    results = {}
    
    for S in sweep:
        # Prevent indexing larger than data allows
        actual_S = min(S, data.shape[1])
        data_sub = data[:, :actual_S, :]
        
        S_res = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = SetConCA(hidden_dim=D, concept_dim=64, use_topk=True)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_model(model, opt, data_sub)
            
            model.eval()
            with torch.no_grad():
                f_hat, z, _ = model(data_sub)
                mse = ((f_hat - data_sub)**2).mean().item()
            S_res.append({"mse": mse, "z": z})
            
        z_v1 = S_res[0]["z"].numpy()
        z_v2 = S_res[1]["z"].numpy()
        stab = topk_overlap(z_v1, z_v2)
        avg_mse = np.mean([r["mse"] for r in S_res])
        
        results[S] = {"stability": stab, "mse": avg_mse}
        print(f"S={S} | Stability: {stab:.4f} | MSE: {avg_mse:.4f}")
        
    return results

if __name__ == "__main__":
    test_data = torch.randn(10, 32, 128)
    run_exp2(test_data)

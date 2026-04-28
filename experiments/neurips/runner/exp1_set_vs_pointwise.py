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

def run_exp1(data, seeds=[42, 1337, 2024]):
    """
    Experiment 1: Set vs Pointwise
    data shape: (N_sets, S, D)
    """
    print("--- Running Experiment 1: Set vs Pointwise ---")
    D = data.shape[-1]
    
    results = {"SetConCA": [], "Pointwise": []}
    
    for seed in seeds:
        torch.manual_seed(seed)
        
        # 1. SetConCA (S=8, though S comes from data shape)
        model_set = SetConCA(hidden_dim=D, concept_dim=64, use_topk=True)
        opt_set = torch.optim.Adam(model_set.parameters(), lr=1e-3)
        train_model(model_set, opt_set, data)
        
        # Eval SetConCA
        model_set.eval()
        with torch.no_grad():
            f_hat_set, z_set, _ = model_set(data)
            mse_set = ((f_hat_set - data)**2).mean().item()
        
        # 2. Pointwise (S=1 baseline)
        # We simulate pointwise by flattening S into batch dimension or just using S=1
        data_point = data[:, 0:1, :]
        model_point = SetConCA(hidden_dim=D, concept_dim=64, use_topk=True)
        opt_point = torch.optim.Adam(model_point.parameters(), lr=1e-3)
        train_model(model_point, opt_point, data_point)
        
        model_point.eval()
        with torch.no_grad():
            f_hat_point, z_point, _ = model_point(data_point)
            mse_point = ((f_hat_point - data_point)**2).mean().item()
            
        results["SetConCA"].append({"mse": mse_set, "z": z_set})
        results["Pointwise"].append({"mse": mse_point, "z": z_point})
        
    # Analyze stability (variance of Z across seeds)
    def calc_stability(res_list):
        # We approximate stability by cross-seed overlap
        z_v1 = res_list[0]["z"].numpy()
        z_v2 = res_list[1]["z"].numpy()
        return topk_overlap(z_v1, z_v2)

    set_mse_avg = np.mean([r["mse"] for r in results["SetConCA"]])
    pt_mse_avg = np.mean([r["mse"] for r in results["Pointwise"]])
    
    set_stab = calc_stability(results["SetConCA"])
    pt_stab = calc_stability(results["Pointwise"])
    
    print(f"SetConCA MSE: {set_mse_avg:.4f} | Stability (Overlap): {set_stab:.4f}")
    print(f"Pointwise MSE: {pt_mse_avg:.4f} | Stability (Overlap): {pt_stab:.4f}")
    
    return {
        "SetConCA": {"mse": set_mse_avg, "stability": set_stab},
        "Pointwise": {"mse": pt_mse_avg, "stability": pt_stab}
    }

if __name__ == "__main__":
    # Dummy data test
    test_data = torch.randn(10, 8, 128)
    run_exp1(test_data)

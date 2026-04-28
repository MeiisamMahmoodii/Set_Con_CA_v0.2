import torch
import torch.nn as nn
import numpy as np
from setconca.model.setconca import SetConCA, compute_loss
from .eval_metrics import apply_steering
from .exp4_cross_model_transfer import train_setconca, train_bridge

def run_exp5(data_source, data_target, concept_idx=0):
    """
    Experiment 5: Interventional Steering
    data_source: shape (N_anchors, S, D_source)
    data_target: shape (N_anchors, S, D_target)
    
    We simulate finding a concept vector in the source and measuring effect size
    upon application to target vectors.
    """
    print("--- Running Experiment 5: Interventional Steering ---")
    
    model_s, z_s = train_setconca(data_source)
    model_t, z_t = train_setconca(data_target)
    
    B = train_bridge(z_s, z_t)
    
    # 1. Get concept vector
    # Usually this is isolated by matching prompts "refusal". For simulation, we take a specific anchor's concept representation.
    z_refusal = z_s[concept_idx] # (C)
    
    # Target baseline (say, anchor 1, not refusal)
    x = data_target[1, 0, :] # Base activation in target model
    
    # 2. Add intervention
    alpha = 5.0
    x_prime = apply_steering(x, z_refusal, B, alpha=alpha, decoder=model_t.decoder) # shape (D)
    
    # Random steering baseline
    z_rand = torch.randn_like(z_refusal)
    x_prime_rand = apply_steering(x, z_rand, B, alpha=alpha, decoder=model_t.decoder)
    
    # 3. Simulate Run Model / Probability Shift
    # Since we can't run full Llama inference easily, we measure the cosine change towards the true mapped target concept
    x_target_true = data_target[concept_idx, 0, :].detach().numpy()
    
    def measure_effect(x_mod, x_target):
        # We want x_mod to move closer to x_target
        if torch.is_tensor(x_mod):
            x_mod = x_mod.detach().numpy()
        cos_sim = np.dot(x_mod, x_target) / (np.linalg.norm(x_mod) * np.linalg.norm(x_target) + 1e-8)
        return cos_sim
        
    base_sim = measure_effect(x, x_target_true)
    sc_sim = measure_effect(x_prime, x_target_true)
    rand_sim = measure_effect(x_prime_rand, x_target_true)
    
    # Strong effect if similarity to concept target increases heavily
    print(f"Base Similarity to Concept: {base_sim:.4f}")
    print(f"Set-ConCA Intervention  : {sc_sim:.4f}")
    print(f"Random Intervention     : {rand_sim:.4f}")
    
    return {
        "base": base_sim,
        "setconca": sc_sim,
        "random": rand_sim
    }

if __name__ == "__main__":
    data_1 = torch.randn(20, 8, 128)
    rot = torch.randn(128, 128)
    q, _ = torch.linalg.qr(rot)
    data_2 = (data_1 @ q) + torch.randn_like(data_1)*0.1
    run_exp5(data_1, data_2)

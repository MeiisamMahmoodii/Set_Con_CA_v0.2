import torch
import torch.nn as nn
import numpy as np
from setconca.model.setconca import SetConCA, compute_loss
from .eval_metrics import cka, evaluate_transfer

def train_bridge(z_source, z_target, epochs=200):
    """
    Orthogonal Procrustes / Linear bridge
    B = argmin || z_source @ B - z_target ||
    We learn it via standard GD with an orthogonality penalty.
    """
    C = z_source.shape[-1]
    B = nn.Parameter(torch.eye(C) + torch.randn(C, C)*0.01)
    opt = torch.optim.Adam([B], lr=1e-2)
    
    for _ in range(epochs):
        opt.zero_grad()
        mapped = z_source @ B
        mse = ((mapped - z_target)**2).mean()
        # Orthogonality loss
        ortho = ((B.T @ B - torch.eye(C))**2).mean()
        loss = mse + 0.1 * ortho
        loss.backward()
        opt.step()
        
    return B.detach()

def train_setconca(data):
    D = data.shape[-1]
    model = SetConCA(hidden_dim=D, concept_dim=64, use_topk=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(50):
        opt.zero_grad()
        loss, _ = compute_loss(model, data)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        _, z, _ = model(data)
    return model, z

def run_exp4(data_source, data_target):
    """
    Experiment 4: Cross-Model Transfer
    """
    print("--- Running Experiment 4: Cross-Model Transfer ---")
    
    # 1. Train SetConCA on both
    model_s, z_s = train_setconca(data_source)
    model_t, z_t = train_setconca(data_target)
    
    # 2. Train Bridge
    B = train_bridge(z_s, z_t)
    
    # 3. Eval SetConCA Transfer
    transfer_score = evaluate_transfer(z_s, z_t, B)
    cka_score = cka(z_s, z_t)
    
    # 4. Pointwise Baseline (Using raw means or elements to simulate pointwise)
    # To keep Top-k valid and dimensions matching (64-dim concept bottleneck), 
    # we simulate pointwise concept extraction by picking S=1 model on the raw data.
    # To keep Top-k valid, we simulate pointwise concept extraction by picking S=1 model.
    model_pt_s, z_pt_s = train_setconca(data_source[:,0:1,:])
    model_pt_t, z_pt_t = train_setconca(data_target[:,0:1,:])
    B_pt = train_bridge(z_pt_s, z_pt_t)
    pt_overlap = evaluate_transfer(z_pt_s, z_pt_t, B_pt)
    
    # 5. Random
    z_rand = torch.randn_like(z_s)
    B_rand = train_bridge(z_s, z_rand)
    rand_overlap = evaluate_transfer(z_s, z_t, B_rand)
    
    print(f"Set-ConCA Top-K Overlap : {transfer_score:.4f} | CKA: {cka_score:.4f}")
    print(f"Pointwise Top-K Overlap : {pt_overlap:.4f}")
    print(f"Random Map Top-K Overlap: {rand_overlap:.4f}")
    
    return {
        "setconca_overlap": transfer_score,
        "pointwise_overlap": pt_overlap,
        "random_overlap": rand_overlap,
        "cka": cka_score
    }

if __name__ == "__main__":
    # Same anchors, different spaces (different norms/angles)
    data_1 = torch.randn(20, 8, 128)
    # Target is rotated/shifted version
    rot = torch.randn(128, 128)
    q, _ = torch.linalg.qr(rot)
    data_2 = (data_1 @ q) + torch.randn_like(data_1)*0.1
    run_exp4(data_1, data_2)

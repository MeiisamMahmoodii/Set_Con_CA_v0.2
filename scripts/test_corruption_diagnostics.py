
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setconca.model.setconca import SetConCA, compute_loss

def topk_overlap(z1, z2, k=32):
    n = min(len(z1), len(z2))
    ov = []
    for i in range(n):
        s1 = set(np.argsort(np.abs(z1[i]))[-k:])
        s2 = set(np.argsort(np.abs(z2[i]))[-k:])
        ov.append(len(s1 & s2) / k)
    return np.mean(ov)

def test_corruption_diagnostics():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {DEVICE}")
    
    # Load a small slice of data
    data_g = torch.load("data/gemma3_4b_dataset.pt", weights_only=False)["hidden"][:128].float().to(DEVICE)
    data_l = torch.load("data/llama_8b_dataset.pt", weights_only=False)["hidden"][:128].float().to(DEVICE)
    
    N, S, D = data_g.shape
    
    # Train Model A (Gemma)
    model_a = SetConCA(D, 128, use_topk=True, k=32).to(DEVICE)
    model_a.decoder.residual_scale = 0.0 # FORCE SHARED
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    for _ in range(50):
        loss, _ = compute_loss(model_a, data_g)
        opt_a.zero_grad(); loss.backward(); opt_a.step()
    
    # Train Model B (Llama)
    model_b = SetConCA(data_l.shape[-1], 128, use_topk=True, k=32).to(DEVICE)
    model_b.decoder.residual_scale = 0.0 # FORCE SHARED
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    for _ in range(50):
        loss, _ = compute_loss(model_b, data_l)
        opt_b.zero_grad(); loss.backward(); opt_b.step()
        
    model_a.eval()
    model_b.eval()
    
    with torch.no_grad():
        _, z_a_clean, _ = model_a(data_g)
        _, z_b_clean, _ = model_b(data_l)
        
    # Baseline Transfer (Clean)
    B = torch.linalg.lstsq(z_a_clean, z_b_clean).solution
    z_a_mapped = z_a_clean @ B
    clean_overlap = topk_overlap(z_a_mapped.cpu().numpy(), z_b_clean.cpu().numpy())
    print(f"Clean Transfer: {clean_overlap:.4f}")
    
    # 100% Corruption: Replace ALL elements with random elements from other anchors
    data_corrupt = data_g.clone()
    for i in range(N):
        rand_idx = torch.randint(0, N, (1,)).item()
        data_corrupt[i] = data_g[rand_idx]
        
    with torch.no_grad():
        _, z_a_corrupt, _ = model_a(data_corrupt)
        
    z_a_corrupt_mapped = z_a_corrupt @ B
    corrupt_overlap = topk_overlap(z_a_corrupt_mapped.cpu().numpy(), z_b_clean.cpu().numpy())
    print(f"Corrupt Transfer (Random Anchors): {corrupt_overlap:.4f}")
    
    # Self-Similarity Check
    # Is every z vector the same?
    sim_matrix = torch.nn.functional.cosine_similarity(z_a_clean.unsqueeze(1), z_a_clean.unsqueeze(0), dim=-1)
    avg_self_sim = (sim_matrix.sum() - N) / (N * (N - 1))
    print(f"Avg Pairwise Cosine Sim (Model A Clean): {avg_self_sim:.4f}")
    
    if avg_self_sim > 0.9:
        print("DIAGNOSIS: REPRESENTATION COLLAPSE! All anchors produce nearly the same Z vector.")
    
    # Check Residual vs Shared Energy
    with torch.no_grad():
        u = model_a.encoder(data_g)
        z_hat, _, u_res = model_a.aggregator(u)
        # Apply TopK to z_hat to match training
        topk_vals, topk_indices = torch.topk(z_hat, 32, dim=-1)
        mask = torch.zeros_like(z_hat); mask.scatter_(-1, topk_indices, 1.0)
        z_sparse = z_hat * mask
        
        shared_out = model_a.decoder.shared(z_sparse)
        resid_out = model_a.decoder.residual(u_res)
        
        shared_norm = shared_out.norm()
        resid_norm = resid_out.norm()
        print(f"Shared stream norm: {shared_norm:.4f}")
        print(f"Residual stream norm: {resid_norm:.4f}")
        
        if resid_norm > shared_norm * 2:
            print("DIAGNOSIS: RESIDUAL BYPASS! Most reconstruction info is in the residual stream.")

if __name__ == "__main__":
    test_corruption_diagnostics()

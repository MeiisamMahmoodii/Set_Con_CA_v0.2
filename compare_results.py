import torch
from setconca.model.setconca import SetConCA, compute_loss
from setconca.data.dataset import RepresentationSetDataset

def get_stats(model, data_path, alpha=0.1, beta=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    loaded = torch.load(data_path, weights_only=False)
    hidden = loaded['hidden'] if isinstance(loaded, dict) else loaded
    N, S, D = hidden.shape
    
    dataset = RepresentationSetDataset(hidden)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    total_mse = 0
    total_spar = 0
    total_u_bar = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # encoder -> aggregator -> decoder
            u = model.encoder(batch)
            z_hat, u_bar, u_res = model.aggregator(u)
            f_hat = model.decoder(z_hat, u_res)
            
            mse = ((f_hat - batch) ** 2).mean(dim=-1).mean(dim=-1)
            total_mse += mse.sum().item()
            
            # sparsity
            g = torch.sigmoid(u_bar)
            total_spar += g.mean(dim=-1).sum().item()
            total_u_bar.append(u_bar.cpu())
            
    avg_mse = total_mse / N
    avg_spar = (total_spar / N) * alpha # alpha weighted for consistency with train.py
    # Also calc raw active concepts (prob > 0.5)
    U = torch.cat(total_u_bar, dim=0)
    active_mask = (torch.sigmoid(U) > 0.5).float()
    avg_active = active_mask.sum(dim=1).mean().item()
    
    return {
        "mse": avg_mse,
        "spar_loss": avg_spar,
        "avg_active": avg_active
    }

def main():
    # Model parameters from train config
    H, C = 2560, 4096
    
    set_model_path = "data/model_set.pt"
    point_model_path = "data/model_point.pt"
    set_data_path = "data/hf_real_dataset.pt"
    point_data_path = "data/hf_point_dataset.pt"
    
    if not (os.path.exists(set_model_path) and os.path.exists(point_model_path)):
        print("Error: Models not found. Run benchmarker.py first.")
        return

    set_model = SetConCA(H, C)
    set_model.load_state_dict(torch.load(set_model_path, map_location='cpu'))
    
    point_model = SetConCA(H, C)
    point_model.load_state_dict(torch.load(point_model_path, map_location='cpu'))
    
    print("\n" + "="*50)
    print("SET-CONCA ABLATION COMPARISON")
    print("="*50)
    
    print("Calculating Set-ConCA (S=8) Stats...")
    set_stats = get_stats(set_model, set_data_path, beta=0.01)
    
    print("Calculating Point-ConCA (S=1) Stats...")
    point_stats = get_stats(point_model, point_data_path, beta=0.0)
    
    print("\nRESULTS TABLE:")
    print("-" * 65)
    print(f"{'Metric':<25} | {'Point (S=1)':<15} | {'Set (S=8)':<15}")
    print("-" * 65)
    print(f"{'MSE (Reconstruction)':<25} | {point_stats['mse']:<15.6f} | {set_stats['mse']:<15.6f}")
    print(f"{'Active Concepts (p>0.5)':<25} | {point_stats['avg_active']:<15.2f} | {set_stats['avg_active']:<15.2f}")
    print(f"{'Sparsity (%)':<25} | {(point_stats['avg_active']/C)*100:<15.2f}% | {(set_stats['avg_active']/C)*100:<15.2f}%")
    print("-" * 65)
    print("\nINTERPRETATION:")
    if point_stats['mse'] < set_stats['mse']:
        print("PASS Point-ConCA has lower MSE (expected: less compression needed).")
    else:
        print("FAIL Set-ConCA has lower MSE (unexpected).")
        
    print(f"Set-ConCA active concepts: {set_stats['avg_active']:.1f}")
    print(f"Point-ConCA active concepts: {point_stats['avg_active']:.1f}")
    
    print("\n>>> SUGGESTION: Run 'uv run python evaluate.py' with each model path")
    print(">>> to qualitatively compare concept coherence.")

import os
if __name__ == "__main__":
    main()

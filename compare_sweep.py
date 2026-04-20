import torch
import os
from setconca.model.setconca import SetConCA

def get_stats(model, data_path, alpha=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    loaded = torch.load(data_path, weights_only=False)
    hidden = loaded['hidden'] 
    N, S, D = hidden.shape
    
    # Simple batch evaluation
    total_mse = 0
    total_active = 0
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = hidden[i:i+batch_size].to(device)
            # Full pass
            u = model.encoder(batch)
            z_hat, u_bar, u_res = model.aggregator(u)
            f_hat = model.decoder(z_hat, u_res)
            
            # Reconstruction error (MSE)
            mse = ((f_hat - batch) ** 2).mean()
            total_mse += mse.item() * len(batch)
            
            # Active concepts (prob > 0.5)
            active = (torch.sigmoid(u_bar) > 0.5).float().sum(dim=1).mean()
            total_active += active.item() * len(batch)
            
    return {
        "mse": total_mse / N,
        "active": total_active / N
    }

def main():
    print("\n" + "="*50)
    print("GEMMA-2 NEIGHBORHOOD SWEEP RESULTS")
    print("="*50)
    
    results = []
    H, C = 2304, 4096
    
    for s in [1, 3, 8, 16,24,32]:
        model_path = f"data/model_gemma_s{s}.pt"
        data_path = f"data/gemma_s{s}_subset.pt"
        
        if not os.path.exists(model_path):
            continue
            
        model = SetConCA(H, C)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        stats = get_stats(model, data_path)
        stats['s'] = s
        results.append(stats)

    print(f"{'S Value':<10} | {'MSE (Fidelity)':<15} | {'Active Concepts':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['s']:<10} | {r['mse']:<15.6f} | {r['active']:<15.2f}")
    
    print("-" * 50)
    print("\nANALYSIS:")
    # Calculate gradients
    if len(results) >= 2:
        mse_increase = (results[-1]['mse'] - results[0]['mse']) / results[0]['mse'] * 100
        print(f"MSE increased by {mse_increase:.1f}% as S went from {results[0]['s']} to {results[-1]['s']}.")
        print("Interpretation: Higher S forces more compression, slightly degrading reconstruction.")
        
        print("\n>>> CONCLUSION: Is it worth going higher?")
        if results[-1]['active'] < results[0]['active']:
            print("YES. Higher S resulted in more selective (sparser) concepts, indicating better abstraction.")
        else:
            print("MAYBE. Higher S increased concept count, possibly capturing more nuanced distributional features.")
    
    print("\n>>> Qualitative check recommended for S=32 to verify 'Thematic Breadth'.")

if __name__ == "__main__":
    main()

import argparse
import torch
import numpy as np
from setconca.model.setconca import SetConCA

def main():
    parser = argparse.ArgumentParser(description="Evaluate Set-ConCA Sparse Concepts")
    parser.add_argument('--data_path', type=str, default="data/hf_real_dataset.pt")
    parser.add_argument('--model_path', type=str, default="data/setconca_model.pt")
    parser.add_argument('--hidden_dim', type=int, default=2560)
    parser.add_argument('--concept_dim', type=int, default=4096)
    parser.add_argument('--k_top_concepts', type=int, default=5, help="Number of concepts to inspect")
    parser.add_argument('--top_sets', type=int, default=3, help="Number of sets to print per concept")
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}")
    data_dict = torch.load(args.data_path, weights_only=False)
    
    if not isinstance(data_dict, dict) or "texts" not in data_dict:
        raise ValueError("Data file must be a dictionary containing 'hidden' and 'texts'. Re-run build_hf_dataset.py.")

    hidden = data_dict["hidden"]  # (N, S, D)
    texts = data_dict["texts"]    # list of N lists, each len S
    N, S, D = hidden.shape
    args.hidden_dim = D

    print(f"Loaded {N} representation sets (size={S}, dim={D})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SetConCA(hidden_dim=args.hidden_dim, concept_dim=args.concept_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print(f"Model loaded from {args.model_path}")

    # Extract all u_bar (pre-normalized log-posteriors)
    all_u_bars = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = hidden[i:i+batch_size].to(device)
            # encode_and_aggregate gives (z_hat, u_bar, u)
            _, u_bar, _ = model.encode_and_aggregate(batch)
            all_u_bars.append(u_bar.cpu())

    U = torch.cat(all_u_bars, dim=0) # (N, C)
    
    # Analyze sparsity map
    active_mask = (torch.sigmoid(U) > 0.5).float() # Threshold for active concepts
    avg_active_concepts = active_mask.sum(dim=1).mean().item()
    print(f"\nAverage active concepts per set: {avg_active_concepts:.2f} / {args.concept_dim}")

    # Find the concepts with the highest variance (most discriminating)
    vars = U.var(dim=0)
    top_concept_indices = vars.argsort(descending=True)[:args.k_top_concepts]

    print("\n" + "="*80)
    print("TOP CONCEPT VISUALIZATION")
    print("="*80)

    for rank, c_idx in enumerate(top_concept_indices):
        c_idx = c_idx.item()
        # Find which sets activate this concept the most
        concept_activations = U[:, c_idx]
        top_set_indices = concept_activations.argsort(descending=True)[:args.top_sets]
        
        print(f"\n[Rank {rank+1}] Concept Dimension: {c_idx}")
        print(f"Activation Variance: {vars[c_idx]:.4f}")
        print("-" * 50)
        
        for k, s_idx in enumerate(top_set_indices):
            s_idx = s_idx.item()
            activation_val = concept_activations[s_idx].item()
            probability_val = torch.sigmoid(torch.tensor(activation_val)).item()
            print(f"  ({k+1}) Set Index [{s_idx}] | Activation: {activation_val:.3f} | Probability: {probability_val:.3f}")
            # Print the first text of the set as a representative of the neighborhood
            # We truncate to 150 chars to keep it readable
            rep_text = texts[s_idx][0].replace('\n', ' ')
            if len(rep_text) > 150: rep_text = rep_text[:147] + "..."
            print(f"      Text: \"{rep_text}\"")

if __name__ == '__main__':
    main()

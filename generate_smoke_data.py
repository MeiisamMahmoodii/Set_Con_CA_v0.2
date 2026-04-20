import torch
import os

def generate_smoke_data(path="smoke_dataset.pt", n_sets=100, set_size=8, hidden_dim=64):
    """
    Generates a synthetic dataset that mimics the real data.
    Instead of pure noise, we create a base 'concept' per set and add variance,
    creating a more realistic testbed for the Consistency loss to work on.
    """
    print(f"Generating smoke dataset with {n_sets} sets...")
    
    # Base concept per set: shape (n_sets, 1, hidden_dim)
    base_concepts = torch.randn(n_sets, 1, hidden_dim) * 2.0
    
    # Element-specific variance (paraphrase noise): shape (n_sets, set_size, hidden_dim)
    instance_noise = torch.randn(n_sets, set_size, hidden_dim) * 0.5
    
    # Final data: sets of related representations
    data = base_concepts + instance_noise
    
    torch.save(data, path)
    print(f"Saved dataset of shape {data.shape} to {path}")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    generate_smoke_data("data/smoke_dataset.pt")

"""
grand_audit_eval.py — Automated comparison across all extracted models.
Tests: Gemma-2-2B, Gemma-2-9B, Llama-3-8B, Gemma-3-1B, Gemma-3-4B.
Metrics: MSE (Sigmoid), MSE (Top-K).
"""

import torch
import torch.optim as optim
from setconca.model.setconca import SetConCA, compute_loss
from setconca.data.dataset import RepresentationSetDataset, make_dataloader
import time

MODELS = [
    ("Gemma-2-2B", "data/gemma_sweep_dataset.pt"),
    ("Gemma-2-9B", "data/gemma_9b_dataset.pt"),
    ("Llama-3-8B", "data/llama_8b_dataset.pt"),
    ("Gemma-3-1B", "data/gemma3_1b_dataset.pt"),
    ("Gemma-3-4B", "data/gemma3_4b_dataset.pt"),
]

def run_test(name, path, use_topk=False, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing {name} (Top-K: {use_topk})...")
    
    loaded = torch.load(path, weights_only=False)
    data = loaded['hidden'] if isinstance(loaded, dict) else loaded
    dataset = RepresentationSetDataset(data)
    dataloader = make_dataloader(dataset, batch_size=32, shuffle=True)
    
    N, S, D = data.shape
    model = SetConCA(hidden_dim=D, concept_dim=4096, use_topk=use_topk, k=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    final_mse = 0
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, parts = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
        final_mse = parts['mse'].item()
        
    return final_mse, D

def main():
    results = []
    print("# Set-ConCA Grand Audit: Fidelity Comparison Table")
    print("| Model Family | Param Size | Features | MSE (Sigmoid) | MSE (Top-K, k=50) | Improvement |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for name, path in MODELS:
        # Run Sigmoid
        mse_sig, dim = run_test(name, path, use_topk=False, epochs=5)
        # Run Top-K
        mse_topk, _ = run_test(name, path, use_topk=True, epochs=5)
        
        impr = (mse_sig - mse_topk) / mse_sig * 100
        print(f"| {name} | {name.split('-')[-1]} | {dim} | {mse_sig:.4f} | {mse_topk:.4f} | {impr:.1f}% |")

if __name__ == "__main__":
    main()

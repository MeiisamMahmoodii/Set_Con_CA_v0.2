"""
extract_concepts.py — Phase 3: Dataset Concept Extraction
Extracts the Top-K concept latents (z) from saved hidden state datasets.
Reduces (N, S, D) hidden states to (N, C) concept vectors.
"""

import torch
from setconca.model.setconca import SetConCA
from tqdm import tqdm
import argparse
import os

def extract(data_path, model_path, output_path, k=50, device="cuda"):
    print(f"Extracting concepts from {data_path} using {model_path}...")
    
    # 1. Load Data
    loaded = torch.load(data_path, map_location="cpu", weights_only=False)
    hidden = loaded['hidden'] # (N, S, D)
    texts = loaded['texts']
    
    N, S, D = hidden.shape
    
    # 2. Load Model
    model = SetConCA(hidden_dim=D, concept_dim=4096, use_topk=True, k=k)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    
    all_z = []
    
    # 3. Process in batches to avoid VRAM overflow
    batch_size = 32
    for i in tqdm(range(0, N, batch_size), desc="Extracting"):
        batch = hidden[i:i+batch_size].to(device)
        with torch.no_grad():
            f_hat, z, u = model(batch)
            all_z.append(z.cpu())
            
    # 4. Save
    final_z = torch.cat(all_z, dim=0)
    torch.save({
        'z': final_z,
        'texts': texts,
        'config': {'k': k, 'model': model_path, 'source': data_path}
    }, output_path)
    print(f"Saved concepts to {output_path}. Shape: {final_z.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    extract(args.data_path, args.model_path, args.output_path, k=args.k)

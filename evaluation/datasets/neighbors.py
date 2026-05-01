import numpy as np
import torch

def build_anchor_neighbors(dataset_vectors_list, S=4):
    """
    Given a list of items:
    {
      "anchor_id": "...",
      "vector": [np.array]
    }
    Group by anchor ID, and ensure we have sets of size S (trim or pad with nearby data if needed).
    Returns a tensor of shape (N_anchors, S, D).
    """
    grouped = {}
    
    for item in dataset_vectors_list:
        grouped.setdefault(item["anchor_id"], []).append(item["vector"])
    
    valid_sets = []
    
    for anchor_id, vectors in grouped.items():
        if len(vectors) < S:
            # Duplicate locally to fulfill S if missing variations, although dataset should have exactly 4 variations here
            needed = S - len(vectors)
            extras = [vectors[i % len(vectors)] for i in range(needed)]
            vectors.extend(extras)
            
        vectors = vectors[:S] # restrict to S
        valid_sets.append(np.stack(vectors, axis=0))
        
    return torch.tensor(np.stack(valid_sets, axis=0)).float()

if __name__ == "__main__":
    # Test logic
    dummy = [
        {"anchor_id": "1", "vector": np.ones(10)},
        {"anchor_id": "1", "vector": np.ones(10)*2},
        {"anchor_id": "1", "vector": np.ones(10)*3},
        {"anchor_id": "1", "vector": np.ones(10)*4},
        {"anchor_id": "2", "vector": np.ones(10)},
        {"anchor_id": "2", "vector": np.ones(10)*2},
        {"anchor_id": "2", "vector": np.ones(10)*3},
        {"anchor_id": "2", "vector": np.ones(10)*4},
    ]
    tensor = build_anchor_neighbors(dummy, S=4)
    print("Neighborhood tensor shape:", tensor.shape)

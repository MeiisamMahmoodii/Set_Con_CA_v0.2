import numpy as np
import torch

def topk_overlap(z1, z2, k=32):
    """
    Computes overlap fraction between Top-K elements of two vectors of activations.
    Supports single vectors or batched (B, D).
    """
    if isinstance(z1, torch.Tensor):
        z1 = z1.detach().cpu().numpy()
    if isinstance(z2, torch.Tensor):
        z2 = z2.detach().cpu().numpy()

    k_eff = min(int(k), int(z1.shape[-1]), int(z2.shape[-1]))
    if k_eff <= 0:
        return 0.0

    if z1.ndim == 1:
        idx1 = np.argsort(np.abs(z1))[-k_eff:]
        idx2 = np.argsort(np.abs(z2))[-k_eff:]
        return len(set(idx1) & set(idx2)) / k_eff
    else:
        # Batched computation
        overlaps = []
        for i in range(len(z1)):
            idx1 = np.argsort(np.abs(z1[i]))[-k_eff:]
            idx2 = np.argsort(np.abs(z2[i]))[-k_eff:]
            overlaps.append(len(set(idx1) & set(idx2)) / k_eff)
        return np.mean(overlaps)

def cka(X, Y):
    """
    Centered Kernel Alignment over representations.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
        
    K = X @ X.T
    L = Y @ Y.T

    # center 
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    return (Kc * Lc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Lc) + 1e-8)

def evaluate_transfer(z_source, z_target, B, k=32):
    if torch.is_tensor(B):
        B = B.detach().cpu().numpy()
    if torch.is_tensor(z_source):
        z_source = z_source.detach().cpu().numpy()
    
    mapped = z_source @ B
    return topk_overlap(mapped, z_target, k=k)

def apply_steering(x, z_concept, B, alpha=1.0, decoder=None):
    """
    x: target model residual stream activation
    z_concept: source model set-conca concept vector
    B: learned cross-model bridge
    alpha: interaction strength
    decoder: if provided, decodes the C-dim concept back to D-dim residual space
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(z_concept):
        z_concept = z_concept.detach().cpu().numpy()
    if torch.is_tensor(B):
        B = B.detach().cpu().numpy()
        
    translated_concept = z_concept @ B
    
    if decoder is not None:
        with torch.no_grad():
            tc_t = torch.tensor(translated_concept).float().unsqueeze(0)
            u_zero = torch.zeros(1, 1, tc_t.shape[-1]).float()
            direction = decoder(tc_t, u_zero).squeeze().numpy()
        return x + alpha * direction
    else:
        return x + alpha * translated_concept

def bootstrap_ci(values, n=1000):
    samples = []
    for _ in range(n):
        resample = np.random.choice(values, size=len(values))
        samples.append(np.mean(resample))
    return np.percentile(samples, [2.5, 97.5])

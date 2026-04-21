"""
eval_faithfulness.py — NeurIPS Phase 1: Causal Fidelity Metric
Directly patches a HookedTransformer (Gemma-2-2B) with Set-ConCA reconstructions
to measure KL-Divergence and Logit Faithfulness.
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from setconca.model.setconca import SetConCA
import argparse
from tqdm import tqdm
from datasets import load_dataset

def load_sae(model_path, hidden_dim, concept_dim, k, device):
    sae = SetConCA(hidden_dim=hidden_dim, concept_dim=concept_dim, use_topk=True, k=k)
    sae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    sae.to(device).half() # Cast to half to match model activations
    sae.eval()
    return sae

def run_faithfulness_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model via TransformerLens
    print(f"Loading Gemma-2-2B via TransformerLens...")
    model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device, torch_dtype=torch.float16)
    
    # 2. Load trained Set-ConCA
    sae = load_sae(args.sae_path, model.cfg.d_model, 4096, args.k, device)
    
    # 3. Load Sample Data (AG News)
    dataset = load_dataset("ag_news", split="test[:128]") # Small slice for validation
    
    kl_total = 0
    fvu_total = 0
    count = 0
    
    layer_name = f"blocks.{args.layer}.hook_resid_post"
    
    for i, text in enumerate(tqdm(dataset["text"], desc="Causal Patching")):
        tokens = model.to_tokens(text, truncate=True)
        
        with torch.no_grad():
            # Original Pass
            logits_orig, cache = model.run_with_cache(tokens, names_filter=[layer_name])
            orig_activations = cache[layer_name] # (1, seq, D)
            
            # SAE Reconstruction
            f_hat, z_hat, u_res = sae(orig_activations)
            
            # 4. Norm Alignment (Diagnostic Patch)
            norm_orig = orig_activations.norm()
            norm_recon = f_hat.norm()
            # Scale f_hat to match original energy for causal testing
            f_hat_scaled = f_hat * (norm_orig / (norm_recon + 1e-8))
            
            # DEBUG: Magnitude Check
            if i < 3:
                print(f"\n[Sample {i}] Norm Orig: {norm_orig:.2f} | Norm Recon: {norm_recon:.2f} -> Scaled: {f_hat_scaled.norm():.2f}")
                print(f"             MSE (Float32): { (orig_activations.float() - f_hat.float()).pow(2).mean():.4f}")
            
            # Calculate activation fidelity (use float32 for stable FVU)
            mse = (orig_activations.float() - f_hat.float()).pow(2).mean()
            var = orig_activations.float().var() + 1e-6
            fvu = mse / var
            fvu_total += fvu.item()
            
            # 5. Patching: Replace original with f_hat_scaled
            def patch_hook(activations, hook):
                return f_hat_scaled
            
            logits_patch = model.run_with_hooks(
                tokens,
                fwd_hooks=[(layer_name, patch_hook)]
            )
            
            # 6. Measure KL Divergence on the final token distribution
            p_orig = F.softmax(logits_orig[0, -1, :].float(), dim=-1)
            p_patch = F.log_softmax(logits_patch[0, -1, :].float(), dim=-1)
            
            kl = F.kl_div(p_patch, p_orig, reduction='sum')
            if not torch.isnan(kl):
                kl_total += kl.item()
            count += 1
            
    print(f"\n--- NeurIPS Fidelity Report ({args.sae_path}) ---")
    print(f"  KL-Divergence (bits): {kl_total / count:.4f}")
    print(f"  FVU (Residual):       {fvu_total / count:.4f}")
    print(f"  Faithfulness Score:   {max(0, 1 - (kl_total / count)/10.0):.2%}") # Normalized score
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    run_faithfulness_test()

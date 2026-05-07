import torch
import json
import sys
import os
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Project root and this package on path (run from repo root or from evaluation/).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPT_DIR)

from dataset_utils.dataset_builder import build_dataset, SEED_DATA
from runner.exp1_set_vs_pointwise import run_exp1
from runner.exp2_set_size_scaling import run_exp2
from runner.exp3_aggregator_ablation import run_exp3
from runner.exp4_cross_model_transfer import run_exp4
from runner.exp5_interventional_steering import run_exp5

from dataset_utils.extract_activations import ActivationExtractor
from dataset_utils.neighbors import build_anchor_neighbors

def extract_real_activations(dataset, model_name, layer, S=32):
    """
    Extracts activations using the exact target LLM layer and groups them into 
    local neighborhoods of size S.
    """
    print(f"\n--> Extracting manuscript representations from {model_name} (Layer {layer})")
    # Initialize extractor which automatically handles device allocation
    extractor = ActivationExtractor(model_name, layer=layer)
    results = []
    
    total = len(dataset)
    for i, item in enumerate(dataset):
        print(f"    Extracting anchor {i+1}/{total}... \r", end="")
        # Iterate through the conceptually varying string instances 
        for prompt_key, prompt_text in item["variations"].items():
            vec = extractor.extract(prompt_text)
            results.append({
                "anchor_id": item["anchor_id"],
                "model": model_name,
                "vector": vec
            })
    print("\nExtraction complete.")
    
    # Clean up massive tensor buffers out of VRAM before loading the next transformer
    del extractor.model
    del extractor
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Group the vectors into exactly dimensioned neighborhoods for Set-ConCA
    return build_anchor_neighbors(results, S=S)

if __name__ == "__main__":
    print("="*60)
    print("NeurIPS Set-ConCA Evaluation Suite Runner")
    print("="*60)
    
    # 1. Dataset Generation 
    print("\n[PHASE 1: Dataset Validation]")
    ds = build_dataset(SEED_DATA)
    print(f"Generated structured dataset format: {len(ds)} anchors, 4 variations each.")
    
    # 2. Extract Data via HuggingFace
    print("\n[PHASE 2: Real Data Extraction]")
    source_model_str = "google/gemma-2-2b"
    target_model_str = "meta-llama/Meta-Llama-3-8B"
    
    source_data = extract_real_activations(ds, source_model_str, layer=20, S=32)
    target_data = extract_real_activations(ds, target_model_str, layer=20, S=32)
    
    print("\n[PHASE 3: Running Experiments]")
    # For speed of testing S=8 for main experiments, we logically cap S=8 where appropriate
    source_s8 = source_data[:, :8, :]
    target_s8 = target_data[:, :8, :]
    
    print("\n[PHASE 3: Running Experiments]")
    
    print("\n" + "="*40)
    res1 = run_exp1(source_s8)
    
    print("\n" + "="*40)
    res2 = run_exp2(source_data, sweep=[1, 3, 8, 16, 32])
    
    print("\n" + "="*40)
    res3 = run_exp3(source_s8)
    
    print("\n" + "="*40)
    res4 = run_exp4(source_s8, target_s8)
    
    print("\n" + "="*40)
    res5 = run_exp5(source_s8, target_s8, concept_idx=2)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")
    print("Export these results to LaTeX manuscript.")

import os
import torch
import subprocess
import time

def run_training(name, data_path, epochs, s_val, alpha=0.1, beta=0.01):
    save_path = f"data/model_gemma_s{s_val}.pt"
    cmd = [
        "uv", "run", "python", "train.py",
        "--data_path", data_path,
        "--epochs", str(epochs),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--save_path", save_path,
        "--hidden_dim", "2304",  # Gemma-2-2B hidden size
        "--concept_dim", "4096",
        "--set_size", str(s_val)
    ]
    print(f"\n>>> Running sweep for S={s_val}")
    print(f">>> Hyperparams: epochs={epochs}, alpha={alpha}, beta={beta}")
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print(f">>> S={s_val} sweep complete in {end - start:.2f} seconds.")
    return save_path

def main():
    master_data_path = "data/gemma_sweep_dataset.pt"
    
    if not os.path.exists(master_data_path):
        print(f"Error: {master_data_path} not found. Run build_hf_dataset.py first.")
        return

    print(f"Loading Master Dataset from {master_data_path}...")
    master_loaded = torch.load(master_data_path, weights_only=False)
    hidden = master_loaded['hidden']  # (N, S_master=32, D)
    texts = master_loaded['texts']    # N lists of 32 strings
    
    N, S_master, D = hidden.shape
    
    sweep_configs = [
        {"s": 1, "epochs": 240}, # Point Baseline
        {"s": 3, "epochs": 80},  # Low-S Test
        {"s": 8, "epochs": 30},
        {"s": 16, "epochs": 15},
        {"s": 32, "epochs": 8}
    ]

    for config in sweep_configs:
        s = config["s"]
        epochs = config["epochs"]
        
        # 1. Slice the master data
        print(f"\nSlicing data for S={s}...")
        sliced_hidden = hidden[:, :s, :]
        sliced_texts = [t[:s] for t in texts]
        
        suffix_path = f"data/gemma_s{s}_subset.pt"
        torch.save({"hidden": sliced_hidden, "texts": sliced_texts}, suffix_path)
        
        # 2. Run training
        run_training(f"g{s}", suffix_path, epochs, s)

    print("\n" + "="*50)
    print("Gemma Scaling Sweep Complete")
    print("Models saved in: data/model_gemma_s{8,16,32}.pt")
    print("="*50)

if __name__ == "__main__":
    main()

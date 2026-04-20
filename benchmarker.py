import os
import torch
import subprocess
import time

def run_training(name, data_path, epochs, alpha=0.1, beta=0.01, set_size=8):
    save_path = f"data/model_{name}.pt"
    cmd = [
        "uv", "run", "python", "train.py",
        "--data_path", data_path,
        "--epochs", str(epochs),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--save_path", save_path,
        "--hidden_dim", "2560",
        "--concept_dim", "4096",
        "--set_size", str(set_size)
    ]
    print(f"\n>>> Running benchmarking for: {name.upper()}")
    print(f">>> Hyperparams: epochs={epochs}, alpha={alpha}, beta={beta}, set_size={set_size}")
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print(f">>> {name.upper()} training complete in {end - start:.2f} seconds.")
    return save_path

def main():
    original_data_path = "data/hf_real_dataset.pt"
    point_data_path = "data/hf_point_dataset.pt"
    
    if not os.path.exists(original_data_path):
        print(f"Error: {original_data_path} not found. Run build_hf_dataset.py first.")
        return

    # 1. Prepare Point Dataset
    print(f"Preparing Point Dataset from {original_data_path}...")
    loaded = torch.load(original_data_path, weights_only=False)
    if isinstance(loaded, dict):
        hidden = loaded['hidden']  # (N, S, D)
        texts = loaded['texts']    # N sets of S strings
    else:
        hidden = loaded
        texts = None
    
    N, S, D = hidden.shape
    # Flatten N*S into new N
    point_hidden = hidden.view(N * S, 1, D)
    
    if texts:
        # Flattened list of strings
        point_texts = [[t] for text_set in texts for t in text_set]
        point_save_dict = {
            "hidden": point_hidden,
            "texts": point_texts
        }
        torch.save(point_save_dict, point_data_path)
    else:
        torch.save(point_hidden, point_data_path)
    
    print(f"Point data saved to {point_data_path}. Total points: {N*S}")

    # 2. Run Set-ConCA (S=8) - 30 epochs
    set_epochs = 30
    run_training("set", original_data_path, epochs=set_epochs, set_size=8)

    # 3. Run Point-ConCA (S=1) - Keep total samples seen constant.
    # Set run seen N * S * set_epochs = 2048 * 8 * 30 = 491,520 samples.
    # Point run seen (N*S) * point_epochs = 16384 * point_epochs = 491,520.
    # point_epochs = 30 / 8 = 3.75. Let's do 4 epochs for Point.
    point_epochs = 4
    run_training("point", point_data_path, epochs=point_epochs, set_size=1, beta=0.0) # Beta=0 for points

    print("\n" + "="*50)
    print("Ablation Run Complete")
    print("Models saved in: data/model_set.pt and data/model_point.pt")
    print("="*50)

if __name__ == "__main__":
    main()

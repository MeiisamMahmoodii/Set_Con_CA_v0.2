"""
train.py – Set-ConCA training script
=====================================
Flags
-----
Data
  --data_path   PATH     Path to a .pt file: raw tensor (N,S,D) or dict with 'hidden' key.
                         Omit to train on synthetic Gaussian data.
  --n_sets      INT      Number of synthetic sets (ignored when --data_path is given). [1024]
  --set_size    INT      Set size S for synthetic data. [8]
  --hidden_dim  INT      Hidden dimension D for synthetic data. [64]

Architecture
  --concept_dim INT      Concept bottleneck dimension C. [128]
  --agg_mode    STR      Set aggregator: 'mean' | 'attention'. [mean]
  --use_topk           Use hard Top-K sparsity instead of Sigmoid-L1.
  --k           INT      k for Top-K activation. [32]

Loss
  --alpha       FLOAT    Sparsity loss coefficient. [0.1]
  --beta        FLOAT    Consistency loss coefficient. [0.01]

Training
  --epochs      INT      Number of training epochs. [100]
  --batch_size  INT      Batch size. [32]
  --lr          FLOAT    Learning rate. [2e-4]
  --seed        INT      Random seed. [0]

I/O
  --save_path   PATH     Where to save the trained model. [checkpoints/model.pt]
"""

import argparse
import os
import torch
import torch.optim as optim

from setconca.model.setconca import SetConCA, compute_loss
from setconca.data.dataset import RepresentationSetDataset, make_synthetic_dataset, make_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Train Set-ConCA", formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    p.add_argument("--data_path",   type=str,   default=None,              help="Path to .pt dataset")
    p.add_argument("--n_sets",      type=int,   default=1024,              help="Synthetic set count")
    p.add_argument("--set_size",    type=int,   default=8,                 help="Set size S (synthetic)")
    p.add_argument("--hidden_dim",  type=int,   default=64,                help="Hidden dim D (synthetic)")

    # Architecture
    p.add_argument("--concept_dim", type=int,   default=128,               help="Concept dim C")
    p.add_argument("--agg_mode",    type=str,   default="mean",
                   choices=["mean", "attention"],                           help="Aggregator mode")
    p.add_argument("--use_topk",    action="store_true",                   help="Hard Top-K sparsity")
    p.add_argument("--k",           type=int,   default=32,                help="k for Top-K")

    # Loss
    p.add_argument("--alpha",       type=float, default=0.1,               help="Sparsity coefficient")
    p.add_argument("--beta",        type=float, default=0.01,              help="Consistency coefficient")

    # Training
    p.add_argument("--epochs",      type=int,   default=100,               help="Training epochs")
    p.add_argument("--batch_size",  type=int,   default=32,                help="Batch size")
    p.add_argument("--lr",          type=float, default=2e-4,              help="Learning rate")
    p.add_argument("--seed",        type=int,   default=0,                 help="Random seed")

    # I/O
    p.add_argument("--save_path",   type=str,   default="checkpoints/model.pt", help="Output path")

    return p.parse_args()


def load_data(args):
    if args.data_path:
        print(f"Loading data from {args.data_path} ...")
        raw = torch.load(args.data_path, weights_only=False)
        tensor = raw["hidden"] if isinstance(raw, dict) else raw
        dataset = RepresentationSetDataset(tensor)
        N, S, D = tensor.shape
        args.hidden_dim, args.set_size = D, S
        print(f"  {N} sets  |  S={S}  |  D={D}")
    else:
        print(f"Generating synthetic data: {args.n_sets} sets, S={args.set_size}, D={args.hidden_dim}")
        dataset = make_synthetic_dataset(args.n_sets, args.set_size, args.hidden_dim)
    return dataset


def build_model(args, device):
    model = SetConCA(
        hidden_dim=args.hidden_dim,
        concept_dim=args.concept_dim,
        use_topk=args.use_topk,
        k=args.k,
    )
    if args.agg_mode == "attention":
        from setconca.model.aggregator import AttentionAggregator
        model.aggregator.mode = "attention"
        model.aggregator.att = AttentionAggregator(args.concept_dim)
    return model.to(device)


def train():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_data(args)
    loader  = make_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    model   = build_model(args, device)
    opt     = optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"agg={args.agg_mode}  topk={args.use_topk}  k={args.k}  "
          f"alpha={args.alpha}  beta={args.beta}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"total": 0.0, "mse": 0.0, "sparsity": 0.0, "consistency": 0.0}

        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss, parts = compute_loss(model, batch, alpha=args.alpha, beta=args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            totals["total"]       += loss.item()
            totals["mse"]         += parts["mse"].item()
            totals["sparsity"]    += parts["sparsity"].item()
            totals["consistency"] += parts["consistency"].item()

        n = len(loader)
        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"total={totals['total']/n:.4f}  "
            f"mse={totals['mse']/n:.4f}  "
            f"spar={totals['sparsity']/n:.6f}  "
            f"cons={totals['consistency']/n:.4f}"
        )

    print("-" * 60)
    print("Training complete.")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved → {args.save_path}")


if __name__ == "__main__":
    train()

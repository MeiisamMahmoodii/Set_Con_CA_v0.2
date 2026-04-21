import argparse
import torch
import torch.optim as optim
import wandb
from setconca.model.setconca import SetConCA, compute_loss
from setconca.data.dataset import make_synthetic_dataset, make_dataloader

def train():
    parser = argparse.ArgumentParser(description="Train Set-ConCA")
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--concept_dim', type=int, default=128)
    parser.add_argument('--set_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Sparsity coefficient. 1e-3 is too weak vs MSE; 0.1 is balanced.')
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_sets', type=int, default=1024)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default="data/setconca_model.pt")
    
    # NEW: Architectural options for MSE improvement
    parser.add_argument('--use_topk', action='store_true', help='Use Top-K sparsity instead of Sigmoid')
    parser.add_argument('--k', type=int, default=32, help='k for Top-K activation')
    parser.add_argument('--agg_mode', type=str, default='mean', choices=['mean', 'attention'], help='Set aggregation method')
    
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="setconca", config=vars(args), mode="disabled")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    if args.data_path:
        print(f"Loading data from {args.data_path}...")
        from setconca.data.dataset import RepresentationSetDataset
        
        # Load file, handle both raw tensors and generic dicts containing 'hidden' and 'texts'
        loaded = torch.load(args.data_path, weights_only=False)
        if isinstance(loaded, dict):
            data_tensor = loaded['hidden']
        else:
            data_tensor = loaded
            
        dataset = RepresentationSetDataset(data_tensor)
        
        # Override dimensions from data to prevent crashes
        N, S, D = data_tensor.shape
        args.hidden_dim = D
        args.set_size = S
        print(f"Dataset loaded. Total sets: {N}, Set size: {S}, Hidden dim: {D}")
    else:
        dataset = make_synthetic_dataset(n_sets=args.n_sets, set_size=args.set_size, hidden_dim=args.hidden_dim)
        
    dataloader = make_dataloader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = SetConCA(
        hidden_dim=args.hidden_dim, 
        concept_dim=args.concept_dim,
        use_topk=args.use_topk,
        k=args.k
    ).to(device)
    
    # Update aggregator mode if not default
    if args.agg_mode != 'mean':
        model.aggregator.mode = args.agg_mode
        if args.agg_mode == 'attention':
            from setconca.model.aggregator import AttentionAggregator
            model.aggregator.att = AttentionAggregator(args.concept_dim).to(device)
            
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_spar = 0
        total_cons = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            loss, parts = compute_loss(model, batch, alpha=args.alpha, beta=args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevents gradient explosion
            optimizer.step()

            total_loss += loss.item()
            total_mse += parts['mse'].item()
            total_spar += parts['sparsity'].item()
            total_cons += parts['consistency'].item()

        n_batches = len(dataloader)
        metrics = {
            'epoch': epoch + 1,
            'loss/total': total_loss / n_batches,
            'loss/mse': total_mse / n_batches,
            'loss/sparsity': total_spar / n_batches,
            'loss/consistency': total_cons / n_batches,
        }
        wandb.log(metrics)
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Total: {metrics['loss/total']:.4f} "
              f"MSE: {metrics['loss/mse']:.4f} "
              f"Spar: {metrics['loss/sparsity']:.6f} "
              f"Cons: {metrics['loss/consistency']:.4f}")

    print("Training complete.")
    
    if args.save_path:
        import os
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

    wandb.finish()

if __name__ == '__main__':
    train()

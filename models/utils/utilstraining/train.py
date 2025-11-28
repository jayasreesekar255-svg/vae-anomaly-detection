"""
Train a VAE on the synthetic dataset.

Saves:
- model checkpoint to <out_path>
- recon_errors.npz containing per-sample reconstruction errors and labels
"""
import argparse
import os
from typing import Tuple
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from models.vae import VAE, loss_function
from utils.dataloaders import load_dataset_npz
from utils.metrics import evaluate_scores

def train_vae(
    data_path: str = "dataset.npz",
    latent_dim: int = 8,
    beta: float = 1.0,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "cpu",
    out_path: str = "vae_checkpoint.pt",
) -> Tuple[str, dict]:
    loader, X, y = load_dataset_npz(data_path, batch_size=batch_size, shuffle=True)
    input_dim = X.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = loss_function(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / X.shape[0]
        print(f"Epoch {epoch}/{epochs} | avg_loss: {avg_loss:.6f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "latent_dim": latent_dim}, out_path)
    print(f"Saved model to {out_path}")

    # Compute reconstruction errors for all samples
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        recon, mu, logvar = model(X_tensor)
        recon_np = recon.cpu().numpy()
        recon_errors = np.mean((recon_np - X) ** 2, axis=1)  # per-sample MSE

    np.savez_compressed("recon_errors.npz", recon_errors=recon_errors.astype(np.float32), y=y.astype(np.int32))
    print("Saved recon_errors.npz")

    # Evaluate scores
    metrics = evaluate_scores(y, recon_errors)
    print("Evaluation metrics:", metrics)
    return out_path, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset.npz")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-path", type=str, default="models/vae_beta.pt")
    args = parser.parse_args()

    train_vae(
        data_path=args.data_path,
        latent_dim=args.latent_dim,
        beta=args.beta,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        out_path=args.out_path,
  )

"""
Run training for multiple beta values and collect evaluation metrics.

Produces a CSV summary 'beta_sweep_results.csv' and saves recon_errors for the last run.
"""
import argparse
import csv
from training.train import train_vae
import os

def sweep_betas(betas, epochs=30, latent_dim=8, device="cpu", out_dir="sweep_results"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "beta_sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beta", "AUC_ROC", "AUPR", "P@k"])

        for beta in betas:
            out_path = os.path.join(out_dir, f"vae_beta_{beta:.3f}.pt")
            print(f"\n--- Training beta={beta} ---")
            _, metrics = train_vae(
                data_path="dataset.npz",
                latent_dim=latent_dim,
                beta=beta,
                epochs=epochs,
                lr=1e-3,
                batch_size=128,
                device=device,
                out_path=out_path,
            )
            writer.writerow([beta, metrics["AUC_ROC"], metrics["AUPR"], metrics["P@k"]])
    print(f"Sweep finished. Results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    sweep_betas(args.betas, epochs=args.epochs, latent_dim=args.latent_dim, device=args.device)

"""
Run the full pipeline:
1. Generate dataset
2. Sweep beta values (train + evaluate)
3. Plot last run's recon errors (optional)

Adjust parameters in the script or via command line.
"""
import os
from data.generate_dataset import generate_synthetic_data
from training.sweep_beta import sweep_betas
from evaluation.evaluate import plot_recon_errors

def run_pipeline():
    # 1. Generate dataset (saves dataset.npz)
    generate_synthetic_data()

    # 2. Sweep betas and train models
    betas = [0.1, 0.5, 1.0]
    sweep_betas(betas, epochs=25, latent_dim=8, device="cpu", out_dir="sweep_results")

    # 3. Visualize reconstruction errors from last run
    if os.path.exists("recon_errors.npz"):
        plot_recon_errors("recon_errors.npz")
    else:
        print("No recon_errors.npz found — run training first.")

if __name__ == "__main__":
    run_pipeline()

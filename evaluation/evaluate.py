"""
Utility to load recon_errors.npz and display evaluation metrics and simple plots.

Produces:
- Prints AUC/AUPR/P@k
- Shows histogram of reconstruction errors and labels overlay (requires matplotlib)
"""
import numpy as np
from utils.metrics import evaluate_scores
import matplotlib.pyplot as plt
import seaborn as sns

def plot_recon_errors(path="recon_errors.npz"):
    data = np.load(path)
    errors = data["recon_errors"]
    y = data["y"]

    metrics = evaluate_scores(y, errors)
    print("Metrics:", metrics)

    plt.figure(figsize=(8, 5))
    sns.histplot(errors[y == 0], label="normal", element="step", stat="density")
    sns.histplot(errors[y == 1], label="anomaly", element="step", stat="density")
    plt.legend()
    plt.xlabel("Reconstruction error (MSE)")
    plt.title("Reconstruction error distribution")
    plt.show()

if __name__ == "__main__":
    plot_recon_errors()

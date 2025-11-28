# Variational Autoencoder (VAE) for Anomaly Detection

This repository implements a VAE tailored for anomaly detection in high-dimensional
synthetic data. It includes dataset generation, a configurable PyTorch VAE, training,
hyperparameter sweep over β (beta-VAE), anomaly scoring based on reconstruction error,
and evaluation metrics (AUC, Precision/Recall).

## Quick start

1. Install dependencies:
```bash
pip install -r requirements.txt

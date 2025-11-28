"""
Generate a synthetic high-dimensional dataset with latent structure and injected anomalies.

Saves dataset.npz containing:
- X: float32 array shape (n_samples, n_features)
- y: int array shape (n_samples,) where 1 indicates anomaly
- meta: dictionary with generation params
"""
import numpy as np
from typing import Tuple
import os

def generate_synthetic_data(
    n_samples: int = 5000,
    n_features: int = 16,
    latent_dim: int = 4,
    anomaly_frac: float = 0.03,
    random_seed: int = 42,
    output_path: str = "dataset.npz",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dataset with latent structure: X = Z @ W + nonlinear + noise.
    Inject anomalies by replacing a small fraction of points with out-of-distribution samples.

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    y : np.ndarray
        Binary labels (1=anomaly, 0=normal)
    """
    rng = np.random.default_rng(random_seed)
    # Latent factors (structured)
    Z = rng.normal(0, 1.0, size=(n_samples, latent_dim))
    W = rng.normal(0, 1.0, size=(latent_dim, n_features)) * 0.8

    # Nonlinear transform + noise
    X_clean = Z @ W
    X_clean += 0.2 * np.sin(Z @ (W * 0.5))  # mild nonlinearity
    X_clean += rng.normal(0, 0.05, size=X_clean.shape)

    # Optionally augment some features with random walk-style signals to increase complexity
    for j in range(n_features // 4):
        rw = np.cumsum(rng.normal(0, 0.01, size=n_samples))
        X_clean[:, j] += rw

    # Inject anomalies
    n_anom = max(1, int(n_samples * anomaly_frac))
    y = np.zeros(n_samples, dtype=int)
    anom_idx = rng.choice(n_samples, size=n_anom, replace=False)
    y[anom_idx] = 1

    # Create anomalies by adding large offsets or sampling from different distribution
    X = X_clean.copy()
    for idx in anom_idx:
        if rng.random() < 0.5:
            # large offset on a subset of features
            X[idx, rng.choice(n_features, size=max(1, n_features // 5), replace=False)] += rng.normal(5, 1.5, size=(max(1, n_features // 5),))
        else:
            # out-of-manifold sample
            X[idx] = rng.normal(3.0, 2.0, size=(n_features,))

    # Normalize features (store mean/std for potential inverse transform)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-9
    X = (X - mean) / std

    # Save dataset
    np.savez_compressed(output_path, X=X.astype(np.float32), y=y.astype(np.int32), mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Saved dataset to {output_path}. n_samples={n_samples}, n_features={n_features}, n_anom={n_anom}")
    return X, y

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    # Save in repo root for convenience
    generate_synthetic_data(output_path="dataset.npz")

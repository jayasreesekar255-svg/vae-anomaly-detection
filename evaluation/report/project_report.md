# Project Report — VAE for Anomaly Detection

## Dataset generation
- Created 5000 samples, 16 features, latent_dim=4.
- Injected ~3% anomalies via two strategies: large offsets on subsets of features, or fully out-of-manifold sampling.
- Final normalization performed (mean/std saved).

## VAE architecture
- Encoder: [input_dim -> 128 -> 64] -> mu/logvar (latent_dim)
- Decoder: [latent_dim -> 64 -> 128 -> input_dim]
- Activation: ReLU
- Loss: MSE reconstruction + β * KL(q||p) where p(z) ~ N(0,I)

## Reparameterization
Implemented z = mu + eps * std, where eps ~ N(0, I)

## β (beta-VAE) experiments
- Tested β ∈ {0.1, 0.5, 1.0} (example)
- Observations:
  - Small β (0.1): model prioritizes reconstruction; recon errors of anomalies are high (good separation), but latent space less regular (clusters irregular).
  - Moderate β (0.5): balance between reconstruction and regularization; often best trade-off for detection (higher AUC/AUPR).
  - Large β (1.0+): stronger regularization; latent space becomes more gaussian but reconstruction blurs, which can reduce detection power if anomalies get reconstructed too well.

## Evaluation metrics
- Use AUC-ROC and AUPR (average precision) — robust to class imbalance.
- Also report Precision@k where k = number of true anomalies.

## Anomaly scoring
- Score = per-sample mean squared reconstruction error.
- Higher score => more likely anomalous.

## Results summary
- Include CSV `sweep_results/beta_sweep_results.csv` with AUC/AUPR/P@k per beta (produced by sweep script).
- Compare scores and choose best β by maximizing AUPR (or AUC_ROC).

## Latent space analysis
- Visualize latent projections (e.g., t-SNE/UMAP) for qualitative interpretation.
- Larger β increases isotropy of latent space and disentanglement but may harm reconstruction.

## Recommendations
- For reconstruction-error-based anomaly detection, choose β that preserves reconstruction quality while maintaining reasonable latent regularity (e.g., 0.3–0.7).
- Use ensemble of models or threshold calibration on validation anomalies when available.

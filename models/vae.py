"""
PyTorch implementation of a Variational Autoencoder (VAE) for tabular data.

Includes:
- Encoder producing mu and logvar
- Reparameterization trick
- Decoder reconstructing input
- Loss: reconstruction (MSE) + beta * KL
"""
from typing import Tuple, Optional
import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dims: Optional[list] = None):
        """
        Parameters
        ----------
        input_dim: int
            Number of input features
        latent_dim: int
            Dimensionality of latent space
        hidden_dims: Optional[List[int]]
            Sizes of hidden layers for encoder/decoder
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        hidden_dims.reverse()
        decoder_layers = []
        prev_dim = latent_dim
        for h in hidden_dims:
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns total loss, reconstruction loss (MSE), and KL divergence term.
    """
    # Reconstruction: mean squared error per sample averaged across batch
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    # KL divergence between q(z|x) and p(z) (standard normal)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kld
    return total, recon_loss.detach(), kld.detach()

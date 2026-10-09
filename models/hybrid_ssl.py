"""Hybrid Self-Supervised Learning Model: Autoencoder + Contrastive Regularization.
Combines global continuum preservation (reconstruction) with local line robustness (contrastive).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from models.autoencoder import SpectrumAutoencoder1D, weighted_mse_loss
from models.contrastive_model import NTXentLoss


class HybridSpectralSSL(nn.Module):
    """
    Hybrid SSL architecture combining 1D Autoencoder with Contrastive Loss.
    Encoder maps x -> latent z.
    Decoder maps z -> reconstructed x.
    Projector maps z -> contrastive projection p.
    """

    def __init__(self, latent_dim: int = 64, proj_dim: int = 64, seq_len: int = 4096):
        super().__init__()
        self.autoencoder = SpectrumAutoencoder1D(latent_dim=latent_dim, seq_len=seq_len)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, proj_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns latent representation z."""
        return self.autoencoder.encode(x)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Projects latent z to contrastive hypersphere."""
        p = self.projector(z)
        return F.normalize(p, dim=-1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns recon1, recon2, p1, p2.
        """
        recon1, z1 = self.autoencoder(x1)
        recon2, z2 = self.autoencoder(x2)
        p1 = self.project(z1)
        p2 = self.project(z2)
        return recon1, recon2, p1, p2

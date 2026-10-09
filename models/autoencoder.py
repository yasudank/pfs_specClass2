"""1D Convolutional Autoencoder with Weighted/Masked MSE Loss for Spectra."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class DeconvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale_factor: int = 2, kernel_size: int = 7):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False)
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class SpectrumAutoencoder1D(nn.Module):
    """
    1D Convolutional Autoencoder for 4096-length spectra.
    Maps (B, 1, 4096) -> latent (B, latent_dim) -> (B, 1, 4096).
    """

    def __init__(self, latent_dim: int = 32, seq_len: int = 4096):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # Encoder: 4096 -> 2048 -> 1024 -> 512 -> 256 -> 128
        self.enc1 = ConvBlock1D(1, 32, stride=2, kernel_size=15)    # -> 2048
        self.enc2 = ConvBlock1D(32, 64, stride=2, kernel_size=7)    # -> 1024
        self.enc3 = ConvBlock1D(64, 128, stride=2, kernel_size=7)   # -> 512
        self.enc4 = ConvBlock1D(128, 256, stride=2, kernel_size=7)  # -> 256
        self.enc5 = ConvBlock1D(256, 256, stride=2, kernel_size=7)  # -> 128

        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),  # 256 * 16 = 4096
            nn.Flatten(),
            nn.Linear(256 * 16, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )

        # Decoder: latent -> 256 * 128 -> ... -> 4096
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256 * 128),
            nn.GELU()
        )

        self.dec5 = DeconvBlock1D(256, 256, scale_factor=2)  # 128 -> 256
        self.dec4 = DeconvBlock1D(256, 128, scale_factor=2)  # 256 -> 512
        self.dec3 = DeconvBlock1D(128, 64, scale_factor=2)   # 512 -> 1024
        self.dec2 = DeconvBlock1D(64, 32, scale_factor=2)    # 1024 -> 2048
        self.dec1 = DeconvBlock1D(32, 16, scale_factor=2)    # 2048 -> 4096

        self.final_conv = nn.Conv1d(16, 1, kernel_size=7, stride=1, padding=3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 4096) -> z: (B, latent_dim)"""
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) -> recon: (B, 1, 4096)"""
        h = self.from_latent(z).view(-1, 256, 128)
        h = self.dec5(h)
        h = self.dec4(h)
        h = self.dec3(h)
        h = self.dec2(h)
        h = self.dec1(h)
        out = self.final_conv(h)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns recon (B, 1, 4096) and z (B, latent_dim)."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def weighted_mse_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes weighted MSE loss ignoring masked pixels (weight == 0).
    Args:
        recon: (B, 1, N) or (B, N)
        target: (B, 1, N) or (B, N)
        weight: (B, 1, N) or (B, N)
    """
    diff_sq = (recon - target) ** 2
    weighted_diff = weight * diff_sq
    per_sample_loss = torch.sum(weighted_diff, dim=-1) / (torch.sum(weight, dim=-1) + eps)
    return torch.mean(per_sample_loss)


def compute_anomaly_scores(
    recon: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """Computes per-sample weighted reconstruction error (anomaly score)."""
    with torch.no_grad():
        diff_sq = (recon - target) ** 2
        weighted_diff = weight * diff_sq
        scores = torch.sum(weighted_diff, dim=-1) / (torch.sum(weight, dim=-1) + eps)
        if scores.ndim > 1:
            scores = scores.squeeze(1)
        return scores

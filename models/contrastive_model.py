"""1D-ResNet Backbone and Contrastive Learning (SimCLR) for Spectra."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + res)
        return out


class ResNet1DBackbone(nn.Module):
    """
    1D-ResNet for spectral feature extraction.
    Input: (B, 1, 4096) -> Output embedding: (B, embed_dim)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),  # -> 2048
            nn.BatchNorm1d(32),
            nn.GELU()
        )

        # 4 stages of residual blocks
        self.stage1 = nn.Sequential(
            ResidualBlock1D(32, 32, stride=1),
            ResidualBlock1D(32, 32, stride=1)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(32, 64, stride=2),  # -> 1024
            ResidualBlock1D(64, 64, stride=1)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2), # -> 512
            ResidualBlock1D(128, 128, stride=1)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock1D(128, 256, stride=2), # -> 256
            ResidualBlock1D(256, 256, stride=1)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.pool(h).flatten(1)
        embed = self.fc(h)
        return embed


class SpectralSimCLR(nn.Module):
    """
    SimCLR framework for 1D spectra.
    Backbone f_theta maps spectrum -> representation h (embed_dim).
    Projection head g_phi maps h -> projection z (proj_dim) for contrastive loss.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.backbone = ResNet1DBackbone(in_channels=in_channels, embed_dim=embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized feature representation h."""
        h = self.backbone(x)
        return F.normalize(h, dim=-1)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Returns normalized projection z."""
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a pair of augmented views."""
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        z1 = self.project(h1)
        z2 = self.project(h2)
        return z1, z2


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent / InfoNCE).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (B, D) L2-normalized projections
        """
        B, D = z1.shape
        z_all = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Similarity matrix: (2B, 2B)
        sim = torch.matmul(z_all, z_all.T) / self.temperature

        # Mask out self-contrast cases (diagonal)
        mask = torch.eye(2 * B, device=z1.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive targets: for i in [0..B-1], positive is i + B, and vice-versa
        targets = torch.cat([
            torch.arange(B, 2 * B, device=z1.device),
            torch.arange(0, B, device=z1.device)
        ], dim=0)

        loss = F.cross_entropy(sim, targets)
        return loss


class GatedAttentionPooling1D(nn.Module):
    """
    Gated Attention-based Pooling (Ilse et al. 2018).
    Dynamically computes importance weights across spectral wavelength tokens.
    Allows prioritizing sharp/broad emission or absorption lines over uniform background.
    """
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_v = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attn_u = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        Args:
            x: (B, L, C) token sequence
        Returns:
            pooled: (B, C) attention-weighted representation
            weights: (B, L) attention distribution across wavelength positions
        """
        v = self.attn_v(x)
        u = self.attn_u(x)
        scores = self.attn_w(v * u)  # (B, L, 1)
        weights = F.softmax(scores, dim=1)  # (B, L, 1)

        pooled = torch.sum(x * weights, dim=1)  # (B, C)
        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled


class SpectralTransformerBlock(nn.Module):
    """
    Transformer Encoder layer with Multi-Head Self-Attention for 1D spectra.
    Enables long-range relational reasoning between disparate emission/absorption lines
    across different arms and wavelengths.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Transformer
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class AttentiveResNet1DBackbone(nn.Module):
    """
    1D-ResNet Backbone augmented with Multi-Head Self-Attention and Gated Attention Pooling.
    Combines localized line sensitivity (Attention Pooling) with broad continuum tracking (GAP).
    Input: (B, 1, 4096) -> Output: (B, embed_dim)
    """
    def __init__(self, embed_dim: int = 128, seq_len: int = 256, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),  # -> 2048
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.stage1 = nn.Sequential(
            ResidualBlock1D(32, 32, stride=1),
            ResidualBlock1D(32, 32, stride=1)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(32, 64, stride=2),   # -> 1024
            ResidualBlock1D(64, 64, stride=1)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),  # -> 512
            ResidualBlock1D(128, 128, stride=1)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock1D(128, d_model, stride=2),  # -> 256
            ResidualBlock1D(d_model, d_model, stride=1)
        )

        # 1D Positional Embedding for 256 spectral sequence positions
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Multi-Head Self-Attention block
        self.transformer_block = SpectralTransformerBlock(d_model=d_model, nhead=nhead, dim_feedforward=512)

        # Gated Attention Pooling
        self.attn_pool = GatedAttentionPooling1D(in_features=d_model, hidden_dim=128)

        # Dual fusion: Attention-weighted (line-sensitive) + Global Average (continuum-sensitive)
        self.norm_combine = nn.LayerNorm(d_model * 2)
        self.fc = nn.Linear(d_model * 2, embed_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        h = self.init_conv(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)  # (B, d_model=256, L=256)

        # Reshape to token sequence: (B, L, C)
        seq = h.transpose(1, 2)
        seq = seq + self.pos_embed

        # Multi-Head Self-Attention
        seq = self.transformer_block(seq)

        # Gated Attention Pooling & Average Pooling
        h_attn, weights = self.attn_pool(seq, return_weights=True)
        h_avg = torch.mean(seq, dim=1)

        # Fuse both representations
        h_combined = torch.cat([h_attn, h_avg], dim=-1)
        h_normed = self.norm_combine(h_combined)
        embed = self.fc(h_normed)

        if return_attention:
            return embed, weights
        return embed


class SpectralAttentionSimCLR(nn.Module):
    """
    SimCLR framework with Self-Attention & Attentive Pooling Backbone.
    """
    def __init__(self, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.backbone = AttentiveResNet1DBackbone(embed_dim=embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )

    def encode(self, x: torch.Tensor, return_attention: bool = False):
        """Returns normalized feature representation h and optional attention weights."""
        if return_attention:
            h, weights = self.backbone(x, return_attention=True)
            return F.normalize(h, dim=-1), weights
        h = self.backbone(x)
        return F.normalize(h, dim=-1)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        z1 = self.project(h1)
        z2 = self.project(h2)
        return z1, z2


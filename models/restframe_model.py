"""Rest-Frame Latent Spectral Model with Unsupervised Redshift Alignment.

Mathematical Principle:
On uniform log-wavelength grid x = ln(lambda), redshift z is a pure 1D translation:
    x_obs = x_rest + Delta, where Delta = ln(1+z) >= 0.

Architecture:
1. Shift-Invariant 1D-ResNet Encoder: maps clean observed spectrum -> intrinsic spectral latent h in R^d.
2. Rest-Frame Decoder: decodes h -> canonical rest-frame model spectrum T_rest(ln lambda_rest) spanning 80 nm to 1260 nm.
3. Fast 1D Alignment Module: aligns T_rest with f_obs without supervision via weighted cross-correlation/chi^2,
   reconstructing the observed spectrum while simultaneously predicting the unsupervised redshift z_est = exp(Delta*) - 1.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from config import WL_MIN, WL_MAX, N_GRID


# Compute physical log-wavelength step
DX = float((np.log(WL_MAX) - np.log(WL_MIN)) / (N_GRID - 1))
WL_REST_MIN = 80.0
LN_REST_MIN = float(np.log(WL_REST_MIN))
LN_OBS_MIN = float(np.log(WL_MIN))

# Rest-frame grid size (covers rest-UV 80 nm to rest-NIR 1260 nm at same resolution dx)
N_REST = int(np.round((np.log(WL_MAX) - LN_REST_MIN) / DX)) + 1
OFFSET_Z0 = int(np.round((LN_OBS_MIN - LN_REST_MIN) / DX))  # Offset index when z = 0 (~5118)

REST_LOG_WL = LN_REST_MIN + np.arange(N_REST) * DX
REST_WL = np.exp(REST_LOG_WL)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class RestFrameDecoder(nn.Module):
    """
    Decodes latent vector h -> rest-frame spectrum T_rest (N_REST = 9214 points).
    """

    def __init__(self, latent_dim: int = 64, out_len: int = N_REST):
        super().__init__()
        self.out_len = out_len
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128 * 72),  # Initial feature map: (128, 72)
            nn.GELU()
        )

        # Upsampling stages to reach >= 9214 points
        # 72 -> 144 -> 288 -> 576 -> 1152 -> 2304 -> 4608 -> 9216
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 144
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResBlock1D(64),
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),   # -> 288
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 576
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResBlock1D(32),
            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1),   # -> 1152
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),   # -> 2304
            nn.BatchNorm1d(16),
            nn.GELU(),
            ResBlock1D(16),
            nn.ConvTranspose1d(16, 16, kernel_size=4, stride=2, padding=1),   # -> 4608
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),    # -> 9216
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=7, padding=3)                          # -> (1, 9216)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        x = self.fc(h).view(b, 128, 72)
        out = self.deconv(x)  # (B, 1, 9216)
        return out[:, :, :self.out_len]  # Trim to exact N_REST


class RestFrameAlignModel(nn.Module):
    """
    Complete Unsupervised Rest-Frame Spectral Alignment & Redshift Estimation Architecture.
    """

    def __init__(self, latent_dim: int = 64, max_z: float = 3.6, search_step: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_z = max_z
        self.search_step = search_step
        self.n_obs = N_GRID
        self.n_rest = N_REST
        self.dx = DX
        self.offset_z0 = OFFSET_Z0

        # Max shift in pixels (from z=0 to z=max_z)
        max_shift_pix = int(np.round(np.log(1.0 + max_z) / self.dx))
        self.min_start_idx = max(0, self.offset_z0 - max_shift_pix)
        self.max_start_idx = self.offset_z0

        # Precompute candidate start offsets for rest-frame slice search
        self.candidate_offsets = np.arange(self.min_start_idx, self.max_start_idx + 1, self.search_step)
        # Corresponding redshifts for each candidate offset:
        # start_idx = offset_z0 - shift_pix -> shift_pix = offset_z0 - start_idx
        shift_pix = self.offset_z0 - self.candidate_offsets
        self.candidate_redshifts = np.exp(shift_pix * self.dx) - 1.0

        # 1D-ResNet Encoder
        from models.contrastive_model import ResNet1DBackbone
        self.encoder = ResNet1DBackbone(embed_dim=latent_dim)

        # Rest-Frame Decoder
        self.decoder = RestFrameDecoder(latent_dim=latent_dim, out_len=self.n_rest)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent representation h (B, latent_dim)."""
        return F.normalize(self.encoder(x), dim=-1)

    def decode_rest(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstructs rest-frame spectrum T_rest (B, 1, N_REST)."""
        return self.decoder(h)

    def align_and_estimate_redshift(
        self,
        f_obs: torch.Tensor,
        w_obs: torch.Tensor,
        t_rest: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs fast GPU-vectorized weighted cross-correlation / chi^2 search
        to find optimal redshift shift for each sample in batch.

        Returns:
            f_model_obs: (B, N_GRID) Aligned model spectrum in observed frame
            z_est: (B,) Estimated redshifts
            best_chi2: (B,) Weighted chi^2 reconstruction loss per sample
        """
        B = f_obs.shape[0]
        device = f_obs.device

        # Unfold rest-frame template into candidate slices of size N_GRID
        # t_rest shape: (B, 1, N_REST)
        # Slice from min_start_idx to max_start_idx + N_GRID
        valid_span = t_rest[:, :, self.min_start_idx : self.max_start_idx + self.n_obs]
        # Unfold along dimension 2: (B, 1, n_cand, N_GRID)
        slices = valid_span.unfold(dimension=2, size=self.n_obs, step=self.search_step).squeeze(1)

        # f_obs: (B, N_GRID), w_obs: (B, N_GRID)
        f_b = f_obs[:, None, :]      # (B, 1, N_GRID)
        w_b = w_obs[:, None, :]      # (B, 1, N_GRID)

        # Vectorized weighted squared residual: chi^2 = sum(w * (f - slice)^2) / sum(w)
        # (f - slice)^2
        diff_sq = (f_b - slices) ** 2  # (B, n_cand, N_GRID)
        w_sum = torch.clamp(w_b.sum(dim=-1), min=1e-5)  # (B, 1)
        chi2_all = (diff_sq * w_b).sum(dim=-1) / w_sum  # (B, n_cand)

        # Find best shift index minimizing chi2
        best_cand_idx = torch.argmin(chi2_all, dim=-1)  # (B,)

        # Gather best slices
        idx_gather = best_cand_idx[:, None, None].expand(-1, 1, self.n_obs)
        f_model_obs = torch.gather(slices, dim=1, index=idx_gather).squeeze(1)

        best_chi2 = torch.gather(chi2_all, dim=1, index=best_cand_idx[:, None]).squeeze(1)

        # Convert candidate index to redshift
        cand_z_tensor = torch.from_numpy(self.candidate_redshifts.astype(np.float32)).to(device)
        z_est = cand_z_tensor[best_cand_idx]

        return f_model_obs, z_est, best_chi2

    def forward(
        self,
        f_clean: torch.Tensor,
        f_raw: torch.Tensor,
        w_obs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass:
        1. Encodes clean flux -> latent h
        2. Decodes rest-frame template T_rest
        3. Aligns T_rest with f_raw via weighted chi^2 minimization
        4. Outputs estimated redshift and aligned reconstruction
        """
        # Ensure 3D input: (B, 1, N_GRID)
        if f_clean.ndim == 2:
            f_clean = f_clean.unsqueeze(1)
        if f_raw.ndim == 2:
            f_raw = f_raw.unsqueeze(1)
        if w_obs.ndim == 2:
            w_obs = w_obs.unsqueeze(1)

        h = self.encode(f_clean)
        t_rest = self.decode_rest(h)  # (B, 1, N_REST)

        f_model_obs, z_est, chi2 = self.align_and_estimate_redshift(
            f_raw.squeeze(1), w_obs.squeeze(1), t_rest
        )

        return {
            'latent': h,                  # (B, latent_dim)
            't_rest': t_rest.squeeze(1),  # (B, N_REST) rest-frame template
            'f_model_obs': f_model_obs,   # (B, N_GRID) aligned reconstruction in observed frame
            'z_est': z_est,               # (B,) unsupervised estimated redshift
            'chi2': chi2,                 # (B,) weighted chi^2 loss
            'loss': chi2.mean()
        }

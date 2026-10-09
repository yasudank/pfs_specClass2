"""Spectral Data Augmentation for Self-Supervised Contrastive Learning.
Includes random shift (redshift invariance), random masking, and continuum tilt.
"""
import torch
import numpy as np


class SpectralAugmentation:
    """
    Data augmentation module for 1D spectra on log-wavelength grid.
    Translates wavelength shift directly into array translation.
    """

    def __init__(
        self,
        max_shift: int = 512,        # Maximum pixel shift in log-lambda (~ delta ln(lambda) ~ 0.15, z up to ~0.16 or relative shifts)
        mask_prob: float = 0.5,       # Probability of random cutout
        num_masks: int = 3,           # Number of mask segments
        mask_width: int = 64,         # Width of each mask segment
        noise_std: float = 0.02,      # Additive Gaussian noise
        tilt_std: float = 0.1,        # Continuum tilt std
    ):
        self.max_shift = max_shift
        self.mask_prob = mask_prob
        self.num_masks = num_masks
        self.mask_width = mask_width
        self.noise_std = noise_std
        self.tilt_std = tilt_std

    def __call__(self, flux: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """
        Applies random augmentations to a single spectrum or batch.
        Args:
            flux: (B, 1, L) or (1, L)
        Returns:
            aug_flux: augmented tensor of same shape
        """
        is_batched = (flux.ndim == 3)
        if not is_batched:
            flux = flux.unsqueeze(0)  # (1, 1, L)

        B, C, L = flux.shape
        device = flux.device
        aug = flux.clone()

        # 1. Random Shift (Simulating Redshift variations) across all channels synchronously
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=device)
        for i in range(B):
            sh = shifts[i].item()
            if sh > 0:
                # Shift right (redshift)
                aug[i, :, sh:] = flux[i, :, :-sh]
                aug[i, :, :sh] = 0.0
            elif sh < 0:
                # Shift left (blueshift)
                sh_abs = abs(sh)
                aug[i, :, :-sh_abs] = flux[i, :, sh_abs:]
                aug[i, :, -sh_abs:] = 0.0

        # 2. Random Continuum Tilt (Multiplicative gradual slope)
        # S(lambda) * (lambda / lambda_0)^alpha -> in log space: exp(alpha * (x - x_mid))
        if self.tilt_std > 0:
            alphas = torch.randn(B, 1, 1, device=device) * self.tilt_std
            x = torch.linspace(-1.0, 1.0, L, device=device).view(1, 1, L)
            tilt_curve = torch.exp(alphas * x)
            aug = aug * tilt_curve

        # 3. Random Masking (Cutout) across all channels synchronously
        if self.mask_prob > 0:
            for i in range(B):
                if torch.rand(1).item() < self.mask_prob:
                    for _ in range(self.num_masks):
                        start = torch.randint(0, L - self.mask_width, (1,)).item()
                        aug[i, :, start : start + self.mask_width] = 0.0

        # 4. Additive Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(aug) * self.noise_std
            aug = aug + noise

        if not is_batched:
            aug = aug.squeeze(0)

        return aug

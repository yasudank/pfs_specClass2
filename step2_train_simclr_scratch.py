"""Step 2: Train Spectral SimCLR Contrastive Representation Learning Model from Scratch
- Uses 53,740 representative spectra from full survey (including 10x augmented QSOs)
- Clean slate: trains completely from scratch (random initialization)
- Uses translation invariance (max_shift=512 pixels in ln(lambda))
- Runs on RTX 4090 GPU
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, N_GRID
from models.contrastive_model import SpectralSimCLR, NTXentLoss
from augmentations import SpectralAugmentation


def train_simclr_scratch(epochs: int = 30, batch_size: int = 256, lr: float = 1.2e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Step 2: Training SimCLR from Scratch on {device} ===")
    t0 = time.time()

    # 1. Load dataset
    cache_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    print(f"Loading dataset from {cache_path}...")
    data = np.load(cache_path, allow_pickle=True)
    flux_clean = data['flux_clean']  # (53740, 4096)
    classes = data['classes']
    print(f"Loaded {len(flux_clean)} spectra: classes={dict(zip(*np.unique(classes, return_counts=True)))}")

    # 2. Build Model & Augmenter
    model = SpectralSimCLR(embed_dim=128, proj_dim=64).to(device)
    augmenter = SpectralAugmentation(
        max_shift=512,  # up to ~0.15 in ln(lambda), broad shift invariance
        mask_prob=0.6,
        num_masks=3,
        mask_width=48,
        noise_std=0.02,
        tilt_std=0.08
    )

    dataset = TensorDataset(torch.from_numpy(flux_clean).float().unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Training SpectralSimCLR for {epochs} epochs (batch_size={batch_size})...")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        ep_t0 = time.time()

        for (bx,) in loader:
            bx = bx.to(device, non_blocking=True)
            v1 = augmenter(bx)
            v2 = augmenter(bx)

            optimizer.zero_grad(set_to_none=True)
            z1, z2 = model(v1, v2)
            loss = criterion(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        ep_time = time.time() - ep_t0
        avg_loss = total_loss / max(n_batches, 1)

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs:02d} ({ep_time:.1f}s) | NT-Xent Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    out_model_path = RESULTS_DIR / "spectral_simclr_scratch_v3.pt"
    torch.save(model.state_dict(), out_model_path)
    print(f"Model saved to {out_model_path} in {time.time() - t0:.1f}s.")
    return out_model_path


if __name__ == "__main__":
    train_simclr_scratch()

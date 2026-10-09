"""Train Attentive Spectral SimCLR Model with Multi-Head Self-Attention and Gated Attention Pooling.
Runs on RTX 4090 GPU using the balanced unique dataset.
"""
import os
import sys
import time
from pathlib import Path
import numpy as np

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from models.contrastive_model import SpectralAttentionSimCLR, NTXentLoss
from augmentations import SpectralAugmentation
from config import RESULTS_DIR


def train_attentive_simclr(epochs: int = 15, batch_size: int = 128, lr: float = 1.0e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Attentive SimCLR on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    t0 = time.time()

    data_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    data = np.load(data_path, allow_pickle=True)
    all_flux = data['flux_clean']
    all_classes = data['classes']
    obj_ids = data['objId']

    # Filter to unique targets (41,374 unique spectra: 1,374 QSOs, 20,000 Galaxies, 20,000 Stars)
    _, unique_indices = np.unique(obj_ids, return_index=True)
    flux = all_flux[unique_indices]
    classes = all_classes[unique_indices]

    counts = dict(zip(*np.unique(classes, return_counts=True)))
    print(f"Unique representative dataset: {len(flux)} targets -> {counts}")

    # WeightedRandomSampler: balances sampling frequency of STAR, GALAXY, and QSO equally in every epoch
    weights = np.array([1.0 / counts[c] for c in classes], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(classes), replacement=True)

    dataset = TensorDataset(torch.from_numpy(flux).float().unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    model = SpectralAttentionSimCLR(embed_dim=128, proj_dim=64).to(device)
    augmenter = SpectralAugmentation(
        max_shift=512,
        mask_prob=0.6,
        num_masks=3,
        mask_width=48,
        noise_std=0.02,
        tilt_std=0.08
    )
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        ep_t0 = time.time()

        for (bx,) in loader:
            bx = bx.to(device)
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

        if ep % 2 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs:02d} ({ep_time:.1f}s) | NT-Xent Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    out_path = RESULTS_DIR / "spectral_simclr_attention_model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Attentive SimCLR model saved to {out_path} in {time.time() - t0:.1f}s.")
    return out_path


if __name__ == "__main__":
    train_attentive_simclr()

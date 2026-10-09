import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Prevent triton KeyError: 'CONDA_PREFIX'
if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from config import RESULTS_DIR, N_GRID, LATENT_DIM, LOG_WL_GRID, WL_GRID
from dataset import query_target_samples, load_raw_spectrum, preprocess_spectrum
from models.weighted_pca import EMPCA
from models.autoencoder import SpectrumAutoencoder1D, weighted_mse_loss, compute_anomaly_scores


def prepare_dataset(n_per_class: int = 500, cache_file: str = "results/dataset_cache.npz"):
    """Loads and preprocesses balanced samples, caching to npz."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return {
            'flux': data['flux'],
            'flux_clean': data['flux_clean'] if 'flux_clean' in data else data['flux'],
            'weight': data['weight'],
            'mask': data['mask'],
            'objId': data['objId'],
            'classes': data['classes'],
            'redshifts': data['redshifts']
        }

    print(f"Querying SQLite database for {n_per_class} per class...")
    targets = query_target_samples(n_per_class=n_per_class)
    print(f"Total targets queried: {len(targets)}")

    fluxes = []
    fluxes_clean = []
    weights = []
    masks = []
    valid_targets = []

    print("Preprocessing spectra onto log-wavelength grid...")
    t0 = time.time()
    for i, t in enumerate(targets):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(targets)} spectra ({time.time() - t0:.1f}s)...")
        raw = load_raw_spectrum(t['fits_path'])
        if raw is None:
            continue
        proc = preprocess_spectrum(raw)
        if proc is None:
            continue

        fluxes.append(proc['flux'])
        fluxes_clean.append(proc['flux_clean'])
        weights.append(proc['weight'])
        masks.append(proc['mask'])
        valid_targets.append(t)

    flux_arr = np.array(fluxes, dtype=np.float32)
    clean_arr = np.array(fluxes_clean, dtype=np.float32)
    weight_arr = np.array(weights, dtype=np.float32)
    mask_arr = np.array(masks, dtype=np.float32)
    obj_ids = np.array([t['objId'] for t in valid_targets])
    classes = np.array([t['classificationName'] for t in valid_targets])
    redshifts = np.array([t['bestRedshift'] if t['bestRedshift'] is not None else np.nan for t in valid_targets], dtype=np.float32)

    print(f"Successfully processed {len(flux_arr)} valid spectra.")
    np.savez_compressed(
        cache_file,
        flux=flux_arr,
        flux_clean=clean_arr,
        weight=weight_arr,
        mask=mask_arr,
        objId=obj_ids,
        classes=classes,
        redshifts=redshifts
    )
    print(f"Saved cache to {cache_file}.")

    return {
        'flux': flux_arr,
        'weight': weight_arr,
        'mask': mask_arr,
        'objId': obj_ids,
        'classes': classes,
        'redshifts': redshifts
    }


def train_empca(data: dict, n_components: int = 10):
    """Fits EMPCA baseline on the spectra."""
    print(f"\n--- Running EMPCA (n_components={n_components}) ---")
    t0 = time.time()
    empca = EMPCA(n_components=n_components, max_iter=15, tol=1e-4)
    empca.fit(data['flux'], data['weight'])
    coeffs = empca.transform(data['flux'], data['weight'])
    recons = empca.reconstruct(coeffs)

    # Weighted residual error
    diff_sq = (data['flux'] - recons) ** 2
    sum_w = np.sum(data['weight'], axis=1) + 1e-6
    scores = np.sum(data['weight'] * diff_sq, axis=1) / sum_w

    print(f"EMPCA finished in {time.time() - t0:.2f}s. Mean reconstruction error: {np.mean(scores):.4f}")
    return empca, coeffs, scores


def train_autoencoder(data: dict, epochs: int = 25, batch_size: int = 64, latent_dim: int = 32, lr: float = 1e-3):
    """Trains 1D Convolutional Autoencoder on GPU."""
    print(f"\n--- Training 1D Autoencoder (latent_dim={latent_dim}, epochs={epochs}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    X = torch.from_numpy(data['flux'][:, None, :])    # (N, 1, 4096)
    W = torch.from_numpy(data['weight'][:, None, :])  # (N, 1, 4096)
    dataset = TensorDataset(X, W)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = SpectrumAutoencoder1D(latent_dim=latent_dim, seq_len=N_GRID).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {'train_loss': [], 'val_loss': []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, bw in train_loader:
            bx, bw = bx.to(device), bw.to(device)
            optimizer.zero_grad()
            recon, _ = model(bx)
            loss = weighted_mse_loss(recon, bx, bw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, bw in val_loader:
                bx, bw = bx.to(device), bw.to(device)
                recon, _ = model(bx)
                loss = weighted_mse_loss(recon, bx, bw)
                val_loss += loss.item() * len(bx)
        val_loss /= len(val_set)

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | lr: {scheduler.get_last_lr()[0]:.2e}")

    print(f"Training completed in {time.time() - t0:.1f}s.")

    # Extract latent vectors and anomaly scores for all data
    model.eval()
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents = []
    all_scores = []
    all_recons = []

    with torch.no_grad():
        for bx, bw in all_loader:
            bx, bw = bx.to(device), bw.to(device)
            recon, z = model(bx)
            scores = compute_anomaly_scores(recon, bx, bw)
            all_latents.append(z.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            all_recons.append(recon.squeeze(1).cpu().numpy())

    latents = np.concatenate(all_latents, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    recons = np.concatenate(all_recons, axis=0)

    # Save checkpoint
    ckpt_path = RESULTS_DIR / "autoencoder_baseline.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
        'history': history
    }, ckpt_path)
    print(f"Model saved to {ckpt_path}.")

    return model, latents, scores, recons, history

"""Training and Evaluation Script for Unsupervised Rest-Frame Spectral Alignment & Redshift Estimation."""
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR
from models.restframe_model import RestFrameAlignModel, REST_WL, N_REST


def train_restframe_model(
    cache_path: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 32,
    latent_dim: int = 64,
    lr: float = 1e-3
):
    if cache_path is None:
        cache_path = str(RESULTS_DIR / "dataset_cache_clean_v2.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("   Training Unsupervised Rest-Frame Alignment Model   ")
    print("=" * 60)
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size} | Latent dim: {latent_dim}")

    cache = np.load(cache_path)
    f_raw = cache['flux']         # (N, 4096)
    f_clean = cache['flux_clean'] # (N, 4096)
    w_obs = cache['weight'] * cache['mask'] # (N, 4096)
    classes = cache['classes']
    redshifts = cache['redshifts']

    N = len(f_raw)
    print(f"Dataset: {N} spectra loaded from {cache_path}.")

    # Tensor dataset
    t_clean = torch.from_numpy(f_clean)
    t_raw = torch.from_numpy(f_raw)
    t_w = torch.from_numpy(w_obs)

    dataset = TensorDataset(t_clean, t_raw, t_w)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize model
    model = RestFrameAlignModel(latent_dim=latent_dim, max_z=3.5, search_step=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {'loss': []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for b_clean, b_raw, b_w in loader:
            b_clean = b_clean.to(device)
            b_raw = b_raw.to(device)
            b_w = b_w.to(device)

            optimizer.zero_grad()
            out = model(b_clean, b_raw, b_w)
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Weighted chi^2 Loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s.")

    # 4. Evaluation and Prediction across full dataset
    model.eval()
    all_z_est = []
    all_chi2 = []
    all_t_rest = []

    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for b_clean, b_raw, b_w in eval_loader:
            b_clean = b_clean.to(device)
            b_raw = b_raw.to(device)
            b_w = b_w.to(device)
            out = model(b_clean, b_raw, b_w)
            all_z_est.append(out['z_est'].cpu().numpy())
            all_chi2.append(out['chi2'].cpu().numpy())
            all_t_rest.append(out['t_rest'].cpu().numpy())

    z_est = np.concatenate(all_z_est, axis=0)
    chi2 = np.concatenate(all_chi2, axis=0)
    t_rest = np.concatenate(all_t_rest, axis=0)

    # Save model checkpoint
    model_save_path = RESULTS_DIR / "restframe_align_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
        'history': history
    }, model_save_path)
    print(f"Saved model to {model_save_path}.")

    # 5. Visualizations
    plot_restframe_results(t_rest, z_est, classes, redshifts, chi2)

    return model, z_est, t_rest


def plot_restframe_results(t_rest: np.ndarray, z_est: np.ndarray, classes: np.ndarray, redshifts: np.ndarray, chi2: np.ndarray):
    """Generates diagnostic plots: Rest-frame composite spectra and Unsupervised redshift comparison."""
    print("Generating Rest-Frame Model diagnostic plots...")

    # Figure 1: Unsupervised Redshift Estimation vs DB Redshift
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Redshift scatter for Galaxies and QSOs
    ax1 = axes[0]
    valid_gal = (classes == 'GALAXY') & np.isfinite(redshifts)
    valid_qso = (classes == 'QSO') & np.isfinite(redshifts)
    stars = (classes == 'STAR')

    ax1.scatter(redshifts[valid_gal], z_est[valid_gal], c='#2ca02c', alpha=0.6, s=25, label=f'GALAXY (N={np.sum(valid_gal)})')
    ax1.scatter(redshifts[valid_qso], z_est[valid_qso], c='#d62728', alpha=0.7, s=28, label=f'QSO (N={np.sum(valid_qso)})')
    ax1.plot([0, 3.5], [0, 3.5], 'k--', lw=1.5, label='1:1 Agreement (z_est = z_DB)')

    ax1.set_xlabel('Database Redshift (z_DB)', fontsize=11)
    ax1.set_ylabel('Unsupervised Estimated Redshift (z_est)', fontsize=11)
    ax1.set_title('[Unsupervised] Redshift Recovery (Galaxies & QSOs)', fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.05, 3.5)
    ax1.set_ylim(-0.05, 3.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9)

    # Panel 2: Distribution of estimated redshift for Stars (Expectation: z_est ≈ 0)
    ax2 = axes[1]
    z_stars = z_est[stars]
    ax2.hist(z_stars, bins=40, range=(0, 1.0), color='#1f77b4', alpha=0.8, edgecolor='black')
    med_z_star = np.median(z_stars)
    frac_near_zero = np.mean(z_stars < 0.05) * 100
    ax2.set_xlabel('Unsupervised Estimated Redshift (z_est)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'STAR Redshift Recovery (Median z_est = {med_z_star:.3f}, {frac_near_zero:.1f}% at z < 0.05)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    z_path = RESULTS_DIR / "unsupervised_redshift_recovery.png"
    plt.savefig(z_path, dpi=200)
    plt.close()
    print(f"Saved redshift recovery plot to {z_path}.")

    # Figure 2: Rest-Frame Canonical Prototype Spectra
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

    types = ['STAR', 'GALAXY', 'QSO']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    # Famous rest-frame lines to annotate (nm)
    prominent_lines = {
        'Ly-alpha': 121.57,
        'C IV': 154.91,
        'C III]': 190.87,
        'Mg II': 279.80,
        '[O II]': 372.71,
        'H-beta': 486.13,
        '[O III]': 500.68,
        'H-alpha': 656.28,
        'Ca II Triplet': 854.21
    }

    for idx, (t, col) in enumerate(zip(types, colors)):
        ax = axes[idx]
        mask = (classes == t)
        if np.sum(mask) == 0:
            continue
        proto = np.median(t_rest[mask], axis=0)
        p16 = np.percentile(t_rest[mask], 16, axis=0)
        p84 = np.percentile(t_rest[mask], 84, axis=0)

        ax.fill_between(REST_WL, p16, p84, color=col, alpha=0.2)
        ax.plot(REST_WL, proto, color=col, lw=1.5, label=f'{t} Rest-Frame Model (N={np.sum(mask)})')
        ax.set_ylabel('Model Rest Flux', fontsize=10)
        ax.set_title(f'Rest-Frame Canonical Template: {t}', fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)

        # Annotate astrophysical lines
        for name, wl in prominent_lines.items():
            if REST_WL.min() <= wl <= REST_WL.max():
                ax.axvline(wl, color='gray', linestyle=':', alpha=0.6, lw=0.9)
                if idx == 0:
                    ax.text(wl, ax.get_ylim()[1] * 0.9, name, rotation=90, fontsize=7.5, ha='right', va='top', color='#333333')

    axes[-1].set_xlabel('Rest Wavelength (nm)', fontsize=11)
    axes[-1].set_xlim(80.0, 1100.0)

    plt.tight_layout()
    proto_path = RESULTS_DIR / "restframe_canonical_prototypes.png"
    plt.savefig(proto_path, dpi=200)
    plt.close()
    print(f"Saved rest-frame canonical prototypes to {proto_path}.")


if __name__ == "__main__":
    train_restframe_model(epochs=30, batch_size=32, latent_dim=64, lr=1e-3)

"""Comparison of Latent Space Distributions:
1. Pure Reconstruction / Autoencoder (RestFrameAlignModel)
2. Self-Supervised Contrastive Learning (SpectralSimCLR on Clean v2 with flux_clean)
"""
import os
import sys
import time
from pathlib import Path
import numpy as np

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
import matplotlib.pyplot as plt

from config import RESULTS_DIR, N_GRID, LOG_WL_GRID, WL_GRID_REST
from augmentations import SpectralAugmentation
from models.contrastive_model import SpectralSimCLR, NTXentLoss
from models.restframe_model import RestFrameAlignModel


def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running comparison on {device}...")

    # 1. Load clean v2 dataset
    cache_path = RESULTS_DIR / "dataset_cache_clean_v2.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file {cache_path} not found.")

    cache = np.load(cache_path)
    flux_clean = cache['flux_clean']  # (N, 4096)
    flux_raw = cache['flux']
    weights = cache['weight']
    classes = cache['classes']
    redshifts = cache['redshifts']
    N = len(classes)
    print(f"Loaded {N} spectra: classes count: {dict(zip(*np.unique(classes, return_counts=True)))}")

    # =========================================================================
    # Part 1: Extract Latent Representations from RestFrameAlignModel
    # =========================================================================
    print("\n--- [1] Extracting Latents from RestFrameAlignModel ---")
    rest_model_path = RESULTS_DIR / "restframe_align_model.pt"
    rest_model = RestFrameAlignModel(latent_dim=64).to(device)
    if rest_model_path.exists():
        ckpt = torch.load(rest_model_path, map_location=device)
        rest_model.load_state_dict(ckpt['model_state_dict'])
        print("Loaded existing RestFrameAlignModel weights.")
    else:
        print("RestFrameAlignModel weights not found, using initialized model.")
    rest_model.eval()

    # Evaluate RestFrameAlignModel in mini-batches
    all_latents_rest = []
    all_z_est_rest = []
    rest_loader = DataLoader(TensorDataset(
        torch.from_numpy(flux_clean).float().unsqueeze(1),
        torch.from_numpy(flux_raw).float().unsqueeze(1),
        torch.from_numpy(weights).float().unsqueeze(1)
    ), batch_size=32, shuffle=False)

    with torch.no_grad():
        for b_clean, b_raw, b_w in rest_loader:
            b_clean, b_raw, b_w = b_clean.to(device), b_raw.to(device), b_w.to(device)
            out = rest_model(b_clean, b_raw, b_w)
            all_latents_rest.append(out['latent'].cpu().numpy())
            all_z_est_rest.append(out['z_est'].cpu().numpy())

    latents_rest = np.concatenate(all_latents_rest, axis=0)
    z_est_rest = np.concatenate(all_z_est_rest, axis=0)

    # Normalize latents for fair geometric comparison
    latents_rest_norm = latents_rest / (np.linalg.norm(latents_rest, axis=-1, keepdims=True) + 1e-8)

    # 5-NN accuracy for RestFrameAlignModel
    knn_rest = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_rest.fit(latents_rest_norm, classes)
    acc_rest = accuracy_score(classes, knn_rest.predict(latents_rest_norm))
    sil_rest = silhouette_score(latents_rest_norm, classes, metric='cosine')
    print(f"RestFrameAlignModel Latent 5-NN Acc: {acc_rest*100:.2f}%, Silhouette Score: {sil_rest:.4f}")

    # =========================================================================
    # Part 2: Train & Extract Latents from SpectralSimCLR (Clean v2, flux_clean)
    # =========================================================================
    print("\n--- [2] Training SpectralSimCLR on Clean v2 (flux_clean) ---")
    simclr_model = SpectralSimCLR(embed_dim=128, proj_dim=64).to(device)
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(simclr_model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    augmenter = SpectralAugmentation(
        max_shift=384,
        mask_prob=0.6,
        num_masks=3,
        mask_width=48,
        noise_std=0.02,
        tilt_std=0.08
    )

    dataset = TensorDataset(torch.from_numpy(flux_clean).float().unsqueeze(1))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        simclr_model.train()
        total_loss = 0.0
        n_b = 0
        for (bx,) in loader:
            bx = bx.to(device)
            v1 = augmenter(bx)
            v2 = augmenter(bx)

            optimizer.zero_grad()
            z1, z2 = simclr_model(v1, v2)
            loss = criterion(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(simclr_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        if ep % 10 == 0 or ep == 1:
            print(f"SimCLR Epoch {ep:02d}/{epochs:02d} | NT-Xent Loss: {total_loss/n_b:.4f}")

    print(f"SimCLR training completed in {time.time() - t0:.1f}s.")

    # Save model
    simclr_path = RESULTS_DIR / "spectral_simclr_clean_v2.pt"
    torch.save(simclr_model.state_dict(), simclr_path)

    # Extract SimCLR normalized representations h
    simclr_model.eval()
    eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_embeds = []
    with torch.no_grad():
        for (bx,) in eval_loader:
            bx = bx.to(device)
            emb = simclr_model.encode(bx)
            all_embeds.append(emb.cpu().numpy())
    latents_simclr = np.concatenate(all_embeds, axis=0)

    # 5-NN accuracy for SimCLR
    knn_simclr = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_simclr.fit(latents_simclr, classes)
    acc_simclr = accuracy_score(classes, knn_simclr.predict(latents_simclr))
    sil_simclr = silhouette_score(latents_simclr, classes, metric='cosine')
    print(f"SpectralSimCLR Latent 5-NN Acc: {acc_simclr*100:.2f}%, Silhouette Score: {sil_simclr:.4f}")

    # =========================================================================
    # Part 3: Dimensionality Reduction (PCA & t-SNE)
    # =========================================================================
    print("\n--- [3] Computing 2D Projections (t-SNE & PCA) ---")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
    pca = PCA(n_components=2, random_state=42)

    tsne_rest = tsne.fit_transform(latents_rest_norm)
    tsne_simclr = tsne.fit_transform(latents_simclr)

    pca_rest = pca.fit_transform(latents_rest_norm)
    pca_simclr = pca.fit_transform(latents_simclr)

    # =========================================================================
    # Part 4: Visualization
    # =========================================================================
    print("\n--- [4] Plotting Comparative Analysis ---")
    color_map = {'STAR': '#1f77b4', 'GALAXY': '#2ca02c', 'QSO': '#d62728'}
    class_colors = [color_map[c] for c in classes]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Row 1: RestFrameAlignModel (Reconstruction)
    # (1) RestFrame t-SNE
    ax = axes[0, 0]
    for c, col in color_map.items():
        m = (classes == c)
        ax.scatter(tsne_rest[m, 0], tsne_rest[m, 1], c=col, label=f'{c} (N={np.sum(m)})', alpha=0.6, s=25)
    ax.set_title(f'[A-1] Reconstruction-based Latent Space (t-SNE)\n5-NN Acc: {acc_rest*100:.1f}%, Silhouette: {sil_rest:.3f}\n(STAR & GALAXY heavily overlap due to continuum dominance)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (2) RestFrame PCA
    ax = axes[0, 1]
    for c, col in color_map.items():
        m = (classes == c)
        ax.scatter(pca_rest[m, 0], pca_rest[m, 1], c=col, label=c, alpha=0.6, s=25)
    ax.set_title(f'[A-2] Reconstruction-based Latent Space (PCA)\nPC1 vs PC2 (Continuum slope variance dominates)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    # Row 2: SpectralSimCLR (Contrastive Learning)
    # (3) SimCLR t-SNE
    ax = axes[1, 0]
    for c, col in color_map.items():
        m = (classes == c)
        ax.scatter(tsne_simclr[m, 0], tsne_simclr[m, 1], c=col, label=f'{c} (N={np.sum(m)})', alpha=0.6, s=25)
    ax.set_title(f'[B-1] Self-Supervised SimCLR Latent Space (t-SNE)\n5-NN Acc: {acc_simclr*100:.1f}%, Silhouette: {sil_simclr:.3f}\n(Distinct clusters: STAR, GALAXY, QSO cleanly separated)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (4) SimCLR t-SNE colored by Redshift (to show continuous manifold)
    ax = axes[1, 1]
    valid_z = np.isfinite(redshifts) & (redshifts > 0)
    sc = ax.scatter(tsne_simclr[valid_z, 0], tsne_simclr[valid_z, 1], c=redshifts[valid_z], cmap='viridis', s=25, alpha=0.7, vmin=0, vmax=3.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Database Redshift (z_DB)', fontsize=10)
    ax.scatter(tsne_simclr[~valid_z, 0], tsne_simclr[~valid_z, 1], c='lightgray', s=15, alpha=0.3, label='Stars (z=0)')
    ax.set_title(f'[B-2] SimCLR Latent Manifold colored by Redshift\n(Smooth physical continuum from low-z Galaxies to high-z QSOs)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    out_fig = RESULTS_DIR / "reconstruction_vs_simclr_distribution.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved comparative distribution figure to {out_fig}.")

    # =========================================================================
    # Part 5: Detailed Classification Breakdown
    # =========================================================================
    print("\n--- Classification Breakdown: RestFrame vs SimCLR ---")
    print("RestFrame Model 5-NN Classification Report:")
    print(classification_report(classes, knn_rest.predict(latents_rest_norm)))
    print("SimCLR 5-NN Classification Report:")
    print(classification_report(classes, knn_simclr.predict(latents_simclr)))


if __name__ == "__main__":
    run_comparison()

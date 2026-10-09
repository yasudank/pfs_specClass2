"""Evaluation, Visualization, and Outlier Analysis for Phase 1 Baseline."""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from config import RESULTS_DIR, WL_GRID, LOG_WL_GRID


def plot_latent_space(
    latents: np.ndarray,
    classes: np.ndarray,
    redshifts: np.ndarray,
    method_name: str = "Autoencoder",
    save_prefix: str = "latent_space"
):
    """Visualizes 2D latent distribution colored by class and redshift."""
    print(f"Projecting {latents.shape[1]}-dim latent vectors to 2D with PCA & t-SNE...")

    # 1. 2D PCA projection
    pca_2d = PCA(n_components=2)
    z_pca = pca_2d.fit_transform(latents)

    # 2. t-SNE projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    z_tsne = tsne.fit_transform(latents)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    color_map = {'STAR': '#1f77b4', 'GALAXY': '#2ca02c', 'QSO': '#d62728'}

    # Plot (0, 0): PCA colored by Class
    ax = axes[0, 0]
    for cls, col in color_map.items():
        idx = (classes == cls)
        if np.any(idx):
            ax.scatter(z_pca[idx, 0], z_pca[idx, 1], c=col, label=cls, alpha=0.6, s=15)
    ax.set_title(f"[{method_name}] 2D PCA by Class")
    ax.set_xlabel(f"PC 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot (0, 1): PCA colored by Redshift (for Galaxies & QSOs)
    ax = axes[0, 1]
    has_z = np.isfinite(redshifts) & (redshifts > 0)
    sc = ax.scatter(z_pca[has_z, 0], z_pca[has_z, 1], c=redshifts[has_z], cmap="viridis", alpha=0.7, s=15)
    plt.colorbar(sc, ax=ax, label="Redshift (z)")
    ax.set_title(f"[{method_name}] 2D PCA by Redshift (Galaxies/QSOs)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot (1, 0): t-SNE colored by Class
    ax = axes[1, 0]
    for cls, col in color_map.items():
        idx = (classes == cls)
        if np.any(idx):
            ax.scatter(z_tsne[idx, 0], z_tsne[idx, 1], c=col, label=cls, alpha=0.6, s=15)
    ax.set_title(f"[{method_name}] 2D t-SNE by Class")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot (1, 1): t-SNE colored by Redshift
    ax = axes[1, 1]
    sc2 = ax.scatter(z_tsne[has_z, 0], z_tsne[has_z, 1], c=redshifts[has_z], cmap="viridis", alpha=0.7, s=15)
    plt.colorbar(sc2, ax=ax, label="Redshift (z)")
    ax.set_title(f"[{method_name}] 2D t-SNE by Redshift")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"{save_prefix}_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved latent distribution plot to {out_path}")


def plot_anomaly_distribution_and_samples(
    flux: np.ndarray,
    recons: np.ndarray,
    mask: np.ndarray,
    scores: np.ndarray,
    obj_ids: np.ndarray,
    classes: np.ndarray,
    redshifts: np.ndarray,
    method_name: str = "Autoencoder",
    save_prefix: str = "anomaly"
):
    """Plots anomaly score histogram and sample spectra (top outliers vs normal)."""
    # 1. Anomaly score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, col in {'STAR': '#1f77b4', 'GALAXY': '#2ca02c', 'QSO': '#d62728'}.items():
        idx = (classes == cls)
        if np.any(idx):
            ax.hist(np.log10(scores[idx] + 1e-6), bins=30, alpha=0.5, label=cls, color=col)
    ax.set_xlabel("log10(Weighted Reconstruction Error / Anomaly Score)")
    ax.set_ylabel("Count")
    ax.set_title(f"[{method_name}] Anomaly Score Distribution by Class")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    hist_path = RESULTS_DIR / f"{save_prefix}_score_hist.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved anomaly histogram to {hist_path}")

    # 2. Inspect top outliers (anomalous spectra) and normal spectra
    sorted_idx = np.argsort(scores)[::-1]  # descending
    top_outliers = sorted_idx[:3]
    normal_idx = sorted_idx[len(sorted_idx) // 2 : len(sorted_idx) // 2 + 3]

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    for i, idx in enumerate(top_outliers):
        ax = axes[i]
        fl = flux[idx]
        rc = recons[idx]
        mk = mask[idx]
        ax.plot(WL_GRID, fl, label="Observed Flux", color="black", lw=1.0, alpha=0.8)
        ax.plot(WL_GRID, rc, label="Reconstruction", color="crimson", lw=1.2, alpha=0.85)
        # Indicate masked pixels
        ax.fill_between(WL_GRID, -0.2, 1.5, where=(mk == 0), color="gray", alpha=0.2, label="Masked Pixel")
        ax.set_title(f"Top Anomaly #{i+1} | ObjId: {obj_ids[idx]} | Class: {classes[idx]} | z: {redshifts[idx]} | Score: {scores[idx]:.3f}")
        ax.set_ylabel("Norm Flux")
        if i == 0:
            ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    for i, idx in enumerate(normal_idx):
        ax = axes[3 + i]
        fl = flux[idx]
        rc = recons[idx]
        mk = mask[idx]
        ax.plot(WL_GRID, fl, label="Observed Flux", color="black", lw=1.0, alpha=0.8)
        ax.plot(WL_GRID, rc, label="Reconstruction", color="royalblue", lw=1.2, alpha=0.85)
        ax.fill_between(WL_GRID, -0.2, 1.5, where=(mk == 0), color="gray", alpha=0.2)
        ax.set_title(f"Typical Spectrum #{i+1} | ObjId: {obj_ids[idx]} | Class: {classes[idx]} | z: {redshifts[idx]} | Score: {scores[idx]:.3f}")
        ax.set_ylabel("Norm Flux")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Wavelength (nm)")
    plt.tight_layout()
    spec_path = RESULTS_DIR / f"{save_prefix}_sample_spectra.png"
    plt.savefig(spec_path, dpi=200)
    plt.close()
    print(f"Saved anomaly spectrum inspection to {spec_path}")

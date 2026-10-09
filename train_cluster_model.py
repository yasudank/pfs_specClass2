"""Trains and saves the SpectralClusterClassifier and generates visualization."""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID
from models.contrastive_model import SpectralSimCLR
from models.clustering import SpectralClusterClassifier


def train_and_visualize_clusters():
    print("=" * 60)
    print("      Step B: Training SpectralClusterClassifier (K=6)     ")
    print("=" * 60)

    # 1. Load preprocessed clean v2 dataset
    cache_path = RESULTS_DIR / "dataset_cache_clean_v2.npz"
    cache = np.load(cache_path)
    flux = cache['flux_clean']
    classes = cache['classes']
    redshifts = cache['redshifts']
    print(f"Loaded {len(flux)} spectra from {cache_path}.")

    # 2. Load trained clean SimCLR model and extract embeddings
    model_path = RESULTS_DIR / "spectral_simclr_clean_v2.pt"
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SpectralSimCLR(embed_dim=128, proj_dim=64)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(flux[:, None, :])
        embeddings = model.encode(x_t).numpy()

    print(f"Extracted embeddings shape: {embeddings.shape}")

    # 3. Fit SpectralClusterClassifier
    clf = SpectralClusterClassifier(n_clusters=6, random_state=42)
    clf.fit(embeddings, flux, classes=classes, redshifts=redshifts)

    # Save classifier model
    save_path = RESULTS_DIR / "spectral_cluster_classifier.joblib"
    clf.save(str(save_path))

    # Print summary of clusters
    print("\n--- Discovered Cluster Taxonomies ---")
    for k, info in clf.cluster_info.items():
        print(f"Cluster {k}: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['n_members']} | Dominant: {info['dominant_class']} | Median z: {info['median_z']:.3f}")
        print(f"   Composition: {info['class_breakdown']}")

    # 4. Generate 2D Latent Clustering Map (PCA & t-SNE colored by clusters)
    print("\nGenerating 2D Latent Clustering Map...")
    pca = PCA(n_components=2, random_state=42)
    embeds_pca = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeds_tsne = tsne.fit_transform(embeddings)

    # Predict cluster labels for each sample
    cluster_labels = np.array([clf.predict(embeddings[i])['cluster_id'] for i in range(len(embeddings))])

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
    cluster_names = [f"C{k}: {clf.cluster_info[k]['name'][:24]}..." for k in range(6)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: PCA colored by cluster
    ax1 = axes[0]
    for k in range(6):
        mask = (cluster_labels == k)
        ax1.scatter(
            embeds_pca[mask, 0], embeds_pca[mask, 1],
            c=colors[k], label=cluster_names[k], alpha=0.75, s=22
        )
    ax1.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax1.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax1.set_title("[Step B] 2D PCA by Discovered Cluster", fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: t-SNE colored by cluster
    ax2 = axes[1]
    for k in range(6):
        mask = (cluster_labels == k)
        ax2.scatter(
            embeds_tsne[mask, 0], embeds_tsne[mask, 1],
            c=colors[k], label=cluster_names[k], alpha=0.75, s=22
        )
    ax2.set_xlabel("t-SNE 1", fontsize=11)
    ax2.set_ylabel("t-SNE 2", fontsize=11)
    ax2.set_title("[Step B] 2D t-SNE by Discovered Cluster", fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    map_path = RESULTS_DIR / "cluster_latent_map.png"
    plt.savefig(map_path, dpi=200)
    plt.close()
    print(f"Saved cluster latent map to {map_path}.")

    # Also generate prototype spectra comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for k in range(6):
        ax = axes[k]
        proto = clf.prototype_spectra[k]
        info = clf.cluster_info[k]
        ax.plot(WL_GRID, proto, color=colors[k], lw=1.5, label=f"Prototype (median)")
        ax.set_title(f"Cluster {k}: {info['name']}\n(Dominant: {info['dominant_class']}, med z={info['median_z']:.2f})",
                     fontsize=10, fontweight='bold')
        ax.set_ylabel("Norm Flux", fontsize=9)
        ax.set_ylim(-0.2, 3.5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)

    axes[4].set_xlabel("Observed Wavelength (nm)", fontsize=10)
    axes[5].set_xlabel("Observed Wavelength (nm)", fontsize=10)
    plt.tight_layout()
    proto_path = RESULTS_DIR / "cluster_prototype_grid.png"
    plt.savefig(proto_path, dpi=200)
    plt.close()
    print(f"Saved cluster prototype grid to {proto_path}.")


if __name__ == "__main__":
    train_and_visualize_clusters()

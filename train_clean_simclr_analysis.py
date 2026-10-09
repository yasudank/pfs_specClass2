"""Full Pipeline for Clean SimCLR:
1. Self-supervised training on clean v2 dataset (flux_clean, 370-1260 nm) with shift invariance.
2. Astrophysical clustering & subtype discovery (GMM / K-Means).
3. Cluster prototype spectra visualization (demonstrating sky line suppression & line recovery).
4. Anomaly and pipeline error (mismatch) detection.
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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
import matplotlib.pyplot as plt

from config import RESULTS_DIR, N_GRID, LOG_WL_GRID, WL_GRID
from augmentations import SpectralAugmentation
from models.contrastive_model import SpectralSimCLR, NTXentLoss


def run_clean_simclr_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Clean SimCLR Pipeline on {device}...")

    # 1. Load Clean v2 Dataset
    cache_path = RESULTS_DIR / "dataset_cache_clean_v2.npz"
    cache = np.load(cache_path)
    flux_clean = cache['flux_clean']  # (1200, 4096)
    flux_raw = cache['flux']
    weights = cache['weight']
    obj_ids = cache['objId']
    classes = cache['classes']
    redshifts = cache['redshifts']
    N = len(classes)
    print(f"Loaded {N} spectra: classes: {dict(zip(*np.unique(classes, return_counts=True)))}")

    # 2. Train or Load SpectralSimCLR
    embed_dim = 128
    proj_dim = 64
    model = SpectralSimCLR(embed_dim=embed_dim, proj_dim=proj_dim).to(device)
    model_path = RESULTS_DIR / "spectral_simclr_clean_v2.pt"

    # Retrain with robust parameters: 50 epochs, max_shift=512 (z translation invariance)
    print("Training SpectralSimCLR with translation-invariance (max_shift=512)...")
    augmenter = SpectralAugmentation(
        max_shift=512,  # up to ~0.15 in ln(lambda), broad shift invariance
        mask_prob=0.6,
        num_masks=3,
        mask_width=48,
        noise_std=0.02,
        tilt_std=0.08
    )

    dataset = TensorDataset(torch.from_numpy(flux_clean).float().unsqueeze(1))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 45
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_b = 0
        for (bx,) in loader:
            bx = bx.to(device)
            v1 = augmenter(bx)
            v2 = augmenter(bx)

            optimizer.zero_grad()
            z1, z2 = model(v1, v2)
            loss = criterion(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        if ep % 15 == 0 or ep == 1:
            print(f"Epoch {ep:02d}/{epochs:02d} | NT-Xent Loss: {total_loss/n_b:.4f}")

    print(f"Training completed in {time.time() - t0:.1f}s.")
    torch.save(model.state_dict(), model_path)

    # 3. Extract Representations
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_embeds = []
    with torch.no_grad():
        for (bx,) in eval_loader:
            bx = bx.to(device)
            emb = model.encode(bx)
            all_embeds.append(emb.cpu().numpy())
    latents = np.concatenate(all_embeds, axis=0)  # (N, 128) normalized

    # 4. Astrophysical Subtype Clustering (GMM with K=6)
    K = 6
    gmm = GaussianMixture(n_components=K, covariance_type='diag', random_state=42)
    cluster_labels = gmm.fit_predict(latents)
    log_probs = gmm.score_samples(latents)

    # Characterize each cluster
    cluster_info = []
    for k in range(K):
        c_mask = (cluster_labels == k)
        c_classes = classes[c_mask]
        c_z = redshifts[c_mask]
        c_z_valid = c_z[np.isfinite(c_z)]

        # Class composition
        counts = {c: int(np.sum(c_classes == c)) for c in ['STAR', 'GALAXY', 'QSO']}
        dom_class = max(counts, key=counts.get)
        med_z = float(np.median(c_z_valid)) if len(c_z_valid) > 0 else 0.0

        cluster_info.append({
            'cluster_id': k,
            'count': int(np.sum(c_mask)),
            'counts': counts,
            'dom_class': dom_class,
            'median_z': med_z
        })
        print(f"Cluster {k}: Total={np.sum(c_mask)} | Composition={counts} | Dom={dom_class} | Med_z={med_z:.3f}")

    # 5. Visualization: Latent Map (t-SNE & PCA)
    print("Computing t-SNE and PCA...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
    pca = PCA(n_components=2, random_state=42)
    lat_tsne = tsne.fit_transform(latents)
    lat_pca = pca.fit_transform(latents)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel 1: t-SNE by DB Class
    ax1 = axes[0]
    color_map = {'STAR': '#1f77b4', 'GALAXY': '#2ca02c', 'QSO': '#d62728'}
    for c, col in color_map.items():
        m = (classes == c)
        ax1.scatter(lat_tsne[m, 0], lat_tsne[m, 1], c=col, label=f'{c} (N={np.sum(m)})', alpha=0.6, s=25)
    ax1.set_title('[1] Clean SimCLR: Macro Classes (t-SNE)\n(Shift-Invariant Representation)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Panel 2: t-SNE by Discovered Astrophysical Cluster
    ax2 = axes[1]
    cluster_colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#d62728', '#ff9896']
    for k in range(K):
        m = (cluster_labels == k)
        info = cluster_info[k]
        lbl = f"C{k}: {info['dom_class']} (N={info['count']})"
        ax2.scatter(lat_tsne[m, 0], lat_tsne[m, 1], c=cluster_colors[k], label=lbl, alpha=0.7, s=28)
    ax2.set_title(f'[2] Discovered Subtypes (K={K} GMM Clusters)\n(Physical Spectral Groupings)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.legend(loc='upper right', fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Panel 3: t-SNE by Redshift
    ax3 = axes[2]
    valid_z = np.isfinite(redshifts) & (redshifts > 0)
    sc = ax3.scatter(lat_tsne[valid_z, 0], lat_tsne[valid_z, 1], c=redshifts[valid_z], cmap='plasma', s=25, alpha=0.8, vmin=0, vmax=3.5)
    ax3.scatter(lat_tsne[~valid_z, 0], lat_tsne[~valid_z, 1], c='#cccccc', s=15, alpha=0.3, label='Stars (z=0)')
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('Redshift (z)', fontsize=10)
    ax3.set_title('[3] Continuous Physical Manifold (Redshift Gradient)\n(Smooth Transition from Galaxies to QSOs)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('t-SNE 1')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    latent_map_path = RESULTS_DIR / "clean_simclr_astrophysical_clusters.png"
    plt.savefig(latent_map_path, dpi=200)
    plt.close()
    print(f"Saved latent map to {latent_map_path}.")

    # 6. Visualization: Cluster Prototypes (Observed Frame Composite Spectra)
    # Showing real median spectra with cleaned flux!
    fig, axes = plt.subplots(K, 1, figsize=(16, 14), sharex=True)

    for k in range(K):
        ax = axes[k]
        m = (cluster_labels == k)
        c_flux = flux_clean[m]
        proto = np.median(c_flux, axis=0)
        p16 = np.percentile(c_flux, 16, axis=0)
        p84 = np.percentile(c_flux, 84, axis=0)

        info = cluster_info[k]
        col = cluster_colors[k]

        ax.fill_between(WL_GRID, p16, p84, color=col, alpha=0.2)
        ax.plot(WL_GRID, proto, color=col, lw=1.2, label=f"Cluster {k}: {info['dom_class']} dominant (N={info['count']}, med z={info['median_z']:.2f})")
        ax.set_ylabel("Flux (Normalized)", fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(np.percentile(proto, 1) - 0.5, np.percentile(proto, 99) + 1.5)

    axes[-1].set_xlabel("Observed Wavelength (nm)", fontsize=11)
    axes[-1].set_xlim(370.0, 1260.0)
    plt.suptitle("Discovered Cluster Composite Prototypes (Sky Line Suppressed, 370–1260 nm)", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    proto_path = RESULTS_DIR / "clean_cluster_prototypes.png"
    plt.savefig(proto_path, dpi=200)
    plt.close()
    print(f"Saved cluster prototypes to {proto_path}.")

    # 7. Pipeline Error (Mismatch) & Anomaly Detection
    # Identify objects where GMM dominant class != DB label, or lowest log-probability (anomalies)
    mismatches = []
    for i in range(N):
        c_k = cluster_labels[i]
        dom_c = cluster_info[c_k]['dom_class']
        db_c = classes[i]
        if dom_c != db_c:
            mismatches.append((i, db_c, dom_c, log_probs[i], redshifts[i]))

    print(f"Identified {len(mismatches)} potential pipeline mismatches out of {N} ({len(mismatches)/N*100:.1f}%).")

    # Sort anomalies by lowest log-probability (outliers)
    anomaly_indices = np.argsort(log_probs)[:4]

    # Plot 4 sample anomalies / interesting mismatches
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    for idx, sample_i in enumerate(anomaly_indices):
        ax = axes[idx]
        obj = obj_ids[sample_i]
        c_db = classes[sample_i]
        z_db = redshifts[sample_i]
        c_k = cluster_labels[sample_i]
        dom_k = cluster_info[c_k]['dom_class']
        lp = log_probs[sample_i]

        f = flux_clean[sample_i]
        w = weights[sample_i]

        ax.plot(WL_GRID, f, color='#1f77b4', lw=1.1, label=f"objId: {obj} | DB: {c_db} (z={z_db:.2f}) | SimCLR Cluster: C{c_k} ({dom_k}) | Log-Likelihood: {lp:.1f}")
        ax.set_ylabel("Flux", fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(np.percentile(f, 2) - 0.5, np.percentile(f, 98) + 1.5)

    axes[-1].set_xlabel("Observed Wavelength (nm)", fontsize=11)
    axes[-1].set_xlim(370.0, 1260.0)
    plt.suptitle("Spectral Anomalies & Pipeline Discrepancies Discovered by Clean SimCLR", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    anom_path = RESULTS_DIR / "clean_simclr_anomalies.png"
    plt.savefig(anom_path, dpi=200)
    plt.close()
    print(f"Saved anomalies figure to {anom_path}.")


if __name__ == "__main__":
    run_clean_simclr_pipeline()

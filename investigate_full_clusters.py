"""Detailed Investigation of Latent Space Distribution & Rest-Frame Composite Spectra
Addressing User Queries:
1. What is the latent space distribution with real class balance (all 1,374 QSOs + sampled Galaxies and Stars)?
2. What do the median rest-frame spectra look like for each of the 6 clusters?
"""
import os
import sys
import time
import sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
from config import RESULTS_DIR, WL_GRID, N_GRID
from dataset import load_raw_spectrum, preprocess_spectrum
from models.contrastive_model import SpectralSimCLR
from models.clustering import SpectralClusterClassifier


def process_spec(item):
    obj_id, db_cls, db_z, fits_p, c_id = item
    raw = load_raw_spectrum(fits_p)
    if raw is None:
        return None
    prep = preprocess_spectrum(raw)
    if prep is None:
        return None
    return {
        'objId': obj_id,
        'db_class': db_cls,
        'db_z': db_z,
        'cluster_id': c_id,
        'flux_clean': prep['flux_clean']
    }


def analyze_clusters_and_latents():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Investigating latent distribution and rest-frame spectra on {device}...")

    db_path = RESULTS_DIR / "pfs_classifications_216k.sqlite3"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Sample targets: ALL QSOs (1,374) + 4,000 Galaxies + 4,000 Stars
    cur.execute("SELECT objId, db_class, db_redshift, fits_path, cluster_id FROM classifications WHERE db_class = 'QSO';")
    qso_targets = cur.fetchall()

    cur.execute("SELECT objId, db_class, db_redshift, fits_path, cluster_id FROM classifications WHERE db_class = 'GALAXY' ORDER BY RANDOM() LIMIT 4000;")
    gal_targets = cur.fetchall()

    cur.execute("SELECT objId, db_class, db_redshift, fits_path, cluster_id FROM classifications WHERE db_class = 'STAR' ORDER BY RANDOM() LIMIT 4000;")
    star_targets = cur.fetchall()

    conn.close()

    all_sample_targets = qso_targets + gal_targets + star_targets
    print(f"Sampled {len(all_sample_targets)} targets: QSOs={len(qso_targets)}, Galaxies={len(gal_targets)}, Stars={len(star_targets)}")

    # 2. Parallel loading of spectra
    print("Loading and preprocessing spectra...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(process_spec, all_sample_targets))

    valid_items = [r for r in results if r is not None]
    print(f"Loaded {len(valid_items)} valid spectra in {time.time() - t0:.1f}s.")

    # 3. Extract Latent Embeddings via SimCLR
    print("Extracting SimCLR latent embeddings on GPU...")
    model_path = RESULTS_DIR / "spectral_simclr_clean_v2.pt"
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    encoder = SpectralSimCLR(embed_dim=128, proj_dim=64).to(device)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    flux_matrix = np.array([it['flux_clean'] for it in valid_items], dtype=np.float32)
    db_classes = np.array([it['db_class'] for it in valid_items])
    db_redshifts = np.array([it['db_z'] if it['db_z'] is not None and np.isfinite(it['db_z']) else 0.0 for it in valid_items])
    cluster_ids = np.array([it['cluster_id'] for it in valid_items])

    # GPU inference in batches of 256
    all_embeds = []
    with torch.no_grad():
        for b_i in range(0, len(flux_matrix), 256):
            b_flux = torch.from_numpy(flux_matrix[b_i:b_i + 256]).unsqueeze(1).to(device)
            emb = encoder.encode(b_flux).cpu().numpy()
            all_embeds.append(emb)
    embeddings = np.concatenate(all_embeds, axis=0)
    print(f"Embeddings extracted: {embeddings.shape}")

    # 4. Dimensionality reduction: PCA and t-SNE
    print("Computing PCA and t-SNE on full representative sample...")
    pca = PCA(n_components=2, random_state=42)
    embeds_pca = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, perplexity=35, random_state=42, metric='cosine')
    embeds_tsne = tsne.fit_transform(embeddings)

    # Figure 1: Full-scale Latent Space Distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    color_map = {'STAR': '#1f77b4', 'GALAXY': '#2ca02c', 'QSO': '#d62728'}
    cluster_colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#d62728', '#ff9896']

    # (A) t-SNE by DB Class
    ax = axes[0, 0]
    for c in ['STAR', 'GALAXY', 'QSO']:
        m = (db_classes == c)
        s = 35 if c == 'QSO' else 15
        alpha = 0.85 if c == 'QSO' else 0.4
        ax.scatter(embeds_tsne[m, 0], embeds_tsne[m, 1], c=color_map[c], label=f'{c} (N={np.sum(m)})', s=s, alpha=alpha)
    ax.set_title('[A-1] Full-Scale Latent Space by True DB Label (t-SNE)\n(Notice: True QSOs form a compact, distinct sub-locus!)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (B) PCA by DB Class
    ax = axes[0, 1]
    for c in ['STAR', 'GALAXY', 'QSO']:
        m = (db_classes == c)
        s = 35 if c == 'QSO' else 15
        alpha = 0.85 if c == 'QSO' else 0.4
        ax.scatter(embeds_pca[m, 0], embeds_pca[m, 1], c=color_map[c], label=f'{c} (N={np.sum(m)})', s=s, alpha=alpha)
    ax.set_title('[A-2] Full-Scale Latent Space by True DB Label (PCA)\nPC 1 vs PC 2', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (C) t-SNE by Assigned Cluster ID
    ax = axes[1, 0]
    for k in range(6):
        m = (cluster_ids == k)
        ax.scatter(embeds_tsne[m, 0], embeds_tsne[m, 1], c=cluster_colors[k], label=f'Cluster {k} (N={np.sum(m)})', s=18, alpha=0.6)
    ax.set_title('[B-1] Latent Space Partitioned by the 6 Clusters (t-SNE)\n(Shows how GMM divided the continuous space)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (D) t-SNE colored by Redshift
    ax = axes[1, 1]
    valid_z_mask = (db_redshifts > 0.02)
    sc = ax.scatter(embeds_tsne[valid_z_mask, 0], embeds_tsne[valid_z_mask, 1], c=db_redshifts[valid_z_mask], cmap='plasma', s=20, alpha=0.7, vmin=0, vmax=3.5)
    ax.scatter(embeds_tsne[~valid_z_mask, 0], embeds_tsne[~valid_z_mask, 1], c='#cccccc', s=12, alpha=0.25, label='Stars (z=0)')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Database Redshift (z)', fontsize=10)
    ax.set_title('[B-2] Latent Space colored by Redshift\n(Physical manifold from low-z to high-z)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    latent_fig_path = RESULTS_DIR / "full_scale_latent_investigation.png"
    plt.savefig(latent_fig_path, dpi=200)
    plt.close()
    print(f"Saved latent investigation plot to {latent_fig_path}.")

    # 5. Median Rest-Frame Spectra for each of the 6 Clusters
    print("Computing Median Rest-Frame Spectra for each of the 6 Clusters...")
    # Common rest wavelength grid: 100 nm to 1000 nm
    N_REST = 3000
    rest_wl = np.linspace(100.0, 1000.0, N_REST)

    fig, axes = plt.subplots(6, 1, figsize=(16, 16), sharex=True)

    # Key spectral lines (nm)
    spectral_lines = {
        r'Ly$\alpha$': 121.6,
        r'C IV': 154.9,
        r'C III]': 190.9,
        r'Mg II': 279.8,
        r'[O II]': 372.7,
        r'4000Å': 400.0,
        r'H$\beta$': 486.1,
        r'[O III]': 500.7,
        r'H$\alpha$': 656.3,
        r'Ca II': 854.2
    }

    cluster_descriptions = {
        0: "Cluster 0 (Mostly Intermediate-z Galaxies)",
        1: "Cluster 1 (Cool Stars & Early-Type / Absorption Galaxies)",
        2: "Cluster 2 (Star-Forming Galaxies & Low-z Galaxies)",
        3: "Cluster 3 (High-z QSOs & Hot Blue Stars)",
        4: "Cluster 4 (Late-Type M-Dwarfs & Passive Red Stars)",
        5: "Cluster 5 (Emission-Line Galaxies / AGNs)"
    }

    for k in range(6):
        ax = axes[k]
        c_mask = (cluster_ids == k)
        c_items = [valid_items[i] for i in np.where(c_mask)[0]]

        # Compute rest-frame flux for each item
        rest_fluxes = []
        for it in c_items:
            z = it['db_z']
            if z is None or not np.isfinite(z) or z < 0.001:
                z = 0.0  # stars or z=0
            f = it['flux_clean']
            # obs_wl = rest_wl * (1 + z)
            obs_wl_cand = rest_wl * (1.0 + z)
            f_interp = np.interp(obs_wl_cand, WL_GRID, f, left=np.nan, right=np.nan)
            rest_fluxes.append(f_interp)

        rest_fluxes = np.array(rest_fluxes)
        proto = np.nanmedian(rest_fluxes, axis=0)
        p16 = np.nanpercentile(rest_fluxes, 16, axis=0)
        p84 = np.nanpercentile(rest_fluxes, 84, axis=0)

        # Count true classes in this sample
        c_db_cls = [it['db_class'] for it in c_items]
        cnt_star = sum(1 for c in c_db_cls if c == 'STAR')
        cnt_gal = sum(1 for c in c_db_cls if c == 'GALAXY')
        cnt_qso = sum(1 for c in c_db_cls if c == 'QSO')
        med_z = float(np.nanmedian([it['db_z'] for it in c_items if it['db_z'] is not None and it['db_z'] > 0]))

        col = cluster_colors[k]
        ax.fill_between(rest_wl, p16, p84, color=col, alpha=0.18)
        lbl = f"C{k}: N={len(c_items)} (STAR:{cnt_star}, GAL:{cnt_gal}, QSO:{cnt_qso}) | Med z={med_z:.2f}"
        ax.plot(rest_wl, proto, color=col, lw=1.3, label=lbl)

        ax.set_ylabel("Rest Flux", fontsize=9.5)
        ax.set_title(f"{cluster_descriptions.get(k, f'Cluster {k}')}", fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

        # Mark spectral lines
        for name, line_wl in spectral_lines.items():
            ax.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)
            if k == 0:
                ax.text(line_wl, ax.get_ylim()[1]*0.88, name, rotation=90, fontsize=7.5, color='#444444', ha='right')

        # Limit y-axis sensibly
        y_lo = np.nanpercentile(proto[np.isfinite(proto)], 2) - 0.2 if np.any(np.isfinite(proto)) else -0.5
        y_hi = np.nanpercentile(proto[np.isfinite(proto)], 98) + 0.6 if np.any(np.isfinite(proto)) else 2.0
        ax.set_ylim(max(y_lo, -0.5), min(y_hi, 4.0))

    axes[-1].set_xlabel("Rest-Frame Wavelength (nm)", fontsize=11)
    axes[-1].set_xlim(110.0, 950.0)
    plt.suptitle("Median Rest-Frame Spectra for the 6 Discovered Clusters (Rest Wavelength 110–950 nm)", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    rest_proto_fig = RESULTS_DIR / "six_clusters_median_restframe_spectra.png"
    plt.savefig(rest_proto_fig, dpi=200)
    plt.close()
    print(f"Saved 6 clusters median rest-frame spectra to {rest_proto_fig}.")


if __name__ == "__main__":
    analyze_clusters_and_latents()

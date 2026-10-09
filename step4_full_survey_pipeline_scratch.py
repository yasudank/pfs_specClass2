"""Step 4 & 5: Full Survey Embedding, QSO-Weighted Clustering, and Database Classification from Scratch
- Encodes all 200k survey spectra using freshly trained SimCLR model (spectral_simclr_scratch_v3.pt)
- Saves full embeddings: survey_embeddings_200k_scratch.npy (128D)
- Performs GMM clustering on full dataset with QSO locus weighting (12x weight) to prevent QSO burial
- Computes canonical redshift matching using freshly built canonical templates (canonical_templates_scratch.npz)
- Updates SQLite3 classifications database from scratch
- Generates publication-grade diagnostic plots (Latent Space & Rest-Frame Composite Spectra)
"""
import os
import sys
import time
import sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
from config import RESULTS_DIR, DB_PATH, N_GRID, WL_GRID
from dataset import load_raw_spectrum, preprocess_spectrum
from models.contrastive_model import SpectralSimCLR


def process_spectrum_worker(fits_path):
    raw = load_raw_spectrum(fits_path)
    if raw is None:
        return None
    prep = preprocess_spectrum(raw)
    if prep is None:
        return None
    return {
        'flux_clean': prep['flux_clean'],
        'mask': prep['mask']
    }


def run_full_pipeline_scratch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Step 4 & 5: Full-Scale Survey Re-Analysis from Scratch on {device} ===")
    t_start = time.time()

    # 1. Query all targets from DB
    conn_in = sqlite3.connect(DB_PATH)
    cur_in = conn_in.cursor()
    print("Querying all valid targets with FITS from target_summary...")
    cur_in.execute("""
        SELECT objId, classificationName, bestRedshift, fits_path
        FROM target_summary
        WHERE has_fits = 1
        ORDER BY objId;
    """)
    all_targets = cur_in.fetchall()
    conn_in.close()
    total_targets = len(all_targets)
    print(f"Total targets to process: {total_targets}")

    # 2. Load Scratch Models & Templates
    simclr_path = RESULTS_DIR / "spectral_simclr_scratch_v3.pt"
    if not simclr_path.exists():
        raise FileNotFoundError(f"Model not found at {simclr_path}! Ensure Step 2 has completed.")

    encoder = SpectralSimCLR(embed_dim=128, proj_dim=64).to(device)
    encoder.load_state_dict(torch.load(simclr_path, map_location=device, weights_only=False))
    encoder.eval()
    print("Loaded freshly trained SimCLR scratch model.")

    tmpl_data = np.load(RESULTS_DIR / "canonical_templates_scratch.npz")
    rest_wl = tmpl_data['rest_wl']
    templates = {
        'STAR': tmpl_data['STAR'],
        'GALAXY': tmpl_data['GALAXY'],
        'QSO': tmpl_data['QSO']
    }

    # Redshift template banks
    gal_z_grid = np.linspace(0.0, 1.5, 200)
    qso_z_grid = np.linspace(0.2, 3.8, 250)
    gal_bank = np.array([np.interp(WL_GRID, rest_wl * (1.0 + z), templates['GALAXY'], left=0.0, right=0.0) for z in gal_z_grid])
    gal_bank = (gal_bank - np.mean(gal_bank, axis=1, keepdims=True)) / np.maximum(np.std(gal_bank, axis=1, keepdims=True), 1e-6)
    qso_bank = np.array([np.interp(WL_GRID, rest_wl * (1.0 + z), templates['QSO'], left=0.0, right=0.0) for z in qso_z_grid])
    qso_bank = (qso_bank - np.mean(qso_bank, axis=1, keepdims=True)) / np.maximum(np.std(qso_bank, axis=1, keepdims=True), 1e-6)

    gal_bank_t = torch.from_numpy(gal_bank).float().to(device)
    qso_bank_t = torch.from_numpy(qso_bank).float().to(device)

    # 3. Parallel Extraction of Embeddings across All Targets
    BATCH_CHUNK = 4096
    NUM_WORKERS = 16

    embeddings_list = []
    metadata_list = []
    redshift_estimates = []

    print(f"Beginning parallel extraction across {total_targets} spectra...")
    t_extract = time.time()

    for chunk_start in range(0, total_targets, BATCH_CHUNK):
        chunk_targets = all_targets[chunk_start:chunk_start + BATCH_CHUNK]
        fits_paths = [t[3] for t in chunk_targets]

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            proc_results = list(executor.map(process_spectrum_worker, fits_paths))

        valid_items = []
        for meta, res in zip(chunk_targets, proc_results):
            if res is not None:
                valid_items.append((meta, res))

        if not valid_items:
            continue

        flux_matrix = np.array([item[1]['flux_clean'] for item in valid_items], dtype=np.float32)
        masks = [item[1]['mask'] for item in valid_items]

        with torch.no_grad():
            x_gpu = torch.from_numpy(flux_matrix).unsqueeze(1).to(device)
            embeds = encoder.encode(x_gpu).cpu().numpy()

            # Cross-correlation for redshifts
            f_mean = flux_matrix.mean(axis=1, keepdims=True)
            f_std = np.maximum(flux_matrix.std(axis=1, keepdims=True), 1e-6)
            f_norm_t = torch.from_numpy((flux_matrix - f_mean) / f_std).float().to(device)

            gal_corrs = torch.matmul(f_norm_t, gal_bank_t.T) / N_GRID
            qso_corrs = torch.matmul(f_norm_t, qso_bank_t.T) / N_GRID

            best_gal_val, best_gal_idx = gal_corrs.max(dim=1)
            best_qso_val, best_qso_idx = qso_corrs.max(dim=1)

            best_gal_z = gal_z_grid[best_gal_idx.cpu().numpy()]
            best_gal_r = best_gal_val.cpu().numpy()
            best_qso_z = qso_z_grid[best_qso_idx.cpu().numpy()]
            best_qso_r = best_qso_val.cpu().numpy()

        for i, (meta, prep) in enumerate(valid_items):
            obj_id, db_cls, db_z, fits_p = meta
            embeddings_list.append(embeds[i])
            metadata_list.append({
                'objId': obj_id,
                'db_class': db_cls,
                'db_redshift': float(db_z) if db_z is not None and np.isfinite(db_z) else 0.0,
                'fits_path': fits_p,
                'mask': masks[i]
            })
            redshift_estimates.append({
                'best_gal_z': float(best_gal_z[i]),
                'best_gal_r': float(best_gal_r[i]),
                'best_qso_z': float(best_qso_z[i]),
                'best_qso_r': float(best_qso_r[i])
            })

        done_cnt = min(chunk_start + BATCH_CHUNK, total_targets)
        rate = done_cnt / max(time.time() - t_extract, 0.1)
        print(f"Progress: {done_cnt}/{total_targets} ({done_cnt/total_targets*100:.1f}%) | Rate: {rate:.1f} spec/s | Valid: {len(embeddings_list)}")

    all_embeddings = np.array(embeddings_list, dtype=np.float32)
    print(f"Extraction completed in {time.time() - t_extract:.1f}s. Total valid embeddings: {all_embeddings.shape}")

    # Save full embeddings and metadata to disk
    np.save(RESULTS_DIR / "survey_embeddings_200k_scratch.npy", all_embeddings)
    np.savez_compressed(
        RESULTS_DIR / "survey_meta_216k_scratch.npz",
        objIds=np.array([m['objId'] for m in metadata_list]),
        db_classes=np.array([m['db_class'] for m in metadata_list]),
        db_redshifts=np.array([m['db_redshift'] for m in metadata_list]),
        fits_paths=np.array([m['fits_path'] for m in metadata_list]),
        masks=np.array([m['mask'] for m in metadata_list]),
        best_gal_z=np.array([z['best_gal_z'] for z in redshift_estimates]),
        best_gal_r=np.array([z['best_gal_r'] for z in redshift_estimates]),
        best_qso_z=np.array([z['best_qso_z'] for z in redshift_estimates]),
        best_qso_r=np.array([z['best_qso_r'] for z in redshift_estimates])
    )
    print(f"Saved {all_embeddings.shape} embeddings and metadata to disk.")

    # 4. QSO-Weighted GMM Clustering from Scratch
    print("=== Training Full-Survey Clustering with QSO Oversampling (12x) ===")
    t_clust = time.time()
    db_classes = np.array([m['db_class'] for m in metadata_list])
    is_qso = (db_classes == 'QSO')
    qso_count = np.sum(is_qso)
    print(f"Survey true counts: GALAXY={np.sum(db_classes=='GALAXY')}, STAR={np.sum(db_classes=='STAR')}, QSO={qso_count}")

    # Replicate QSO embeddings by 11 additional times (total 12x)
    qso_embeds = all_embeddings[is_qso]
    augmented_train_embeds = np.concatenate([all_embeddings] + [qso_embeds] * 11, axis=0)
    print(f"Clustering training matrix: {augmented_train_embeds.shape} (includes 12x augmented QSOs: {len(qso_embeds)*12})")

    K_CLUSTERS = 6
    gmm = GaussianMixture(
        n_components=K_CLUSTERS,
        covariance_type='full',
        max_iter=150,
        random_state=42,
        reg_covar=1e-4
    )
    gmm.fit(augmented_train_embeds)
    cluster_labels = gmm.predict(all_embeddings)
    cluster_probs = gmm.predict_proba(all_embeddings)
    print(f"GMM clustering completed in {time.time() - t_clust:.1f}s.")

    # Save clustering model
    joblib.dump(gmm, RESULTS_DIR / "spectral_cluster_classifier_scratch_v3.joblib")

    # 5. Cluster Taxonomy & Characterization
    print("=== Analyzing Discovered Clusters ===")
    cluster_names = {}
    cluster_broad_class = {}

    for k in range(K_CLUSTERS):
        k_mask = (cluster_labels == k)
        k_cls = db_classes[k_mask]
        n_k = np.sum(k_mask)
        cnt_gal = np.sum(k_cls == 'GALAXY')
        cnt_star = np.sum(k_cls == 'STAR')
        cnt_qso = np.sum(k_cls == 'QSO')
        qso_frac = cnt_qso / max(n_k, 1)

        print(f"Cluster {k} (N={n_k}): STAR={cnt_star} ({cnt_star/n_k*100:.1f}%), GAL={cnt_gal} ({cnt_gal/n_k*100:.1f}%), QSO={cnt_qso} ({cnt_qso/n_k*100:.1f}%)")

        if qso_frac > 0.15 or (cnt_qso > 300 and qso_frac > 0.05):
            broad = 'QSO'
            name = f"Cluster {k}: Broad-Line QSO & High-z AGN"
        elif cnt_star > cnt_gal:
            broad = 'STAR'
            name = f"Cluster {k}: Stellar Spectral Class"
        else:
            broad = 'GALAXY'
            name = f"Cluster {k}: Galaxy Spectral Class"

        cluster_broad_class[k] = broad
        cluster_names[k] = name

    # 6. Store Classifications in SQLite3
    print("Writing classifications to database...")
    out_db_path = RESULTS_DIR / "pfs_classifications_216k_scratch.sqlite3"
    if out_db_path.exists():
        out_db_path.unlink()

    conn_out = sqlite3.connect(out_db_path)
    cur_out = conn_out.cursor()
    cur_out.execute("""
        CREATE TABLE classifications (
            objId INTEGER PRIMARY KEY,
            fits_path TEXT,
            db_class TEXT,
            db_redshift REAL,
            pred_class TEXT,
            pred_confidence REAL,
            cluster_id INTEGER,
            cluster_name TEXT,
            cluster_confidence REAL,
            est_redshift REAL,
            est_redshift_quality REAL,
            arm_mode TEXT
        );
    """)

    db_rows = []
    for i, (meta, z_est) in enumerate(zip(metadata_list, redshift_estimates)):
        c_id = int(cluster_labels[i])
        c_conf = float(cluster_probs[i, c_id])
        pred_cls = cluster_broad_class[c_id]
        c_name = cluster_names[c_id]

        if pred_cls == 'STAR':
            est_z = 0.0
            est_r = 1.0
        elif pred_cls == 'QSO':
            est_z = z_est['best_qso_z']
            est_r = z_est['best_qso_r']
        else:
            est_z = z_est['best_gal_z']
            est_r = z_est['best_gal_r']

        mask = meta['mask']
        in_gap = (WL_GRID >= 670.0) & (WL_GRID <= 690.0)
        is_bmn = bool(np.all(mask[in_gap] == 0))
        arm_mode = "BMN" if is_bmn else "BRN"

        db_rows.append((
            meta['objId'],
            meta['fits_path'],
            meta['db_class'],
            meta['db_redshift'],
            pred_cls,
            c_conf,
            c_id,
            c_name,
            c_conf,
            est_z,
            est_r,
            arm_mode
        ))

    cur_out.executemany("""
        INSERT INTO classifications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, db_rows)
    conn_out.commit()
    conn_out.close()
    print(f"Classifications written to {out_db_path} in {time.time() - t_start:.1f}s.")

    # 7. Generate Verification Figures
    generate_verification_plots(all_embeddings, metadata_list, cluster_labels, cluster_broad_class, cluster_names)


def generate_verification_plots(embeddings, metadata_list, cluster_labels, cluster_broad_class, cluster_names):
    print("Generating comprehensive verification figures...")
    db_classes = np.array([m['db_class'] for m in metadata_list])
    db_redshifts = np.array([m['db_redshift'] for m in metadata_list])

    # Sample 10,000 for t-SNE (All QSOs + balanced Galaxies and Stars)
    qso_idx = np.where(db_classes == 'QSO')[0]
    gal_idx = np.random.choice(np.where(db_classes == 'GALAXY')[0], 4500, replace=False)
    star_idx = np.random.choice(np.where(db_classes == 'STAR')[0], 4500, replace=False)
    sample_idx = np.concatenate([qso_idx, gal_idx, star_idx])

    sample_embeds = embeddings[sample_idx]
    sample_classes = db_classes[sample_idx]
    sample_clusters = cluster_labels[sample_idx]
    sample_z = db_redshifts[sample_idx]

    print("Computing t-SNE and PCA...")
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, metric='cosine')
    embeds_tsne = tsne.fit_transform(sample_embeds)

    pca = PCA(n_components=2, random_state=42)
    embeds_pca = pca.fit_transform(sample_embeds)

    # Plot 1: Latent Space Map
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    color_map = {'STAR': '#1f77b4', 'GALAXY': '#2ca02c', 'QSO': '#d62728'}
    cluster_colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#d62728', '#ff9896']

    # (A-1) t-SNE by DB Class
    ax = axes[0, 0]
    for c in ['STAR', 'GALAXY', 'QSO']:
        m = (sample_classes == c)
        s = 35 if c == 'QSO' else 14
        alpha = 0.85 if c == 'QSO' else 0.35
        ax.scatter(embeds_tsne[m, 0], embeds_tsne[m, 1], c=color_map[c], label=f'{c} (N={np.sum(m)})', s=s, alpha=alpha)
    ax.set_title('[A-1] Full Survey Latent Space from Scratch (t-SNE)\nColored by True DB Class', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (A-2) PCA by DB Class
    ax = axes[0, 1]
    for c in ['STAR', 'GALAXY', 'QSO']:
        m = (sample_classes == c)
        s = 35 if c == 'QSO' else 14
        alpha = 0.85 if c == 'QSO' else 0.35
        ax.scatter(embeds_pca[m, 0], embeds_pca[m, 1], c=color_map[c], label=f'{c} (N={np.sum(m)})', s=s, alpha=alpha)
    ax.set_title('[A-2] Full Survey Latent Space from Scratch (PCA)\nPC 1 vs PC 2', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (B-1) t-SNE by Cluster
    ax = axes[1, 0]
    for k in range(len(cluster_names)):
        m = (sample_clusters == k)
        ax.scatter(embeds_tsne[m, 0], embeds_tsne[m, 1], c=cluster_colors[k % len(cluster_colors)], label=f'{cluster_names[k]} (N={np.sum(m)})', s=16, alpha=0.6)
    ax.set_title('[B-1] Latent Space Partitioned by Full-Survey Clusters\n(Trained with QSO Weighting)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.4)

    # (B-2) t-SNE by Redshift
    ax = axes[1, 1]
    valid_z = (sample_z > 0.02)
    sc = ax.scatter(embeds_tsne[valid_z, 0], embeds_tsne[valid_z, 1], c=sample_z[valid_z], cmap='plasma', s=20, alpha=0.7, vmin=0, vmax=3.5)
    ax.scatter(embeds_tsne[~valid_z, 0], embeds_tsne[~valid_z, 1], c='#cccccc', s=12, alpha=0.25, label='Stars (z=0)')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Database Redshift (z)', fontsize=10)
    ax.set_title('[B-2] Full Survey Latent Space Colored by Redshift', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    out_latent_fig = RESULTS_DIR / "full_survey_latent_space_scratch.png"
    plt.savefig(out_latent_fig, dpi=200)
    plt.close()
    print(f"Saved latent space plot to {out_latent_fig}.")

    # Plot 2: Median Rest-Frame Spectra for each of the 6 Clusters
    print("Computing Median Rest-Frame Spectra for each of the discovered clusters...")
    N_REST = 3000
    rest_wl = np.linspace(100.0, 1000.0, N_REST)
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

    fig, axes = plt.subplots(len(cluster_names), 1, figsize=(16, 16), sharex=True)
    if len(cluster_names) == 1:
        axes = [axes]

    for k in range(len(cluster_names)):
        ax = axes[k]
        k_indices = np.where(cluster_labels == k)[0]
        # Sample up to 1000 targets per cluster for fast loading
        sub_idx = k_indices if len(k_indices) <= 1000 else np.random.choice(k_indices, 1000, replace=False)
        sub_fits = [metadata_list[i]['fits_path'] for i in sub_idx]
        sub_z = [metadata_list[i]['db_redshift'] for i in sub_idx]
        sub_classes = [metadata_list[i]['db_class'] for i in sub_idx]

        with ProcessPoolExecutor(max_workers=16) as ex:
            preps = list(ex.map(process_spectrum_worker, sub_fits))

        rest_fluxes = []
        for p, z in zip(preps, sub_z):
            if p is None:
                continue
            use_z = z if (z is not None and np.isfinite(z) and z > 0.001) else 0.0
            obs_wl_cand = rest_wl * (1.0 + use_z)
            f_interp = np.interp(obs_wl_cand, WL_GRID, p['flux_clean'], left=np.nan, right=np.nan)
            rest_fluxes.append(f_interp)

        rest_fluxes = np.array(rest_fluxes)
        proto = np.nanmedian(rest_fluxes, axis=0)
        p16 = np.nanpercentile(rest_fluxes, 16, axis=0)
        p84 = np.nanpercentile(rest_fluxes, 84, axis=0)

        cnt_star = sum(1 for c in sub_classes if c == 'STAR')
        cnt_gal = sum(1 for c in sub_classes if c == 'GALAXY')
        cnt_qso = sum(1 for c in sub_classes if c == 'QSO')
        med_z = float(np.nanmedian([z for z in sub_z if z > 0])) if any(z > 0 for z in sub_z) else 0.0

        col = cluster_colors[k % len(cluster_colors)]
        ax.fill_between(rest_wl, p16, p84, color=col, alpha=0.18)
        lbl = f"{cluster_names[k]} | Total N={len(k_indices)} (Sample STAR:{cnt_star}, GAL:{cnt_gal}, QSO:{cnt_qso}) | Med z={med_z:.2f}"
        ax.plot(rest_wl, proto, color=col, lw=1.3, label=lbl)

        ax.set_ylabel("Rest Flux", fontsize=9.5)
        ax.set_title(f"{cluster_names[k]}", fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

        for name, line_wl in spectral_lines.items():
            ax.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)
            if k == 0:
                ax.text(line_wl, ax.get_ylim()[1]*0.88, name, rotation=90, fontsize=7.5, color='#444444', ha='right')

        y_lo = np.nanpercentile(proto[np.isfinite(proto)], 2) - 0.2 if np.any(np.isfinite(proto)) else -0.5
        y_hi = np.nanpercentile(proto[np.isfinite(proto)], 98) + 0.6 if np.any(np.isfinite(proto)) else 2.0
        ax.set_ylim(max(y_lo, -0.5), min(y_hi, 4.0))

    axes[-1].set_xlabel("Rest-Frame Wavelength (nm)", fontsize=11)
    axes[-1].set_xlim(110.0, 950.0)
    plt.suptitle("Median Rest-Frame Spectra for Discovered Full-Survey Clusters from Scratch", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_rf_fig = RESULTS_DIR / "six_clusters_median_restframe_scratch.png"
    plt.savefig(out_rf_fig, dpi=200)
    plt.close()
    print(f"Saved rest-frame prototype spectra to {out_rf_fig}.")


if __name__ == "__main__":
    run_full_pipeline_scratch()

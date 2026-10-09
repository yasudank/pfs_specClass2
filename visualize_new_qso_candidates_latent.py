"""Visualize Latent Space Distribution and Physical Characteristics of
New QSO Candidates (DB=GALAXY & 2ch-Pred=QSO, N=145).

Includes:
1. 2D t-SNE Latent Space with explicit high-visibility highlighting of all 145 New QSO Candidates.
2. Zoomed-in view of the QSO Ridge / Galaxy-QSO transition neck.
3. 128D Cosine Similarity distributions (Similarity to Agreed QSO vs Agreed Galaxy centroid).
4. Redshift vs Predicted P(QSO) distribution.
"""
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR

BRAIN_DIR = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")


def main():
    t_start = time.time()
    print("=" * 70)
    print("Analyzing Latent Space Distribution of New QSO Candidates (N=145)")
    print("=" * 70)

    # 1. Load Data
    emb_path = RESULTS_DIR / "survey_embeddings_216k_2channel.npy"
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    pred_path = RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz"

    embeds = np.load(emb_path)
    meta = np.load(meta_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True)

    db_classes = meta['db_classes']
    db_redshifts = meta['db_redshifts']
    oof_preds = preds['oof_preds']
    oof_probs = preds['oof_probs']
    classes = list(preds['classes'])
    qso_idx = classes.index('QSO')

    # Mask definitions
    m_new_qso = (db_classes == 'GALAXY') & (oof_preds == 'QSO')          # 145
    m_agreed_qso = (db_classes == 'QSO') & (oof_preds == 'QSO')          # 451
    m_missed_qso = (db_classes == 'QSO') & (oof_preds == 'GALAXY')       # 626
    m_agreed_gal = (db_classes == 'GALAXY') & (oof_preds == 'GALAXY')    # 95,636
    m_agreed_star = (db_classes == 'STAR') & (oof_preds == 'STAR')       # 91,084

    idx_new_qso = np.where(m_new_qso)[0]
    idx_agreed_qso = np.where(m_agreed_qso)[0]
    idx_missed_qso = np.where(m_missed_qso)[0]

    np.random.seed(42)
    idx_bg_gal = np.random.choice(np.where(m_agreed_gal)[0], 4000, replace=False)
    idx_bg_star = np.random.choice(np.where(m_agreed_star)[0], 4000, replace=False)

    print(f"Sample breakdown:")
    print(f"  New QSO Candidates (DB: GALAXY, Pred: QSO): N = {len(idx_new_qso)}")
    print(f"  Agreed QSOs (DB: QSO, Pred: QSO):            N = {len(idx_agreed_qso)}")
    print(f"  Missed QSOs (DB: QSO, Pred: GALAXY):         N = {len(idx_missed_qso)}")
    print(f"  Background Galaxies:                         N = {len(idx_bg_gal)}")
    print(f"  Background Stars:                            N = {len(idx_bg_star)}")

    # 2. 128D Centroid Similarity Analysis
    def normalize(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    embeds_norm = normalize(embeds)
    proto_agreed_qso = normalize(np.mean(embeds_norm[idx_agreed_qso], axis=0, keepdims=True))[0]
    proto_agreed_gal = normalize(np.mean(embeds_norm[idx_bg_gal], axis=0, keepdims=True))[0]
    proto_agreed_star = normalize(np.mean(embeds_norm[idx_bg_star], axis=0, keepdims=True))[0]

    sim_new_to_qso = np.dot(embeds_norm[idx_new_qso], proto_agreed_qso)
    sim_new_to_gal = np.dot(embeds_norm[idx_new_qso], proto_agreed_gal)
    sim_agreed_qso_to_qso = np.dot(embeds_norm[idx_agreed_qso], proto_agreed_qso)
    sim_agreed_gal_to_gal = np.dot(embeds_norm[idx_bg_gal], proto_agreed_gal)

    # 3. Combined t-SNE Projection (9,222 objects)
    all_sample_idx = np.concatenate([
        idx_new_qso,       # 145
        idx_agreed_qso,    # 451
        idx_missed_qso,    # 626
        idx_bg_gal,        # 4000
        idx_bg_star        # 4000
    ])

    print(f"\nRunning t-SNE on {len(all_sample_idx)} targets...")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=1000, n_jobs=16)
    coords_2d = tsne.fit_transform(embeds[all_sample_idx])
    print(f"t-SNE completed in {time.time() - t0:.1f}s.")

    # Split coordinates
    n0 = 0
    n1 = len(idx_new_qso)
    n2 = n1 + len(idx_agreed_qso)
    n3 = n2 + len(idx_missed_qso)
    n4 = n3 + len(idx_bg_gal)
    n5 = n4 + len(idx_bg_star)

    c_new = coords_2d[n0:n1]
    c_agreed_q = coords_2d[n1:n2]
    c_missed_q = coords_2d[n2:n3]
    c_bg_gal = coords_2d[n3:n4]
    c_bg_star = coords_2d[n4:n5]

    # 4. Generate Comprehensive 4-Panel Figure
    fig = plt.figure(figsize=(19, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.22)

    # --- PANEL [A]: Full Latent Space Overview ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(c_bg_star[:, 0], c_bg_star[:, 1], c='#aec7e8', s=12, alpha=0.35, label=f'Agreed Stars (N={len(idx_bg_star)})')
    ax_a.scatter(c_bg_gal[:, 0], c_bg_gal[:, 1], c='#98df8a', s=12, alpha=0.35, label=f'Agreed Galaxies (N={len(idx_bg_gal)})')
    ax_a.scatter(c_missed_q[:, 0], c_missed_q[:, 1], c='#9467bd', s=26, alpha=0.65, label=f'Missed QSOs (DB: QSO, Pred: GAL, N={len(idx_missed_qso)})')
    ax_a.scatter(c_agreed_q[:, 0], c_agreed_q[:, 1], c='#d62728', s=35, alpha=0.85, label=f'Agreed QSOs (DB: QSO, Pred: QSO, N={len(idx_agreed_qso)})')
    
    # Highlight New QSO Candidates with prominent golden-orange markers
    ax_a.scatter(c_new[:, 0], c_new[:, 1], c='#ff7f0e', s=70, marker='D', edgecolors='black', linewidth=1.0, 
                 alpha=0.95, label=f'★ New QSO Candidates (DB: GALAXY, Pred: QSO, N={len(idx_new_qso)})', zorder=10)

    ax_a.set_title('[A] Full 2-Channel Latent Space (t-SNE Overview)\nNew QSO Candidates Populating the Upper QSO Ridge & Transition Neck', fontsize=12, fontweight='bold')
    ax_a.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax_a.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax_a.grid(True, linestyle='--', alpha=0.35)
    ax_a.legend(loc='lower left', fontsize=9.5, framealpha=0.92)

    # --- PANEL [B]: Zoomed-In View of QSO Ridge and Transition Zone ---
    ax_b = fig.add_subplot(gs[0, 1])
    # Filter for upper region (Dim 2 > 10)
    m_top_star = c_bg_star[:, 1] > 10
    m_top_gal = c_bg_gal[:, 1] > 10
    m_top_missed = c_missed_q[:, 1] > 10
    m_top_agreed = c_agreed_q[:, 1] > 10
    m_top_new = c_new[:, 1] > 10

    ax_b.scatter(c_bg_star[m_top_star, 0], c_bg_star[m_top_star, 1], c='#aec7e8', s=16, alpha=0.45)
    ax_b.scatter(c_bg_gal[m_top_gal, 0], c_bg_gal[m_top_gal, 1], c='#98df8a', s=16, alpha=0.45)
    ax_b.scatter(c_missed_q[m_top_missed, 0], c_missed_q[m_top_missed, 1], c='#9467bd', s=45, alpha=0.75, label='Missed QSOs')
    ax_b.scatter(c_agreed_q[m_top_agreed, 0], c_agreed_q[m_top_agreed, 1], c='#d62728', s=55, alpha=0.9, label='Agreed QSOs')
    ax_b.scatter(c_new[m_top_new, 0], c_new[m_top_new, 1], c='#ff7f0e', s=110, marker='D', edgecolors='black', 
                 linewidth=1.2, alpha=1.0, label='★ New QSO Candidates', zorder=12)

    ax_b.set_title('[B] Zoomed-In View: The Upper QSO Ridge & Galaxy Interface\nCandidates are physically embedded directly inside the QSO Cluster', fontsize=12, fontweight='bold')
    ax_b.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax_b.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax_b.set_ylim(15.0, 70.0)
    ax_b.grid(True, linestyle='--', alpha=0.35)
    ax_b.legend(loc='lower left', fontsize=10, framealpha=0.92)

    # --- PANEL [C]: 128D Cosine Similarity Distribution ---
    ax_c = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0.2, 0.95, 40)
    ax_c.hist(sim_new_to_qso, bins=bins, color='#ff7f0e', alpha=0.75, density=True, edgecolor='black', lw=0.8,
              label=f'New Candidates $\\rightarrow$ Agreed QSO Centroid (Median = {np.median(sim_new_to_qso):.3f})')
    ax_c.hist(sim_new_to_gal, bins=bins, color='#2ca02c', alpha=0.50, density=True, edgecolor='black', lw=0.8,
              label=f'New Candidates $\\rightarrow$ Agreed Galaxy Centroid (Median = {np.median(sim_new_to_gal):.3f})')
    
    ax_c.axvline(np.median(sim_new_to_qso), color='#d62728', linestyle='--', lw=2.0)
    ax_c.axvline(np.median(sim_new_to_gal), color='#1b7837', linestyle='--', lw=2.0)

    ax_c.set_title('[C] 128D Latent Cosine Similarity to Class Prototypes\n(Candidates have significantly higher similarity to QSOs than to Galaxies)', fontsize=12, fontweight='bold')
    ax_c.set_xlabel('Cosine Similarity in 128D Latent Space', fontsize=11)
    ax_c.set_ylabel('Probability Density', fontsize=11)
    ax_c.grid(True, linestyle='--', alpha=0.35)
    ax_c.legend(loc='upper left', fontsize=9.5, framealpha=0.92)

    # --- PANEL [D]: Redshift vs Predicted P(QSO) Probability ---
    ax_d = fig.add_subplot(gs[1, 1])
    z_new = db_redshifts[idx_new_qso]
    probs_new = oof_probs[idx_new_qso, qso_idx]

    valid_z_mask = np.isfinite(z_new) & (z_new > 0)
    sc = ax_d.scatter(z_new[valid_z_mask], probs_new[valid_z_mask], c=sim_new_to_qso[valid_z_mask], cmap='plasma', 
                      s=80, edgecolors='black', linewidth=0.8, vmin=0.4, vmax=0.85)
    cbar = plt.colorbar(sc, ax=ax_d)
    cbar.set_label('Similarity to QSO Centroid', fontsize=10.5)

    ax_d.axhline(0.5, color='gray', linestyle=':', lw=1.2, label='Classification Threshold P=0.5')
    ax_d.set_title('[D] New QSO Candidates: Redshift vs Model Probability P(QSO)\n(Points colored by 128D Cosine Similarity to Agreed QSO Centroid)', fontsize=12, fontweight='bold')
    ax_d.set_xlabel('Database Redshift ($z_{DB}$)', fontsize=11)
    ax_d.set_ylabel('Random Forest Model Probability P(QSO)', fontsize=11)
    ax_d.set_ylim(0.3, 1.0)
    ax_d.grid(True, linestyle='--', alpha=0.35)
    ax_d.legend(loc='lower right', fontsize=10, framealpha=0.92)

    plt.suptitle("Latent Parameter Space Analysis: 'New QSO Candidates' (DB: GALAXY & Pred: QSO, N=145)", 
                 fontsize=14, fontweight='bold', y=0.995)

    out_fig = RESULTS_DIR / "new_qso_candidates_latent_distribution.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved figure to {out_fig}.")

    brain_fig = BRAIN_DIR / "new_qso_candidates_latent_distribution.png"
    shutil.copy2(out_fig, brain_fig)
    print(f"Copied figure to {brain_fig}.")

    print(f"Analysis completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

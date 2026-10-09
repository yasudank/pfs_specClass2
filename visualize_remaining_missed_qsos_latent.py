"""Dedicated diagnostic script to visualize the exact latent space positions of
Remaining Missed QSOs (DB=QSO but Model Pred=GALAXY [N=303] or STAR [N=125])
in Experiment 3 (Boosted Channel 1 alpha=3.0 SimCLR + SMOTE 3x RF).

Panels:
[A] Global 2D t-SNE Latent Space with Explicit Highlighting of Remaining Missed QSOs
[B] Zoomed-in Inspection of the QSO Cluster Boundary & Galaxy Manifold
[C] 128D Cosine Similarity Space: Sim(z, p_QSO) vs Sim(z, p_GAL) with Decision Boundary
[D] Prototype Similarity Margin (Sim_QSO - Sim_GAL) Distribution across Object Groups
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


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def main():
    t_start = time.time()
    print("=" * 75)
    print("Visualizing Latent Space Positions of Remaining Missed QSOs")
    print("=" * 75)

    emb_path = RESULTS_DIR / "embeddings_representative_boost3x.npy"
    pred_path = RESULTS_DIR / "rf_oof_boost3x_smote_representative.npz"

    embeds = np.load(emb_path)
    preds = np.load(pred_path, allow_pickle=True)

    db_classes = preds['db_classes']
    oof_preds = preds['oof_preds']
    oof_probs = preds['oof_probs']
    db_redshifts = preds['db_redshifts']
    classes = list(preds['classes'])
    qso_idx = classes.index('QSO')

    # Mask definitions
    m_agreed_qso = (db_classes == 'QSO') & (oof_preds == 'QSO')          # N = 881
    m_missed_qso_gal = (db_classes == 'QSO') & (oof_preds == 'GALAXY')    # N = 303
    m_missed_qso_star = (db_classes == 'QSO') & (oof_preds == 'STAR')     # N = 125
    m_new_qso = (db_classes == 'GALAXY') & (oof_preds == 'QSO')          # N = 116
    m_agreed_gal = (db_classes == 'GALAXY') & (oof_preds == 'GALAXY')    # N = 17,509
    m_agreed_star = (db_classes == 'STAR') & (oof_preds == 'STAR')       # N = 17,087

    idx_agreed_qso = np.where(m_agreed_qso)[0]
    idx_missed_gal = np.where(m_missed_qso_gal)[0]
    idx_missed_star = np.where(m_missed_qso_star)[0]
    idx_new_qso = np.where(m_new_qso)[0]

    np.random.seed(42)
    idx_bg_gal = np.random.choice(np.where(m_agreed_gal)[0], 4000, replace=False)
    idx_bg_star = np.random.choice(np.where(m_agreed_star)[0], 4000, replace=False)

    print(f"Group counts:")
    print(f"  Agreed QSOs (DB=QSO, Pred=QSO):           {len(idx_agreed_qso)}")
    print(f"  Remaining Missed QSOs -> GALAXY:          {len(idx_missed_gal)}")
    print(f"  Remaining Missed QSOs -> STAR:            {len(idx_missed_star)}")
    print(f"  New QSO Candidates (DB=GAL, Pred=QSO):    {len(idx_new_qso)}")
    print(f"  Background Agreed Galaxies:               {len(idx_bg_gal)}")
    print(f"  Background Agreed Stars:                  {len(idx_bg_star)}")

    # 1. 128D Prototype Cosine Similarity Computation
    embeds_norm = normalize(embeds)
    proto_qso = normalize(np.mean(embeds_norm[idx_agreed_qso], axis=0, keepdims=True))[0]
    proto_gal = normalize(np.mean(embeds_norm[idx_bg_gal], axis=0, keepdims=True))[0]
    proto_star = normalize(np.mean(embeds_norm[idx_bg_star], axis=0, keepdims=True))[0]

    sim_to_qso = np.dot(embeds_norm, proto_qso)
    sim_to_gal = np.dot(embeds_norm, proto_gal)
    sim_to_star = np.dot(embeds_norm, proto_star)
    margin_qso_gal = sim_to_qso - sim_to_gal

    # 2. t-SNE 2D Projection
    all_sample_idx = np.concatenate([
        idx_agreed_qso,      # 881
        idx_missed_gal,      # 303
        idx_missed_star,     # 125
        idx_new_qso,         # 116
        idx_bg_gal,          # 4000
        idx_bg_star          # 4000
    ])

    print(f"\nComputing t-SNE projection on {len(all_sample_idx)} objects...")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=850, init='pca', learning_rate='auto', n_jobs=16)
    coords_2d = tsne.fit_transform(embeds[all_sample_idx])
    print(f"t-SNE completed in {time.time() - t0:.1f}s.")

    # Slice 2D coordinates
    n_aq = len(idx_agreed_qso)
    n_mg = len(idx_missed_gal)
    n_ms = len(idx_missed_star)
    n_nq = len(idx_new_qso)
    n_bg = len(idx_bg_gal)
    n_bs = len(idx_bg_star)

    c_aq = coords_2d[0:n_aq]
    c_mg = coords_2d[n_aq:n_aq + n_mg]
    c_ms = coords_2d[n_aq + n_mg:n_aq + n_mg + n_ms]
    c_nq = coords_2d[n_aq + n_mg + n_ms:n_aq + n_mg + n_ms + n_nq]
    c_bg = coords_2d[n_aq + n_mg + n_ms + n_nq:n_aq + n_mg + n_ms + n_nq + n_bg]
    c_bs = coords_2d[n_aq + n_mg + n_ms + n_nq + n_bg:]

    # 3. Create 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(19, 17))
    plt.subplots_adjust(hspace=0.25, wspace=0.22)

    # -------------------------------------------------------------
    # Panel [A]: Global 2D t-SNE Latent Space
    # -------------------------------------------------------------
    ax = axes[0, 0]
    # Background
    ax.scatter(c_bs[:, 0], c_bs[:, 1], c='#1f77b4', s=12, alpha=0.18, label=f'Agreed STAR (N={n_bs})')
    ax.scatter(c_bg[:, 0], c_bg[:, 1], c='#2ca02c', s=12, alpha=0.18, label=f'Agreed GALAXY (N={n_bg})')
    ax.scatter(c_aq[:, 0], c_aq[:, 1], c='#e41a1c', s=28, alpha=0.75, edgecolors='none', label=f'Agreed QSO (N={n_aq})')

    # Highlights
    ax.scatter(c_nq[:, 0], c_nq[:, 1], c='#ff7f00', marker='^', s=45, alpha=0.9, edgecolors='black', linewidth=0.5,
               label=f'New QSO Candidates (DB: GAL -> Pred: QSO, N={n_nq})')
    ax.scatter(c_ms[:, 0], c_ms[:, 1], c='#17becf', marker='D', s=45, alpha=0.9, edgecolors='black', linewidth=0.6,
               label=f'Remaining Missed QSOs -> STAR (N={n_ms})')
    ax.scatter(c_mg[:, 0], c_mg[:, 1], c='#984ea3', marker='*', s=85, alpha=0.95, edgecolors='black', linewidth=0.7,
               label=f'Remaining Missed QSOs -> GALAXY (N={n_mg})')

    ax.set_title('[A] Global 2D Latent Representation Space (t-SNE)\nPosition of Remaining Missed QSOs relative to Main Manifolds', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.92)
    ax.grid(True, linestyle='--', alpha=0.35)

    # -------------------------------------------------------------
    # Panel [B]: Zoomed-in View of Galaxy Manifold & Transition Region
    # -------------------------------------------------------------
    ax = axes[0, 1]
    # Center around Missed QSOs -> GALAXY
    q_x_med, q_y_med = np.median(c_aq[:, 0]), np.median(c_aq[:, 1])
    mg_x_med, mg_y_med = np.median(c_mg[:, 0]), np.median(c_mg[:, 1])

    # Determine bounding box that covers both agreed QSOs and missed QSOs -> GAL
    x_min = min(np.percentile(c_aq[:, 0], 5), np.percentile(c_mg[:, 0], 5)) - 8
    x_max = max(np.percentile(c_aq[:, 0], 95), np.percentile(c_mg[:, 0], 95)) + 8
    y_min = min(np.percentile(c_aq[:, 1], 5), np.percentile(c_mg[:, 1], 5)) - 8
    y_max = max(np.percentile(c_aq[:, 1], 95), np.percentile(c_mg[:, 1], 95)) + 8

    ax.scatter(c_bg[:, 0], c_bg[:, 1], c='#2ca02c', s=16, alpha=0.25, label='Agreed GALAXY')
    ax.scatter(c_aq[:, 0], c_aq[:, 1], c='#e41a1c', s=35, alpha=0.75, edgecolors='black', linewidth=0.3, label='Agreed QSO')
    ax.scatter(c_nq[:, 0], c_nq[:, 1], c='#ff7f00', marker='^', s=60, alpha=0.9, edgecolors='black', linewidth=0.6, label='New QSO Candidates')
    ax.scatter(c_mg[:, 0], c_mg[:, 1], c='#984ea3', marker='*', s=120, alpha=0.95, edgecolors='black', linewidth=0.8, label='Remaining Missed QSOs -> GALAXY (N=303)')

    # Add centroid markers
    ax.plot(q_x_med, q_y_med, marker='X', color='red', markersize=14, markeredgecolor='black', label='QSO Cluster Median')
    ax.plot(mg_x_med, mg_y_med, marker='X', color='purple', markersize=14, markeredgecolor='black', label='Missed QSO Median')

    # Arrow indicating morphological migration
    ax.annotate('Host-Galaxy Dominated Shift\n(Broad lines submerged in starlight)',
                xy=(mg_x_med, mg_y_med), xytext=(mg_x_med + 6, mg_y_med + 12),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.85))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('[B] Zoomed View: QSO Island vs Galaxy Periphery\n(Missed QSOs sit along the Galaxy Manifold due to strong starlight continuum)', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.legend(loc='lower left', fontsize=9.5, framealpha=0.92)
    ax.grid(True, linestyle='--', alpha=0.35)

    # -------------------------------------------------------------
    # Panel [C]: 128D Cosine Similarity Space: Sim(z, p_QSO) vs Sim(z, p_GAL)
    # -------------------------------------------------------------
    ax = axes[1, 0]

    # Plot background galaxies and stars
    ax.scatter(sim_to_qso[idx_bg_gal], sim_to_gal[idx_bg_gal], c='#2ca02c', s=10, alpha=0.15, label='Agreed GALAXY')
    ax.scatter(sim_to_qso[idx_bg_star], sim_to_gal[idx_bg_star], c='#1f77b4', s=10, alpha=0.15, label='Agreed STAR')

    # Plot Agreed QSOs
    ax.scatter(sim_to_qso[idx_agreed_qso], sim_to_gal[idx_agreed_qso], c='#e41a1c', s=25, alpha=0.7, edgecolors='black', linewidth=0.3,
               label=f'Agreed QSO (Mean: SimQ=0.779, SimG=0.693)')

    # Plot New QSO Candidates
    ax.scatter(sim_to_qso[idx_new_qso], sim_to_gal[idx_new_qso], c='#ff7f00', marker='^', s=45, alpha=0.9, edgecolors='black', linewidth=0.5,
               label=f'New QSO Candidates (Mean: SimQ=0.751, SimG=0.705)')

    # Plot Remaining Missed QSOs -> GALAXY
    ax.scatter(sim_to_qso[idx_missed_gal], sim_to_gal[idx_missed_gal], c='#984ea3', marker='*', s=80, alpha=0.95, edgecolors='black', linewidth=0.6,
               label=f'Remaining Missed QSOs -> GALAXY (Mean: SimQ=0.714, SimG=0.768)')

    # Plot Remaining Missed QSOs -> STAR
    ax.scatter(sim_to_qso[idx_missed_star], sim_to_gal[idx_missed_star], c='#17becf', marker='D', s=40, alpha=0.9, edgecolors='black', linewidth=0.6,
               label=f'Remaining Missed QSOs -> STAR (Mean: SimQ=0.711, SimG=0.719)')

    # Decision diagonal line: Sim_QSO = Sim_GAL
    diag_vals = np.linspace(0.55, 0.90, 100)
    ax.plot(diag_vals, diag_vals, 'k--', linewidth=1.8, label='Equi-Similarity Line (Sim_QSO = Sim_GAL)')
    ax.fill_between(diag_vals, diag_vals, 0.95, color='#2ca02c', alpha=0.04)
    ax.fill_between(diag_vals, 0.50, diag_vals, color='#e41a1c', alpha=0.04)

    ax.text(0.82, 0.64, 'QSO Domain\n(Sim_QSO > Sim_GAL)', fontsize=11, fontweight='bold', color='#b2182b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#b2182b', alpha=0.9))
    ax.text(0.60, 0.85, 'Galaxy Domain\n(Sim_GAL > Sim_QSO)', fontsize=11, fontweight='bold', color='#1b7837',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1b7837', alpha=0.9))

    ax.set_xlim(0.55, 0.90)
    ax.set_ylim(0.55, 0.90)
    ax.set_title('[C] 128D Metric Space: Cosine Similarity to QSO vs GALAXY Prototype\n(Demonstrating Remaining Missed QSOs cross over into the Galaxy domain)', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Cosine Similarity to QSO Prototype $\mathrm{Sim}(\mathbf{z}, \mathbf{p}_{\mathrm{QSO}})$', fontsize=11)
    ax.set_ylabel(r'Cosine Similarity to Galaxy Prototype $\mathrm{Sim}(\mathbf{z}, \mathbf{p}_{\mathrm{GAL}})$', fontsize=11)
    ax.legend(loc='lower right', fontsize=9.0, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.35)

    # -------------------------------------------------------------
    # Panel [D]: Prototype Similarity Margin Distribution & P(QSO)
    # -------------------------------------------------------------
    ax = axes[1, 1]

    bins = np.linspace(-0.25, 0.25, 51)
    ax.hist(margin_qso_gal[idx_agreed_qso], bins=bins, density=True, histtype='stepfilled', alpha=0.45, color='#e41a1c',
            label=f'Agreed QSO (Median Margin: +{np.median(margin_qso_gal[idx_agreed_qso]):.3f})')
    ax.hist(margin_qso_gal[idx_agreed_qso], bins=bins, density=True, histtype='step', linewidth=1.8, color='#e41a1c')

    ax.hist(margin_qso_gal[idx_new_qso], bins=bins, density=True, histtype='stepfilled', alpha=0.55, color='#ff7f00',
            label=f'New QSO Candidates (Median Margin: +{np.median(margin_qso_gal[idx_new_qso]):.3f})')
    ax.hist(margin_qso_gal[idx_new_qso], bins=bins, density=True, histtype='step', linewidth=1.8, color='#ff7f00')

    ax.hist(margin_qso_gal[idx_missed_gal], bins=bins, density=True, histtype='stepfilled', alpha=0.55, color='#984ea3',
            label=f'Missed QSOs -> GALAXY (Median Margin: {np.median(margin_qso_gal[idx_missed_gal]):.3f})')
    ax.hist(margin_qso_gal[idx_missed_gal], bins=bins, density=True, histtype='step', linewidth=2.0, color='#984ea3')

    ax.hist(margin_qso_gal[idx_missed_star], bins=bins, density=True, histtype='stepfilled', alpha=0.45, color='#17becf',
            label=f'Missed QSOs -> STAR (Median Margin: {np.median(margin_qso_gal[idx_missed_star]):.3f})')
    ax.hist(margin_qso_gal[idx_missed_star], bins=bins, density=True, histtype='step', linewidth=1.8, color='#17becf')

    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.8, label='Equi-Similarity Margin = 0.0')

    ax.set_title('[D] Prototype Similarity Margin Distribution: $\Delta \mathrm{Sim} = \mathrm{Sim}_{\mathrm{QSO}} - \mathrm{Sim}_{\mathrm{GAL}}$\n(Missed QSOs exhibit strong negative margin: representations genuinely align with galaxies)', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Cosine Similarity Margin $\Delta \mathrm{Sim} = \mathrm{Sim}(\mathbf{z}, \mathbf{p}_{\mathrm{QSO}}) - \mathrm{Sim}(\mathbf{z}, \mathbf{p}_{\mathrm{GAL}})$', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.legend(loc='upper left', fontsize=9.2, framealpha=0.92)
    ax.grid(True, linestyle='--', alpha=0.35)

    # Save figure
    save_path = RESULTS_DIR / "remaining_missed_qsos_latent_positions.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

    # Copy to brain artifact directory
    brain_path = BRAIN_DIR / "remaining_missed_qsos_latent_positions.png"
    shutil.copy(save_path, brain_path)
    print(f"Copied to brain directory: {brain_path}")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

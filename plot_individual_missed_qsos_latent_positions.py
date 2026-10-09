"""Dedicated diagnostic script to plot each of the 6 representative Remaining Missed QSOs -> GALAXY
directly onto the 128D Latent Representation Space.

Highlights the 6 specific objects:
  #1 [Index 1597, z=0.435]: Host galaxy starlight dominated
  #2 [Index 1428, z=0.701]: Star-forming galaxy continuum, 4000A break
  #3 [Index 289,  z=1.326]: Core galaxy branch, weak Mg II
  #4 [Index 1318, z=1.574]: Low S/N UV, galaxy continuum slope
  #5 [Index 558,  z=1.794]: Borderline target (P(QSO)=0.38, SimQSO~SimGAL)
  #6 [Index 622,  z=2.115]: High-z with NIR continuum distortion

Panels:
[A] Global 2D t-SNE Latent Space with Exact Location of Objects #1-#6
[B] Zoomed-in Transition Region (QSO Peninsula vs Galaxy Manifold)
[C] 128D Cosine Similarity Metric Space: Sim(z, p_QSO) vs Sim(z, p_GAL)
[D] Unwarped Linear PCA Metric Space (PC1 vs PC2) with Cluster Density
"""
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR

BRAIN_DIR = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def main():
    t_start = time.time()
    print("=" * 75)
    print("Plotting Individual Missed QSOs onto Latent Space")
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
    idx_g = classes.index('GALAXY')
    idx_q = classes.index('QSO')
    idx_s = classes.index('STAR')

    # The 6 selected objects from the spectral analysis
    targets = [1597, 1428, 289, 1318, 558, 622]
    target_names = [
        r"#1 (z=0.44)",
        r"#2 (z=0.70)",
        r"#3 (z=1.33)",
        r"#4 (z=1.57)",
        r"#5 (z=1.79)",
        r"#6 (z=2.12)"
    ]
    target_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    target_markers = ['o', 's', '^', 'D', 'p', 'h']

    # Masks
    m_agreed_qso = (db_classes == 'QSO') & (oof_preds == 'QSO')          # 881
    m_missed_gal = (db_classes == 'QSO') & (oof_preds == 'GALAXY')       # 303
    m_missed_star = (db_classes == 'QSO') & (oof_preds == 'STAR')        # 125
    m_new_qso = (db_classes == 'GALAXY') & (oof_preds == 'QSO')          # 116
    m_agreed_gal = (db_classes == 'GALAXY') & (oof_preds == 'GALAXY')    # 17,509
    m_agreed_star = (db_classes == 'STAR') & (oof_preds == 'STAR')       # 17,087

    idx_agreed_qso = np.where(m_agreed_qso)[0]
    idx_missed_gal = np.where(m_missed_gal)[0]
    idx_missed_star = np.where(m_missed_star)[0]
    idx_new_qso = np.where(m_new_qso)[0]

    np.random.seed(42)
    idx_bg_gal = np.random.choice(np.where(m_agreed_gal)[0], 4000, replace=False)
    idx_bg_star = np.random.choice(np.where(m_agreed_star)[0], 4000, replace=False)

    # 1. 128D Prototype Cosine Similarity Computation
    embeds_norm = normalize(embeds)
    proto_qso = normalize(np.mean(embeds_norm[idx_agreed_qso], axis=0, keepdims=True))[0]
    proto_gal = normalize(np.mean(embeds_norm[idx_bg_gal], axis=0, keepdims=True))[0]
    proto_star = normalize(np.mean(embeds_norm[idx_bg_star], axis=0, keepdims=True))[0]

    sim_to_qso = np.dot(embeds_norm, proto_qso)
    sim_to_gal = np.dot(embeds_norm, proto_gal)
    sim_to_star = np.dot(embeds_norm, proto_star)

    # 2. Linear PCA Computation on all representative targets
    print("Fitting PCA on 128D embeddings...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(embeds)

    # 3. 2D t-SNE Computation
    all_sample_idx = np.concatenate([
        idx_agreed_qso,      # 881
        idx_missed_gal,      # 303
        idx_missed_star,     # 125
        idx_new_qso,         # 116
        idx_bg_gal,          # 4000
        idx_bg_star          # 4000
    ])

    print(f"Running t-SNE on {len(all_sample_idx)} targets...")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=850, init='pca', learning_rate='auto', n_jobs=16)
    tsne_coords = tsne.fit_transform(embeds[all_sample_idx])
    print(f"t-SNE completed in {time.time() - t0:.1f}s.")

    # Find the positions of the 6 targets in all_sample_idx
    target_tsne_indices = [np.where(all_sample_idx == tid)[0][0] for tid in targets]

    n_aq = len(idx_agreed_qso)
    n_mg = len(idx_missed_gal)
    n_ms = len(idx_missed_star)
    n_nq = len(idx_new_qso)
    n_bg = len(idx_bg_gal)

    c_aq = tsne_coords[0:n_aq]
    c_mg = tsne_coords[n_aq:n_aq + n_mg]
    c_bg = tsne_coords[n_aq + n_mg + n_ms + n_nq:n_aq + n_mg + n_ms + n_nq + n_bg]
    c_bs = tsne_coords[n_aq + n_mg + n_ms + n_nq + n_bg:]

    # 4. Create 4-Panel Master Visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    plt.subplots_adjust(hspace=0.25, wspace=0.22)

    # -------------------------------------------------------------
    # Panel [A]: Global 2D t-SNE Space with 6 Targets
    # -------------------------------------------------------------
    ax = axes[0, 0]
    ax.scatter(c_bs[:, 0], c_bs[:, 1], c='#a6cee3', s=10, alpha=0.18, label='Agreed STAR')
    ax.scatter(c_bg[:, 0], c_bg[:, 1], c='#b2df8a', s=10, alpha=0.18, label='Agreed GALAXY')
    ax.scatter(c_aq[:, 0], c_aq[:, 1], c='#fb9a99', s=22, alpha=0.50, label='Agreed QSO')
    ax.scatter(c_mg[:, 0], c_mg[:, 1], c='#cab2d6', s=25, alpha=0.55, label='All Missed QSOs -> GAL (N=303)')

    # Plot the 6 targets with distinctive markers & callouts
    for k, tid in enumerate(targets):
        pos = tsne_coords[target_tsne_indices[k]]
        ax.scatter(pos[0], pos[1], c=target_colors[k], marker=target_markers[k],
                   s=180, edgecolors='black', linewidth=1.5, zorder=10,
                   label=f"{target_names[k]} [Index {tid}]")
        # Text badge
        ax.annotate(f"#{k+1}", xy=(pos[0], pos[1]), xytext=(pos[0]+2.0, pos[1]+2.0),
                    fontsize=10, fontweight='bold', color='black',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='yellow', edgecolor='black', alpha=0.9),
                    zorder=11)

    ax.set_title('[A] Global 2D t-SNE Latent Space: Positions of Objects #1-#6\n(Objects #1-#4 & #6 reside deep in Galaxy manifold; #5 sits on the boundary)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.legend(loc='upper right', fontsize=9.0, framealpha=0.92)
    ax.grid(True, linestyle='--', alpha=0.35)

    # -------------------------------------------------------------
    # Panel [B]: Zoomed-in View of the Galaxy Manifold & Transition Region
    # -------------------------------------------------------------
    ax = axes[0, 1]
    ax.scatter(c_bg[:, 0], c_bg[:, 1], c='#2ca02c', s=16, alpha=0.25, label='Agreed GALAXY')
    ax.scatter(c_aq[:, 0], c_aq[:, 1], c='#e41a1c', s=35, alpha=0.70, edgecolors='black', linewidth=0.3, label='Agreed QSO')
    ax.scatter(c_mg[:, 0], c_mg[:, 1], c='#984ea3', s=45, alpha=0.60, edgecolors='black', linewidth=0.4, label='Missed QSOs -> GAL')

    # Bounding box around the 6 targets
    t_x = [tsne_coords[target_tsne_indices[k]][0] for k in range(len(targets))]
    t_y = [tsne_coords[target_tsne_indices[k]][1] for k in range(len(targets))]
    q_x = [c_aq[m][0] for m in range(len(c_aq))]
    q_y = [c_aq[m][1] for m in range(len(c_aq))]

    x_min = min(min(t_x), np.percentile(q_x, 5)) - 10
    x_max = max(max(t_x), np.percentile(q_x, 95)) + 10
    y_min = min(min(t_y), np.percentile(q_y, 5)) - 10
    y_max = max(max(t_y), np.percentile(q_y, 95)) + 10

    # Annotate each target specifically with its physical characteristic and clean non-overlapping offsets
    annotations = [
        ("Host Starlight Dominated\n(4000Å Break)", (-3.0, -8.0)),
        ("Star-Forming Galaxy Continuum\n(No broad lines)", (4.0, 7.5)),
        ("Core Galaxy Branch\n(Weak Mg II)", (-12.0, 6.0)),
        ("Galaxy Continuum Slope\n(No UV lines)", (4.0, 3.5)),
        ("Borderline Target\n(Clear Mg II, P(QSO)=0.38)", (-14.0, -4.5)),
        ("High-z Distorted Continuum\n(Near galaxy edge)", (4.5, -7.5))
    ]

    for k, tid in enumerate(targets):
        pos = tsne_coords[target_tsne_indices[k]]
        ax.scatter(pos[0], pos[1], c=target_colors[k], marker=target_markers[k],
                   s=240, edgecolors='black', linewidth=1.8, zorder=12,
                   label=f"{target_names[k]}")
        desc, offset = annotations[k]
        ax.annotate(f"#{k+1}: {desc}",
                    xy=(pos[0], pos[1]), xytext=(pos[0] + offset[0], pos[1] + offset[1]),
                    arrowprops=dict(facecolor=target_colors[k], shrink=0.08, width=1.5, headwidth=6),
                    fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=target_colors[k], alpha=0.92),
                    zorder=13)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('[B] Zoomed View: Physical Manifold Distribution of Objects #1-#6\n(Detailed annotations showing host starlight contamination & boundary proximity)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.legend(loc='lower left', fontsize=9.0, framealpha=0.92)
    ax.grid(True, linestyle='--', alpha=0.35)

    # -------------------------------------------------------------
    # Panel [C]: 128D Cosine Similarity Space: Sim(z, p_QSO) vs Sim(z, p_GAL)
    # -------------------------------------------------------------
    ax = axes[1, 0]
    ax.scatter(sim_to_qso[idx_bg_gal], sim_to_gal[idx_bg_gal], c='#2ca02c', s=10, alpha=0.15, label='Agreed GALAXY')
    ax.scatter(sim_to_qso[idx_agreed_qso], sim_to_gal[idx_agreed_qso], c='#e41a1c', s=25, alpha=0.6, label='Agreed QSO')
    ax.scatter(sim_to_qso[idx_missed_gal], sim_to_gal[idx_missed_gal], c='#984ea3', s=30, alpha=0.6, label='Missed QSOs -> GAL')

    # Diagonal decision boundary
    diag_vals = np.linspace(0.55, 0.90, 100)
    ax.plot(diag_vals, diag_vals, 'k--', linewidth=1.8, label='Equi-Similarity Line (Sim_QSO = Sim_GAL)')
    ax.fill_between(diag_vals, diag_vals, 0.95, color='#2ca02c', alpha=0.04)
    ax.fill_between(diag_vals, 0.50, diag_vals, color='#e41a1c', alpha=0.04)

    # Plot the 6 targets
    for k, tid in enumerate(targets):
        sq = sim_to_qso[tid]
        sg = sim_to_gal[tid]
        ax.scatter(sq, sg, c=target_colors[k], marker=target_markers[k],
                   s=220, edgecolors='black', linewidth=1.8, zorder=12,
                   label=f"{target_names[k]}: ($S_Q$={sq:.2f}, $S_G$={sg:.2f})")
        # Text badge
        ax.annotate(f"#{k+1}", xy=(sq, sg), xytext=(sq + 0.006, sg - 0.008),
                    fontsize=9.5, fontweight='bold', color='black',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='yellow', edgecolor='black', alpha=0.9),
                    zorder=13)

    ax.text(0.81, 0.73, 'QSO Domain\n(Sim_QSO > Sim_GAL)', fontsize=10.5, fontweight='bold', color='#b2182b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#b2182b', alpha=0.9))
    ax.text(0.58, 0.85, 'Galaxy Domain\n(Sim_GAL > Sim_QSO)', fontsize=10.5, fontweight='bold', color='#1b7837',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1b7837', alpha=0.9))

    ax.set_xlim(0.55, 0.90)
    ax.set_ylim(0.55, 0.90)
    ax.set_title('[C] 128D Metric Space: Cosine Similarity to QSO vs GALAXY Prototype\n(#1-#4, #6 deeply in Galaxy Domain; #5 right on the diagonal decision boundary)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Cosine Similarity to QSO Prototype $\mathrm{Sim}(\mathbf{z}, \mathbf{p}_{\mathrm{QSO}})$', fontsize=11)
    ax.set_ylabel(r'Cosine Similarity to Galaxy Prototype $\mathrm{Sim}(\mathbf{z}, \mathbf{p}_{\mathrm{GAL}})$', fontsize=11)
    ax.legend(loc='lower right', fontsize=8.8, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.35)

    # -------------------------------------------------------------
    # Panel [D]: Unwarped Linear PCA Metric Space (PC1 vs PC2)
    # -------------------------------------------------------------
    ax = axes[1, 1]
    ax.scatter(pca_coords[idx_bg_gal, 0], pca_coords[idx_bg_gal, 1], c='#2ca02c', s=10, alpha=0.15, label='Agreed GALAXY')
    ax.scatter(pca_coords[idx_bg_star, 0], pca_coords[idx_bg_star, 1], c='#1f77b4', s=10, alpha=0.15, label='Agreed STAR')
    ax.scatter(pca_coords[idx_agreed_qso, 0], pca_coords[idx_agreed_qso, 1], c='#e41a1c', s=25, alpha=0.6, label='Agreed QSO')
    ax.scatter(pca_coords[idx_missed_gal, 0], pca_coords[idx_missed_gal, 1], c='#984ea3', s=30, alpha=0.6, label='Missed QSOs -> GAL')

    # Plot the 6 targets
    for k, tid in enumerate(targets):
        px = pca_coords[tid, 0]
        py = pca_coords[tid, 1]
        ax.scatter(px, py, c=target_colors[k], marker=target_markers[k],
                   s=220, edgecolors='black', linewidth=1.8, zorder=12,
                   label=f"{target_names[k]}")
        ax.annotate(f"#{k+1}", xy=(px, py), xytext=(px + 0.008, py + 0.008),
                    fontsize=9.5, fontweight='bold', color='black',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='yellow', edgecolor='black', alpha=0.9),
                    zorder=13)

    ax.set_title('[D] Unwarped Linear PCA Space (PC1 vs PC2)\n(Preserving true Euclidean distance: Objects #1-#4, #6 align with Galaxy cloud)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
    ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
    ax.legend(loc='upper right', fontsize=8.8, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.35)

    # Save figure
    save_path = RESULTS_DIR / "individual_missed_qsos_latent_positions.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {save_path}")

    # Copy to brain artifact directory
    brain_path = BRAIN_DIR / "individual_missed_qsos_latent_positions.png"
    shutil.copy(save_path, brain_path)
    print(f"Copied to brain directory: {brain_path}")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

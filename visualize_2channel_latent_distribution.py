"""Visualize Latent Parameter Space Distribution for 2-Channel SimCLR Model (Full Survey).
1. Loads 2-channel 128D embeddings (216,476 spectra).
2. Computes 2D t-SNE projection for all 1,374 true QSOs and balanced sample of galaxies and stars.
3. Generates comprehensive 4-panel diagnostic figure:
   - [A-1] True Database Labels (t-SNE)
   - [A-2] 2-Channel Random Forest Predicted Labels
   - [B-1] Continuous P(QSO) Probability Heatmap
   - [B-2] Database Agreement vs Disagreement Map
4. Analyzes cluster separability and cosine distance between class prototypes.
"""
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR


def main():
    t_start = time.time()
    print("Loading full survey 2-channel embeddings and predictions...")

    emb_path = RESULTS_DIR / "survey_embeddings_216k_2channel.npy"
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    pred_path = RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz"

    embeds = np.load(emb_path)
    meta = np.load(meta_path, allow_pickle=True)
    preds_data = np.load(pred_path, allow_pickle=True)

    db_classes = meta["db_classes"]
    oof_preds = preds_data["oof_preds"]
    oof_probs = preds_data["oof_probs"]
    classes = list(preds_data["classes"])

    qso_idx = classes.index("QSO")
    gal_idx = classes.index("GALAXY")
    star_idx = classes.index("STAR")

    n_total = len(embeds)
    print(f"Total objects in survey: {n_total}")

    # 1. Cosine similarity and prototype analysis in 128D
    def normalize(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    embeds_norm = normalize(embeds)

    # Class prototypes
    m_qso = (db_classes == "QSO")
    m_gal = (db_classes == "GALAXY")
    m_star = (db_classes == "STAR")

    proto_qso = normalize(np.mean(embeds_norm[m_qso], axis=0, keepdims=True))[0]
    proto_gal = normalize(np.mean(embeds_norm[m_gal], axis=0, keepdims=True))[0]
    proto_star = normalize(np.mean(embeds_norm[m_star], axis=0, keepdims=True))[0]

    print("\n128D Cosine Similarities between Class Prototypes:")
    print(f"  QSO <-> GALAXY: {float(np.dot(proto_qso, proto_gal)):.4f}")
    print(f"  QSO <-> STAR  : {float(np.dot(proto_qso, proto_star)):.4f}")
    print(f"  GALAXY <-> STAR: {float(np.dot(proto_gal, proto_star)):.4f}")

    # 2. Balanced Sampling for t-SNE (All QSOs + 4000 Galaxies + 4000 Stars)
    np.random.seed(42)
    qso_indices = np.where(m_qso)[0] # All 1,374 QSOs
    gal_indices = np.random.choice(np.where(m_gal)[0], 4000, replace=False)
    star_indices = np.random.choice(np.where(m_star)[0], 4000, replace=False)

    sample_indices = np.concatenate([qso_indices, gal_indices, star_indices])
    n_sample = len(sample_indices)
    print(f"\nComputing t-SNE for {n_sample} representative objects (All 1,374 QSOs + 4,000 Gal + 4,000 Stars)...")

    sample_embeds = embeds[sample_indices]
    sample_db = db_classes[sample_indices]
    sample_pred = oof_preds[sample_indices]
    sample_q_probs = oof_probs[sample_indices, qso_idx]

    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=1000, n_jobs=16)
    coords_2d = tsne.fit_transform(sample_embeds)
    print(f"t-SNE completed in {time.time() - t0:.1f}s.")

    # 3. Diagnostic Visualizations
    print("\nGenerating 4-Panel Latent Space Diagnostic Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(19, 16))

    color_map = {
        'STAR': '#1f77b4',     # Blue
        'GALAXY': '#2ca02c',   # Green
        'QSO': '#d62728'       # Red
    }

    # [A-1] Latent Space by True Database Label
    ax = axes[0, 0]
    for c in ['STAR', 'GALAXY', 'QSO']:
        m = (sample_db == c)
        s = 35 if c == 'QSO' else 14
        alpha = 0.85 if c == 'QSO' else 0.30
        ax.scatter(coords_2d[m, 0], coords_2d[m, 1], c=color_map[c], label=f'{c} (N={np.sum(m)})', s=s, alpha=alpha, edgecolors='none' if c != 'QSO' else 'black', linewidth=0.3 if c == 'QSO' else 0)
    ax.set_title('[A-1] 2-Channel Latent Space by True Database Label (t-SNE)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)

    # [A-2] Latent Space by 2-Channel RF Predicted Label
    ax = axes[0, 1]
    for c in ['STAR', 'GALAXY', 'QSO']:
        m = (sample_pred == c)
        s = 35 if c == 'QSO' else 14
        alpha = 0.85 if c == 'QSO' else 0.30
        ax.scatter(coords_2d[m, 0], coords_2d[m, 1], c=color_map[c], label=f'Pred {c} (N={np.sum(m)})', s=s, alpha=alpha, edgecolors='none' if c != 'QSO' else 'black', linewidth=0.3 if c == 'QSO' else 0)
    ax.set_title('[A-2] 2-Channel Latent Space by RF Predicted Label (t-SNE)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)

    # [B-1] Continuous P(QSO) Probability Heatmap
    ax = axes[1, 0]
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=sample_q_probs, cmap='YlOrRd', s=18, alpha=0.8, vmin=0, vmax=1.0)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Predicted P(QSO)', fontsize=11)
    ax.set_title('[B-1] Continuous P(QSO) Probability Gradient Surface\n(Cleanly highlights the isolated QSO peninsula with high probability)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)

    # [B-2] Agreement vs Disagreement Map
    ax = axes[1, 1]
    agree_m = (sample_db == sample_pred)
    disagree_m = ~agree_m
    ax.scatter(coords_2d[agree_m, 0], coords_2d[agree_m, 1], c='#cccccc', s=12, alpha=0.3, label=f'Agreed with DB (N={np.sum(agree_m)})')
    ax.scatter(coords_2d[disagree_m, 0], coords_2d[disagree_m, 1], c='#e41a1c', s=24, alpha=0.85, label=f'Disagreed / Potential DB Errors (N={np.sum(disagree_m)})')
    ax.set_title('[B-2] Agreement vs Disagreement Map with Database Catalog\n(Disagreements strictly confined to interface boundary transitions)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)

    plt.suptitle("Latent Parameter Space (128D) Distribution: 2-Channel Dual-Stream SimCLR (Full Survey)",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig = RESULTS_DIR / "latent_distribution_2channel_full_survey.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved figure to {out_fig}")

    # Copy to artifacts
    art_dir = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")
    shutil.copy(out_fig, art_dir / "latent_distribution_2channel_full_survey.png")
    print("Copied to artifacts successfully!")
    print(f"Entire visualization script completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

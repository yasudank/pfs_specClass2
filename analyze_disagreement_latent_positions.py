import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import shutil

RESULTS_DIR = Path("results")

print("Loading data...")
embeds = np.load(RESULTS_DIR / "survey_embeddings_216k_final.npy")
meta = np.load(RESULTS_DIR / "survey_meta_216k_final.npz", allow_pickle=True)
rf = np.load(RESULTS_DIR / "rf_oof_predictions_216k.npz", allow_pickle=True)

y_true = meta["db_classes"]
y_pred = rf["oof_preds"]
oof_probs = rf["oof_probs"]
classes = list(rf["classes"])
qso_idx = classes.index("QSO")
gal_idx = classes.index("GALAXY")
star_idx = classes.index("STAR")

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)

embeds_norm = normalize(embeds)

# Base prototypes from Agreed objects
m_agreed_qso = (y_true == "QSO") & (y_pred == "QSO")
m_agreed_gal = (y_true == "GALAXY") & (y_pred == "GALAXY")
m_agreed_star = (y_true == "STAR") & (y_pred == "STAR")

p_qso = normalize(np.mean(embeds_norm[m_agreed_qso], axis=0, keepdims=True))[0]
p_gal = normalize(np.mean(embeds_norm[m_agreed_gal], axis=0, keepdims=True))[0]
p_star = normalize(np.mean(embeds_norm[m_agreed_star], axis=0, keepdims=True))[0]

groups = {
    "Agreed QSO (DB=QSO, RF=QSO)": m_agreed_qso,
    "Agreed GALAXY (DB=GAL, RF=GAL)": m_agreed_gal,
    "Agreed STAR (DB=STAR, RF=STAR)": m_agreed_star,
    "Missed QSO -> GALAXY (DB=QSO, RF=GAL)": (y_true == "QSO") & (y_pred == "GALAXY"),
    "Missed QSO -> STAR (DB=QSO, RF=STAR)": (y_true == "QSO") & (y_pred == "STAR"),
    "False QSO from GALAXY (DB=GAL, RF=QSO)": (y_true == "GALAXY") & (y_pred == "QSO"),
    "False QSO from STAR (DB=STAR, RF=QSO)": (y_true == "STAR") & (y_pred == "QSO"),
    "GALAXY misclassified as STAR": (y_true == "GALAXY") & (y_pred == "STAR"),
    "STAR misclassified as GALAXY": (y_true == "STAR") & (y_pred == "GALAXY"),
}

print(f"{'Group':<42} | {'N':>6} | {'Sim(QSO)':>9} | {'Sim(GAL)':>9} | {'Sim(STAR)':>9}")
print("-" * 85)
for name, m in groups.items():
    sub_e = embeds_norm[m]
    s_q = np.mean(np.dot(sub_e, p_qso))
    s_g = np.mean(np.dot(sub_e, p_gal))
    s_s = np.mean(np.dot(sub_e, p_star))
    print(f"{name:<42} | {np.sum(m):>6} | {s_q:>9.3f} | {s_g:>9.3f} | {s_s:>9.3f}")

# Now let us generate a high-clarity 2D Latent Space Visualization specifically focusing on Disagreements
# We sample a balanced background set (e.g., 2000 agreed stars, 2000 agreed galaxies, 470 agreed QSOs)
# and overlay 100% of all QSO-related disagreements and samples of Star/Galaxy disagreements!

np.random.seed(42)
bg_stars = np.random.choice(np.where(m_agreed_star)[0], 3000, replace=False)
bg_gals = np.random.choice(np.where(m_agreed_gal)[0], 3000, replace=False)
bg_qsos = np.where(m_agreed_qso)[0]

# All QSO disagreements
idx_missed_qso_gal = np.where(groups["Missed QSO -> GALAXY (DB=QSO, RF=GAL)"])[0]
idx_missed_qso_star = np.where(groups["Missed QSO -> STAR (DB=QSO, RF=STAR)"])[0]
idx_false_qso_gal = np.where(groups["False QSO from GALAXY (DB=GAL, RF=QSO)"])[0]
idx_false_qso_star = np.where(groups["False QSO from STAR (DB=STAR, RF=QSO)"])[0]

# Sample star-gal disagreements
idx_gal_to_star = np.random.choice(np.where(groups["GALAXY misclassified as STAR"])[0], 1000, replace=False)
idx_star_to_gal = np.random.choice(np.where(groups["STAR misclassified as GALAXY"])[0], 1000, replace=False)

all_plot_idx = np.concatenate([
    bg_stars, bg_gals, bg_qsos,
    idx_missed_qso_gal, idx_missed_qso_star,
    idx_false_qso_gal, idx_false_qso_star,
    idx_gal_to_star, idx_star_to_gal
])

print(f"\nComputing t-SNE for {len(all_plot_idx)} objects to visualize exact positions...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000, n_jobs=8)
coords_tsne = tsne.fit_transform(embeds[all_plot_idx])

# PCA for physical dimension interpretation
pca = PCA(n_components=2, random_state=42)
coords_pca = pca.fit_transform(embeds[all_plot_idx])

# Map back to indices in plot
ptr = 0
def get_slice(n):
    global ptr
    s = slice(ptr, ptr + n)
    ptr += n
    return s

sl_star = get_slice(len(bg_stars))
sl_gal = get_slice(len(bg_gals))
sl_qso = get_slice(len(bg_qsos))
sl_mq_gal = get_slice(len(idx_missed_qso_gal))
sl_mq_star = get_slice(len(idx_missed_qso_star))
sl_fq_gal = get_slice(len(idx_false_qso_gal))
sl_fq_star = get_slice(len(idx_false_qso_star))
sl_g2s = get_slice(len(idx_gal_to_star))
sl_s2g = get_slice(len(idx_star_to_gal))

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Panel 1: t-SNE with highlighted disagreement positions
ax = axes[0]
# Background agreed objects
ax.scatter(coords_tsne[sl_star, 0], coords_tsne[sl_star, 1], c='#4a90e2', alpha=0.15, s=12, label='Agreed STAR (Background)')
ax.scatter(coords_tsne[sl_gal, 0], coords_tsne[sl_gal, 1], c='#50e3c2', alpha=0.15, s=12, label='Agreed GALAXY (Background)')
ax.scatter(coords_tsne[sl_qso, 0], coords_tsne[sl_qso, 1], c='#d0021b', alpha=0.8, s=36, marker='o', edgecolors='black', linewidth=0.5, label='Agreed QSO (Core N=470)')

# Disagreements
ax.scatter(coords_tsne[sl_mq_gal, 0], coords_tsne[sl_mq_gal, 1], c='#9013fe', alpha=0.9, s=40, marker='^', edgecolors='white', linewidth=0.6, label=f'Missed QSO -> Pred GALAXY (N={len(idx_missed_qso_gal)})')
ax.scatter(coords_tsne[sl_mq_star, 0], coords_tsne[sl_mq_star, 1], c='#b8e986', alpha=0.9, s=40, marker='v', edgecolors='black', linewidth=0.6, label=f'Missed QSO -> Pred STAR (N={len(idx_missed_qso_star)})')
ax.scatter(coords_tsne[sl_fq_gal, 0], coords_tsne[sl_fq_gal, 1], c='#f5a623', alpha=0.9, s=45, marker='s', edgecolors='black', linewidth=0.6, label=f'False QSO from DB GALAXY (N={len(idx_false_qso_gal)})')
ax.scatter(coords_tsne[sl_s2g, 0], coords_tsne[sl_s2g, 1], c='#ff69b4', alpha=0.4, s=15, marker='.', label=f'STAR misclassified as GALAXY (Boundary)')
ax.scatter(coords_tsne[sl_g2s, 0], coords_tsne[sl_g2s, 1], c='#7ed321', alpha=0.4, s=15, marker='.', label=f'GALAXY misclassified as STAR (Boundary)')

ax.set_title('[A] t-SNE Latent Space: Disagreement Positions Relative to Major Clusters', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.35)

# Panel 2: Physical Projection Space (Cosine Similarity to QSO Prototype vs GALAXY Prototype)
ax = axes[1]
# Compute similarities for the plotted points
sim_q = np.dot(embeds_norm[all_plot_idx], p_qso)
sim_g = np.dot(embeds_norm[all_plot_idx], p_gal)
sim_s = np.dot(embeds_norm[all_plot_idx], p_star)

# We can plot sim_q vs sim_g or (sim_g - sim_s) vs sim_q
# Let's plot X = sim_g - sim_s (Galaxy-likeness vs Star-likeness), Y = sim_q (QSO-likeness)
x_metric = sim_g - sim_s
y_metric = sim_q

ax.scatter(x_metric[sl_star], y_metric[sl_star], c='#4a90e2', alpha=0.15, s=12, label='Agreed STAR')
ax.scatter(x_metric[sl_gal], y_metric[sl_gal], c='#50e3c2', alpha=0.15, s=12, label='Agreed GALAXY')
ax.scatter(x_metric[sl_qso], y_metric[sl_qso], c='#d0021b', alpha=0.8, s=36, edgecolors='black', linewidth=0.5, label='Agreed QSO')

ax.scatter(x_metric[sl_mq_gal], y_metric[sl_mq_gal], c='#9013fe', alpha=0.9, s=40, marker='^', edgecolors='white', linewidth=0.6, label='Missed QSO (DB=QSO, RF=GALAXY)')
ax.scatter(x_metric[sl_mq_star], y_metric[sl_mq_star], c='#b8e986', alpha=0.9, s=40, marker='v', edgecolors='black', linewidth=0.6, label='Missed QSO (DB=QSO, RF=STAR)')
ax.scatter(x_metric[sl_fq_gal], y_metric[sl_fq_gal], c='#f5a623', alpha=0.9, s=45, marker='s', edgecolors='black', linewidth=0.6, label='False QSO from GALAXY (DB=GAL, RF=QSO)')

ax.set_title('[B] 128D Projection: QSO Similarity vs [Galaxy - Star] Contrast', fontsize=12, fontweight='bold')
ax.set_xlabel('Galaxy - Star Contrast in 128D:  cos(x, Gal) - cos(x, Star)', fontsize=11)
ax.set_ylabel('QSO Likeness in 128D:  cos(x, QSO)', fontsize=11)
ax.axvline(0, color='gray', linestyle=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.35)

plt.tight_layout()
out_fig = RESULTS_DIR / "rf_disagreement_latent_positions.png"
plt.savefig(out_fig, dpi=200)
plt.close()
print(f"Saved figure to {out_fig}")

# Copy to artifacts directory
artifact_dir = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")
shutil.copy(out_fig, artifact_dir / "rf_disagreement_latent_positions.png")
print("Copied to artifacts.")

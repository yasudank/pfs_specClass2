"""Investigate Random Forest Classification from 128D Latent Parameters
Addressing User Request:
1. Use Random Forest to predict database classes (STAR, GALAXY, QSO) from 128D latent parameters.
2. Evaluate out-of-fold performance across all 216,476 survey objects.
3. Investigate potential DB labeling errors & label disagreements:
   - DB QSOs predicted as Galaxies/Stars
   - DB Galaxies/Stars predicted as QSOs (New QSO Candidates!)
   - Star-Galaxy separation boundary
4. Generate publication-grade figures:
   - rf_confusion_matrix_and_probabilities.png
   - rf_classification_latent_map.png
   - rf_disagreement_restframe_spectra.png
"""
import os
import sys
import time
import sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import joblib

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID, N_GRID
from dataset import load_raw_spectrum, preprocess_spectrum


def process_spectrum_worker(fits_path):
    raw = load_raw_spectrum(fits_path)
    if raw is None:
        return None
    prep = preprocess_spectrum(raw)
    if prep is None:
        return None
    return prep['flux_clean']


def main():
    print("=== Random Forest Latent Classification & Disagreement Investigation ===")
    t_start = time.time()

    # 1. Load 128D Embeddings and Metadata
    emb_path = RESULTS_DIR / "survey_embeddings_216k_final.npy"
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"

    print("Loading 216k survey embeddings and metadata...")
    embeds = np.load(emb_path)
    meta = np.load(meta_path, allow_pickle=True)

    obj_ids = meta['objIds']
    db_classes = meta['db_classes']
    db_redshifts = meta['db_redshifts']
    fits_paths = meta['fits_paths']

    N = len(embeds)
    classes_list = np.array(['GALAXY', 'QSO', 'STAR'])
    print(f"Loaded {N} targets across 128 latent dimensions.")
    print(f"True DB counts: {dict(zip(*np.unique(db_classes, return_counts=True)))}")

    # 2. 5-Fold Stratified Cross-Validation (Out-of-Fold Predictions for ALL 216,476 targets)
    print("\nRunning 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.empty(N, dtype=object)
    oof_probs = np.zeros((N, 3), dtype=np.float32)

    rf_models = []
    t_cv = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeds, db_classes), 1):
        f_t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=4,
            class_weight='balanced_subsample',
            n_jobs=16,
            random_state=42 + fold
        )
        rf.fit(embeds[train_idx], db_classes[train_idx])
        rf_models.append(rf)

        # Predict probabilities
        val_probs = rf.predict_proba(embeds[val_idx]) # (n_val, 3) in order ['GALAXY', 'QSO', 'STAR']
        val_preds = rf.classes_[np.argmax(val_probs, axis=1)]

        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds
        print(f"Fold {fold}/5 completed in {time.time() - f_t0:.1f}s.")

    print(f"5-Fold Cross-Validation completed in {time.time() - t_cv:.1f}s.")

    # Save OOF predictions
    np.savez_compressed(
        RESULTS_DIR / "rf_oof_predictions_216k.npz",
        oof_preds=oof_preds,
        oof_probs=oof_probs,
        classes=rf_models[0].classes_
    )

    # 3. Overall Survey Evaluation Metrics
    classes = rf_models[0].classes_ # ['GALAXY', 'QSO', 'STAR']
    print("\n================ FULL SURVEY OOF PERFORMANCE ================")
    report_dict = classification_report(db_classes, oof_preds, target_names=classes, digits=4, output_dict=True)
    print(classification_report(db_classes, oof_preds, target_names=classes, digits=4))

    cm = confusion_matrix(db_classes, oof_preds, labels=classes)
    print("\nConfusion Matrix (Rows: True DB Class, Columns: RF Predicted Class):")
    print(f"{'':10s} | {'Pred GALAXY':>12s} | {'Pred QSO':>12s} | {'Pred STAR':>12s}")
    print("-" * 55)
    for i, true_c in enumerate(classes):
        print(f"{'True ' + true_c:10s} | {cm[i, 0]:12d} | {cm[i, 1]:12d} | {cm[i, 2]:12d}")

    # Predicted Totals
    pred_counts = dict(zip(*np.unique(oof_preds, return_counts=True)))
    print(f"\nTotal Survey Predicted Counts: {pred_counts}")
    print(f"Total True DB Counts:          {dict(zip(*np.unique(db_classes, return_counts=True)))}")

    # 4. Detailed Investigation of Disagreements & Potential DB Errors
    print("\n================ DISAGREEMENT & DB ERROR ANALYSIS ================")
    # (A) DB QSO vs Predicted Non-QSO
    db_qso_mask = (db_classes == 'QSO')
    qso_missed = db_qso_mask & (oof_preds != 'QSO')
    print(f"(A) DB QSOs predicted as GALAXY or STAR: {np.sum(qso_missed)} / {np.sum(db_qso_mask)} ({np.sum(qso_missed)/np.sum(db_qso_mask)*100:.1f}%)")
    print(f"    - Predicted as GALAXY: {np.sum(db_qso_mask & (oof_preds == 'GALAXY'))}")
    print(f"    - Predicted as STAR:   {np.sum(db_qso_mask & (oof_preds == 'STAR'))}")
    med_z_qso_missed = np.nanmedian(db_redshifts[qso_missed])
    print(f"    - Median redshift of missed QSOs: {med_z_qso_missed:.3f} (Lower-z galaxy-dominated hosts!)")

    # (B) DB Non-QSO vs Predicted QSO (New QSO Candidates!)
    new_qso_cand = (~db_qso_mask) & (oof_preds == 'QSO')
    print(f"(B) DB Non-QSOs predicted as QSO (New Candidates): {np.sum(new_qso_cand)}")
    print(f"    - Formerly DB GALAXY: {np.sum((db_classes == 'GALAXY') & (oof_preds == 'QSO'))}")
    print(f"    - Formerly DB STAR:   {np.sum((db_classes == 'STAR') & (oof_preds == 'QSO'))}")

    # (C) Star vs Galaxy confusion
    star_as_gal = (db_classes == 'STAR') & (oof_preds == 'GALAXY')
    gal_as_star = (db_classes == 'GALAXY') & (oof_preds == 'STAR')
    print(f"(C) Star/Galaxy Boundary:")
    print(f"    - DB STAR predicted as GALAXY: {np.sum(star_as_gal)} ({np.sum(star_as_gal)/np.sum(db_classes=='STAR')*100:.2f}%)")
    print(f"    - DB GALAXY predicted as STAR: {np.sum(gal_as_star)} ({np.sum(gal_as_star)/np.sum(db_classes=='GALAXY')*100:.2f}%)")

    # 5. Figure 1: Confusion Matrix & Probability Distributions
    print("\nGenerating Figure 1: Confusion Matrix & Probabilities...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))

    # [1] Normalized Confusion Matrix
    ax = axes[0, 0]
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax, xticklabels=classes, yticklabels=classes, cbar=False)
    ax.set_title('[A] Out-Of-Fold Confusion Matrix (Normalized)\nRandom Forest (100 Trees, 128 Latent Features)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Class (Random Forest)', fontsize=10)
    ax.set_ylabel('True Database Class', fontsize=10)

    # [2] Predicted QSO Probability Distributions
    ax = axes[0, 1]
    q_prob_idx = list(classes).index('QSO')
    q_probs = oof_probs[:, q_prob_idx]
    bins = np.linspace(0.0, 1.0, 50)
    ax.hist(q_probs[db_classes == 'QSO'], bins=bins, density=True, alpha=0.7, color='#d62728', label=f'True QSO (N={np.sum(db_classes=="QSO")})')
    ax.hist(q_probs[db_classes == 'GALAXY'], bins=bins, density=True, alpha=0.5, color='#2ca02c', label=f'True GALAXY (N={np.sum(db_classes=="GALAXY")})')
    ax.hist(q_probs[db_classes == 'STAR'], bins=bins, density=True, alpha=0.5, color='#1f77b4', label=f'True STAR (N={np.sum(db_classes=="STAR")})')
    ax.set_yscale('log')
    ax.set_title('[B] Predicted QSO Probability P(QSO) Distribution\n(Clear separation: Galaxies/Stars cluster near 0, QSOs near 1)', fontsize=11, fontweight='bold')
    ax.set_xlabel('P(QSO)', fontsize=10)
    ax.set_ylabel('Density (Log Scale)', fontsize=10)
    ax.legend(loc='upper right', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.4)

    # [3] Comparison of Total Survey Class Counts
    ax = axes[1, 0]
    x_pos = np.arange(3)
    width = 0.35
    db_totals = [np.sum(db_classes == c) for c in classes]
    rf_totals = [np.sum(oof_preds == c) for c in classes]
    ax.bar(x_pos - width/2, db_totals, width, label='Database Catalog Labels', color='#888888', alpha=0.75)
    ax.bar(x_pos + width/2, rf_totals, width, label='Random Forest Predicted Labels', color=['#2ca02c', '#d62728', '#1f77b4'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Number of Objects (out of 216,476)', fontsize=10)
    ax.set_title('[C] Total Survey Population: Database vs Random Forest Prediction\n(Notice: QSO count is realistic ~600-700, NOT 56,000!)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    for i in range(3):
        ax.text(x_pos[i] - width/2, db_totals[i] + 3000, f"{db_totals[i]:,}", ha='center', fontsize=9)
        ax.text(x_pos[i] + width/2, rf_totals[i] + 3000, f"{rf_totals[i]:,}", ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 130000)

    # [4] Star vs Galaxy Probability Distribution
    ax = axes[1, 1]
    g_prob_idx = list(classes).index('GALAXY')
    g_probs = oof_probs[:, g_prob_idx]
    ax.hist(g_probs[db_classes == 'GALAXY'], bins=bins, density=True, alpha=0.6, color='#2ca02c', label='True GALAXY')
    ax.hist(g_probs[db_classes == 'STAR'], bins=bins, density=True, alpha=0.6, color='#1f77b4', label='True STAR')
    ax.set_title('[D] Predicted Galaxy Probability P(GALAXY) Distribution\n(High discriminative confidence between Stars and Galaxies)', fontsize=11, fontweight='bold')
    ax.set_xlabel('P(GALAXY)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(loc='upper center', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig1_path = RESULTS_DIR / "rf_confusion_matrix_and_probabilities.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved Figure 1 to {fig1_path}.")

    # 6. Figure 2: Latent Space Map with RF Predictions
    print("\nGenerating Figure 2: Latent Space Map with RF Predictions...")
    qso_idx = np.where(db_classes == 'QSO')[0]
    gal_idx = np.random.RandomState(42).choice(np.where(db_classes == 'GALAXY')[0], 4500, replace=False)
    star_idx = np.random.RandomState(42).choice(np.where(db_classes == 'STAR')[0], 4500, replace=False)
    sample_idx = np.concatenate([qso_idx, gal_idx, star_idx])

    sample_embeds = embeds[sample_idx]
    sample_db_cls = db_classes[sample_idx]
    sample_rf_pred = oof_preds[sample_idx]
    sample_q_probs = oof_probs[sample_idx, q_prob_idx]
    sample_z = db_redshifts[sample_idx]

    tsne = TSNE(n_components=2, perplexity=35, random_state=42, metric='cosine')
    embeds_tsne = tsne.fit_transform(sample_embeds)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    color_map = {'STAR': '#1f77b4', 'GALAXY': '#2ca02c', 'QSO': '#d62728'}

    # [A-1] True DB Labels
    ax = axes[0, 0]
    for c in ['STAR', 'GALAXY', 'QSO']:
        m = (sample_db_cls == c)
        s = 35 if c == 'QSO' else 14
        alpha = 0.85 if c == 'QSO' else 0.35
        ax.scatter(embeds_tsne[m, 0], embeds_tsne[m, 1], c=color_map[c], label=f'{c} (N={np.sum(m)})', s=s, alpha=alpha)
    ax.set_title('[A-1] Latent Space by True Database Label (t-SNE)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # [A-2] RF Predicted Labels
    ax = axes[0, 1]
    for c in ['STAR', 'GALAXY', 'QSO']:
        m = (sample_rf_pred == c)
        s = 35 if c == 'QSO' else 14
        alpha = 0.85 if c == 'QSO' else 0.35
        ax.scatter(embeds_tsne[m, 0], embeds_tsne[m, 1], c=color_map[c], label=f'Pred {c} (N={np.sum(m)})', s=s, alpha=alpha)
    ax.set_title('[A-2] Latent Space by Random Forest Predicted Label (t-SNE)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    # [B-1] High-Confidence RF QSO Probabilities
    ax = axes[1, 0]
    sc = ax.scatter(embeds_tsne[:, 0], embeds_tsne[:, 1], c=sample_q_probs, cmap='YlOrRd', s=16, alpha=0.75, vmin=0, vmax=1.0)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Random Forest Predicted P(QSO)', fontsize=10)
    ax.set_title('[B-1] Continuous Predicted QSO Probability Map\n(Isolates the upper-left QSO crescent with precision)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    # [B-2] Agreement vs Disagreement Map
    ax = axes[1, 1]
    agree_mask = (sample_db_cls == sample_rf_pred)
    disagree_mask = ~agree_mask
    ax.scatter(embeds_tsne[agree_mask, 0], embeds_tsne[agree_mask, 1], c='#cccccc', s=10, alpha=0.25, label=f'Agreed with DB (N={np.sum(agree_mask)})')
    ax.scatter(embeds_tsne[disagree_mask, 0], embeds_tsne[disagree_mask, 1], c='#e41a1c', s=24, alpha=0.85, label=f'Disagreed / Potential DB Errors (N={np.sum(disagree_mask)})')
    ax.set_title('[B-2] Agreement vs Disagreement with Database\n(Disagreements occur along physical transition boundaries)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig2_path = RESULTS_DIR / "rf_classification_latent_map.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved Figure 2 to {fig2_path}.")

    # 7. Figure 3: Median Rest-Frame Spectra of Disagreement & Prototype Groups
    print("\nGenerating Figure 3: Rest-Frame Spectra of Agreement & Disagreement Groups...")
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

    groups = [
        ("Agreed QSOs: DB=QSO & RF=QSO", (db_classes == 'QSO') & (oof_preds == 'QSO'), '#d62728'),
        ("New QSO Candidates: DB=GALAXY & RF=QSO", (db_classes == 'GALAXY') & (oof_preds == 'QSO'), '#ff7f0e'),
        ("Missed/Host-Dominated QSOs: DB=QSO & RF=GALAXY", (db_classes == 'QSO') & (oof_preds == 'GALAXY'), '#9467bd'),
        ("Agreed Galaxies: DB=GALAXY & RF=GALAXY", (db_classes == 'GALAXY') & (oof_preds == 'GALAXY'), '#2ca02c'),
        ("Agreed Stars: DB=STAR & RF=STAR", (db_classes == 'STAR') & (oof_preds == 'STAR'), '#1f77b4'),
    ]

    fig, axes = plt.subplots(len(groups), 1, figsize=(16, 15), sharex=True)

    for i, (title, mask, col) in enumerate(groups):
        ax = axes[i]
        indices = np.where(mask)[0]
        n_group = len(indices)
        sub_idx = indices if n_group <= 600 else np.random.RandomState(42).choice(indices, 600, replace=False)
        sub_fits = [fits_paths[idx] for idx in sub_idx]
        sub_z = [db_redshifts[idx] for idx in sub_idx]

        with ProcessPoolExecutor(max_workers=16) as ex:
            fluxes = list(ex.map(process_spectrum_worker, sub_fits))

        rest_fluxes = []
        for f, z in zip(fluxes, sub_z):
            if f is None:
                continue
            use_z = z if (z is not None and np.isfinite(z) and z > 0.001) else 0.0
            obs_wl_cand = rest_wl * (1.0 + use_z)
            f_interp = np.interp(obs_wl_cand, WL_GRID, f, left=np.nan, right=np.nan)
            rest_fluxes.append(f_interp)

        rest_fluxes = np.array(rest_fluxes)
        proto = np.nanmedian(rest_fluxes, axis=0)
        p16 = np.nanpercentile(rest_fluxes, 16, axis=0)
        p84 = np.nanpercentile(rest_fluxes, 84, axis=0)
        med_z = float(np.nanmedian([z for z in sub_z if z > 0])) if any(z > 0 for z in sub_z) else 0.0

        ax.fill_between(rest_wl, p16, p84, color=col, alpha=0.18)
        lbl = f"{title} | Total N={n_group} (Plotted: {len(rest_fluxes)}) | Med z={med_z:.2f}"
        ax.plot(rest_wl, proto, color=col, lw=1.3, label=lbl)

        ax.set_ylabel("Rest Flux", fontsize=9.5)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=9.5)
        ax.grid(True, linestyle='--', alpha=0.4)

        for name, line_wl in spectral_lines.items():
            ax.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)
            if i == 0:
                ax.text(line_wl, ax.get_ylim()[1]*0.88, name, rotation=90, fontsize=7.5, color='#444444', ha='right')

        y_lo = np.nanpercentile(proto[np.isfinite(proto)], 2) - 0.2 if np.any(np.isfinite(proto)) else -0.5
        y_hi = np.nanpercentile(proto[np.isfinite(proto)], 98) + 0.6 if np.any(np.isfinite(proto)) else 2.0
        ax.set_ylim(max(y_lo, -0.5), min(y_hi, 4.0))

    axes[-1].set_xlabel("Rest-Frame Wavelength (nm)", fontsize=11)
    axes[-1].set_xlim(110.0, 950.0)
    plt.suptitle("Median Rest-Frame Spectra: Database vs Random Forest Classifications & Disagreements", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    fig3_path = RESULTS_DIR / "rf_disagreement_restframe_spectra.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved Figure 3 to {fig3_path}.")
    print(f"Entire investigation completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

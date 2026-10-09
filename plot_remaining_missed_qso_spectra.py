"""Plot representative spectra of Remaining Missed QSOs -> GALAXY (N=303 in Experiment 3).
Shows:
Left Column: Observed Frame spectrum (380 - 1260 nm) with normalized flux and Channel 1 residual.
Right Column: Rest-Frame spectrum (de-redshifted by z_DB) with prominent QSO & galaxy line markers.
Header of each panel shows:
  - Database Redshift z_DB
  - RF Predicted Probabilities: P(GALAXY), P(QSO), P(STAR)
  - 128D Latent Cosine Similarities: Sim(z, p_GAL) vs Sim(z, p_QSO)
"""
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID

BRAIN_DIR = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")


def compute_continuum_subtracted(flux, window_size=121):
    cont = median_filter(flux, size=window_size, mode='nearest')
    cont = gaussian_filter1d(cont, sigma=15.0, mode='nearest')
    residual = flux - cont
    return cont, residual


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def main():
    t_start = time.time()
    print("Loading Experiment 3 dataset, predictions, and embeddings...")

    pred_path = RESULTS_DIR / "rf_oof_boost3x_smote_representative.npz"
    emb_path = RESULTS_DIR / "embeddings_representative_boost3x.npy"
    data_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"

    preds = np.load(pred_path, allow_pickle=True)
    embeds = np.load(emb_path)
    data = np.load(data_path, allow_pickle=True)

    all_flux = data['flux_clean']
    all_obj_ids = data['objId']
    _, unique_indices = np.unique(all_obj_ids, return_index=True)

    flux = all_flux[unique_indices]
    db_classes = preds['db_classes']
    oof_preds = preds['oof_preds']
    oof_probs = preds['oof_probs']
    db_redshifts = preds['db_redshifts']

    classes = list(preds['classes'])
    idx_g = classes.index('GALAXY')
    idx_q = classes.index('QSO')
    idx_s = classes.index('STAR')

    # Prototypes in 128D space
    embeds_norm = normalize(embeds)
    m_aq = (db_classes == 'QSO') & (oof_preds == 'QSO')
    m_ag = (db_classes == 'GALAXY') & (oof_preds == 'GALAXY')
    proto_qso = normalize(np.mean(embeds_norm[m_aq], axis=0, keepdims=True))[0]
    proto_gal = normalize(np.mean(embeds_norm[m_ag], axis=0, keepdims=True))[0]

    sim_qso = np.dot(embeds_norm, proto_qso)
    sim_gal = np.dot(embeds_norm, proto_gal)

    # Filter Remaining Missed QSOs -> GALAXY
    m_missed = (db_classes == 'QSO') & (oof_preds == 'GALAXY')
    missed_indices = np.where(m_missed)[0]
    print(f"Total Remaining Missed QSOs -> GALAXY: {len(missed_indices)}")

    # Select 6 diverse, high-information targets across the redshift spectrum:
    # 1. Low-z (z ~ 0.435) - Strong host galaxy starlight
    # 2. Low-z (z ~ 0.701) - 4000A break & weak AGN emission
    # 3. Intermediate-z (z ~ 1.326) - Mg II present, galaxy continuum slope
    # 4. Intermediate-z (z ~ 1.574) - Mg II with significant host contamination
    # 5. Higher-z (z ~ 1.794) - Near boundary (P(GAL)=0.43, P(QSO)=0.38)
    # 6. High-z (z ~ 2.115) - High-z with extinguished UV continuum / noisy lines
    target_candidates = [
        1597,  # z=0.435
        1428,  # z=0.701
        289,   # z=1.326
        1318,  # z=1.574
        558,   # z=1.794 (borderline case)
        622,   # z=2.115
    ]

    # Verify all are in missed_indices
    selected_indices = [idx for idx in target_candidates if idx in missed_indices]
    if len(selected_indices) < 6:
        # Fallback to top 6 if needed
        selected_indices = missed_indices[:6]

    print(f"Plotting {len(selected_indices)} representative spectra: {selected_indices}")

    # Standard astronomical spectral lines for reference (in nm)
    spectral_lines = {
        r'Ly$\alpha$': 121.6,
        r'C IV': 154.9,
        r'C III]': 190.9,
        r'Mg II': 279.8,
        r'[O II]': 372.7,
        r'4000Å': 400.0,
        r'H$\beta$': 486.1,
        r'[O III]': 500.7,
        r'H$\alpha$': 656.3
    }

    n_rows = len(selected_indices)
    fig, axes = plt.subplots(n_rows, 2, figsize=(19, 3.4 * n_rows))
    plt.subplots_adjust(hspace=0.38, wspace=0.15)

    for i, idx in enumerate(selected_indices):
        f = flux[idx]
        z_val = db_redshifts[idx]
        p_gal_val = oof_probs[idx, idx_g]
        p_qso_val = oof_probs[idx, idx_q]
        p_star_val = oof_probs[idx, idx_s]
        s_gal_val = sim_gal[idx]
        s_qso_val = sim_qso[idx]

        cont, res = compute_continuum_subtracted(f)
        wl_rest = WL_GRID / (1.0 + z_val)

        # ---------------------------------------------------------
        # Left Panel: Observed Frame (380 - 1260 nm)
        # ---------------------------------------------------------
        ax_obs = axes[i, 0]
        ax_obs.plot(WL_GRID, f, color='#1f77b4', lw=1.1, label='Observed Flux')
        ax_obs.plot(WL_GRID, cont, color='#d95f02', lw=1.2, linestyle='--', alpha=0.85, label='Continuum (Ch0)')
        ax_obs.plot(WL_GRID, res - 1.2, color='#7570b3', lw=0.8, alpha=0.75, label='Residual Ch1 (offset -1.2)')
        ax_obs.axhline(-1.2, color='gray', linestyle=':', lw=0.7)

        title_obs = (
            f"Object #{i+1} [Index {idx}] | Observed Frame | "
            f"DB Redshift $z={z_val:.3f}$\n"
            f"Model Pred: GALAXY (P(GAL)={p_gal_val:.2f}, P(QSO)={p_qso_val:.2f}, P(STAR)={p_star_val:.2f}) | "
            f"Latent Sim: $\\mathrm{{Sim}}_{{\\mathrm{{GAL}}}}={s_gal_val:.3f} > \\mathrm{{Sim}}_{{\\mathrm{{QSO}}}}={s_qso_val:.3f}$"
        )
        ax_obs.set_title(title_obs, fontsize=10.0, fontweight='bold', loc='left')
        ax_obs.set_ylabel("Obs Flux (norm)", fontsize=9.0)
        ax_obs.set_xlim(370.0, 1260.0)

        finite_f = f[np.isfinite(f)]
        y_min = max(float(np.nanpercentile(finite_f, 0.5)) - 1.5, -2.5) if len(finite_f) > 0 else -2.5
        y_max = min(float(np.nanpercentile(finite_f, 99.5)) + 0.8, 5.5) if len(finite_f) > 0 else 4.0
        ax_obs.set_ylim(y_min, y_max)
        ax_obs.grid(True, linestyle='--', alpha=0.35)
        if i == 0:
            ax_obs.legend(loc='upper right', fontsize=8.5, ncol=3)
        if i == n_rows - 1:
            ax_obs.set_xlabel("Observed Wavelength $\lambda_{\mathrm{obs}}$ [nm]", fontsize=10.0)

        # ---------------------------------------------------------
        # Right Panel: Rest Frame (de-redshifted by z_DB)
        # ---------------------------------------------------------
        ax_rest = axes[i, 1]
        ax_rest.plot(wl_rest, f, color='#2ca02c', lw=1.1, label='Rest-Frame Flux')
        ax_rest.plot(wl_rest, cont, color='#e7298a', lw=1.1, linestyle='--', alpha=0.8, label='Continuum')

        # Add vertical emission line markers
        for line_name, line_wl in spectral_lines.items():
            if wl_rest.min() <= line_wl <= wl_rest.max():
                ax_rest.axvline(line_wl, color='#d62728', linestyle=':', lw=1.0, alpha=0.75)
                # Text label near top
                ax_rest.text(line_wl, y_max - 0.45 * (y_max - y_min), f"{line_name}\n{line_wl:.1f}",
                             rotation=90, fontsize=7.5, color='#b2182b', va='center', ha='right',
                             bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.6))

        title_rest = f"Rest-Frame ($\\lambda_{{\\mathrm{{rest}}}} = \\lambda_{{\\mathrm{{obs}}}} / (1+{z_val:.3f})$) | Spectral Features & Line Alignment"
        ax_rest.set_title(title_rest, fontsize=10.0, fontweight='bold', loc='left')
        ax_rest.set_ylabel("Rest Flux (norm)", fontsize=9.0)
        ax_rest.set_xlim(max(wl_rest.min(), 100.0), min(wl_rest.max(), 950.0))
        ax_rest.set_ylim(y_min, y_max)
        ax_rest.grid(True, linestyle='--', alpha=0.35)
        if i == 0:
            ax_rest.legend(loc='upper right', fontsize=8.5)
        if i == n_rows - 1:
            ax_rest.set_xlabel("Rest Wavelength $\lambda_{\mathrm{rest}}$ [nm]", fontsize=10.0)

    save_path = RESULTS_DIR / "remaining_missed_qso_sample_spectra.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved spectra figure to {save_path}")

    brain_path = BRAIN_DIR / "remaining_missed_qso_sample_spectra.png"
    shutil.copy(save_path, brain_path)
    print(f"Copied to brain directory: {brain_path}")
    print(f"Execution completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

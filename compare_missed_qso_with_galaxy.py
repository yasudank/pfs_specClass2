"""Compare Missed / Host-Dominated QSOs (DB: QSO, Pred: GALAXY) with their
nearest-neighbor Agreed Galaxies (DB: GALAXY, Pred: GALAXY) in 128D latent space.
"""
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID, N_GRID
from dataset import load_raw_spectrum, preprocess_spectrum

BRAIN_DIR = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")


def compute_continuum_subtracted(flux, window_size=121):
    from scipy.ndimage import median_filter, gaussian_filter1d
    cont = median_filter(flux, size=window_size, mode='nearest')
    cont = gaussian_filter1d(cont, sigma=15.0, mode='nearest')
    residual = flux - cont
    return cont, residual


def main():
    meta = np.load(RESULTS_DIR / "survey_meta_216k_final.npz", allow_pickle=True)
    preds = np.load(RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz", allow_pickle=True)
    embeds = np.load(RESULTS_DIR / "survey_embeddings_216k_2channel.npy")

    db_classes = meta['db_classes']
    db_redshifts = meta['db_redshifts']
    fits_paths = meta['fits_paths']
    oof_preds = preds['oof_preds']

    m_missed = np.where((db_classes == 'QSO') & (oof_preds == 'GALAXY'))[0]
    m_agreed_gal = np.where((db_classes == 'GALAXY') & (oof_preds == 'GALAXY'))[0]

    def norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    embeds_norm = norm(embeds)
    gal_embeds = embeds_norm[m_agreed_gal]

    # Select two compelling pairs:
    # Pair 1: Missed QSO idx=1569 (z=1.326)
    # Pair 2: Missed QSO idx=689 (z=2.100)
    selected_missed = [1569, 689]

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

    fig, axes = plt.subplots(len(selected_missed), 2, figsize=(18, 10))

    for row, q_idx in enumerate(selected_missed):
        q_emb = embeds_norm[q_idx]
        sims = np.dot(gal_embeds, q_emb)
        best_gal_sub_idx = np.argmax(sims)
        g_idx = m_agreed_gal[best_gal_sub_idx]
        best_sim = sims[best_gal_sub_idx]

        f_q_fits = fits_paths[q_idx]
        f_g_fits = fits_paths[g_idx]
        z_q = db_redshifts[q_idx]
        z_g = db_redshifts[g_idx]

        prep_q = preprocess_spectrum(load_raw_spectrum(f_q_fits))
        prep_g = preprocess_spectrum(load_raw_spectrum(f_g_fits))

        flux_q = prep_q['flux_clean']
        flux_g = prep_g['flux_clean']

        _, res_q = compute_continuum_subtracted(flux_q)
        _, res_g = compute_continuum_subtracted(flux_g)

        # Left Column: Observed Frame (What SimCLR 2-channel sees)
        ax_obs = axes[row, 0]
        ax_obs.plot(WL_GRID, flux_q, color='#9467bd', lw=1.2, 
                    label=f"Missed QSO #{row+1} (DB: QSO, Pred: GAL, $z_{{DB}}={z_q:.3f}$)")
        ax_obs.plot(WL_GRID, flux_g, color='#2ca02c', lw=1.2, alpha=0.75, linestyle='--',
                    label=f"Nearest Agreed Galaxy (Sim={best_sim:.3f}, $z_{{DB}}={z_g:.3f}$)")

        # Residuals underneath
        ax_obs.plot(WL_GRID, res_q - 1.5, color='#756bb1', lw=1.0, alpha=0.85, label=f"Missed QSO Ch 1: Residual (-1.5)")
        ax_obs.plot(WL_GRID, res_g - 1.5, color='#31a354', lw=1.0, alpha=0.75, linestyle='--', label=f"Galaxy Ch 1: Residual (-1.5)")

        ax_obs.axhline(-1.5, color='gray', linestyle=':', lw=0.8)
        ax_obs.set_title(f"Pair #{row+1} Observed Frame Input (SimCLR 2-Channel View)\n[Cosine Similarity in 128D Latent Space = {best_sim:.4f}]", fontsize=11, fontweight='bold')
        ax_obs.set_xlabel("Observed Wavelength (nm)", fontsize=10)
        ax_obs.set_ylabel("Normalized Flux & Residual", fontsize=10)
        ax_obs.legend(loc='upper right', fontsize=8.8)
        ax_obs.grid(True, linestyle='--', alpha=0.35)

        # Right Column: Rest Frame (De-redshifted by respective DB redshifts)
        ax_rest = axes[row, 1]
        rest_grid = np.linspace(110.0, 900.0, 3000)

        f_rest_q = np.interp(rest_grid * (1.0 + z_q), WL_GRID, flux_q, left=np.nan, right=np.nan)
        f_rest_g = np.interp(rest_grid * (1.0 + z_g), WL_GRID, flux_g, left=np.nan, right=np.nan)

        ax_rest.plot(rest_grid, f_rest_q, color='#9467bd', lw=1.3, label=f"Missed QSO #{row+1} Rest-Frame ($z={z_q:.3f}$)")
        ax_rest.plot(rest_grid, f_rest_g, color='#2ca02c', lw=1.3, alpha=0.75, linestyle='--', label=f"Agreed Galaxy Rest-Frame ($z={z_g:.3f}$)")

        for name, line_wl in spectral_lines.items():
            if 110.0 <= line_wl <= 900.0:
                ax_rest.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)
                ax_rest.text(line_wl, ax_rest.get_ylim()[1]*0.88 if ax_rest.get_ylim()[1] > 0 else 1.5, name, rotation=90, fontsize=7.5, color='#444444', ha='right')

        ax_rest.set_title(f"Pair #{row+1} Rest-Frame Alignment (De-redshifted by DB $z$)\nRevealing why the Galaxy Continuum Dominates over Weak AGN Lines", fontsize=11, fontweight='bold')
        ax_rest.set_xlabel("Rest Wavelength (nm)", fontsize=10)
        ax_rest.set_ylabel("Rest Flux", fontsize=10)
        ax_rest.legend(loc='upper right', fontsize=8.8)
        ax_rest.grid(True, linestyle='--', alpha=0.35)

    plt.suptitle("Spectral Comparison: Missed / Host-Dominated QSOs vs Their Nearest Latent Neighbor Agreed Galaxies\n(Explaining why SimCLR embeds them in the Galaxy Cluster)", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_fig = RESULTS_DIR / "missed_qso_vs_galaxy_spectra_comparison.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved to {out_fig}.")

    brain_fig = BRAIN_DIR / "missed_qso_vs_galaxy_spectra_comparison.png"
    shutil.copy2(out_fig, brain_fig)
    print(f"Copied to {brain_fig}.")


if __name__ == "__main__":
    main()

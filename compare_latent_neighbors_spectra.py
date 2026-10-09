"""Compare observed and rest-frame spectra of New QSO Candidates (DB: GALAXY, Pred: QSO)
with their nearest-neighbor Agreed QSOs in 128D latent space.
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

    m_new_qso = np.where((db_classes == 'GALAXY') & (oof_preds == 'QSO'))[0]
    m_agreed_qso = np.where((db_classes == 'QSO') & (oof_preds == 'QSO'))[0]

    def norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    embeds_norm = norm(embeds)

    # Pick 2 very striking candidates
    # Candidate A: index 258 (z=1.604)
    # Candidate B: index 894 (z=0.839)
    test_cand_indices = [258, 894]

    fig, axes = plt.subplots(len(test_cand_indices), 2, figsize=(18, 10))

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

    for row, cand_idx in enumerate(test_cand_indices):
        cand_emb = embeds_norm[cand_idx]
        sims = np.dot(embeds_norm[m_agreed_qso], cand_emb)
        match_idx = m_agreed_qso[np.argmax(sims)]
        best_sim = np.max(sims)

        f_cand_fits = fits_paths[cand_idx]
        f_qso_fits = fits_paths[match_idx]
        z_cand = db_redshifts[cand_idx]
        z_qso = db_redshifts[match_idx]

        prep_cand = preprocess_spectrum(load_raw_spectrum(f_cand_fits))
        prep_qso = preprocess_spectrum(load_raw_spectrum(f_qso_fits))

        flux_cand = prep_cand['flux_clean']
        flux_qso = prep_qso['flux_clean']

        _, res_cand = compute_continuum_subtracted(flux_cand)
        _, res_qso = compute_continuum_subtracted(flux_qso)

        # Left Column: Observed Frame (What SimCLR Actually Sees in 2 Channels!)
        ax_obs = axes[row, 0]
        ax_obs.plot(WL_GRID, flux_cand, color='#ff7f0e', lw=1.2, label=f"Candidate #{row+1} (DB: GALAXY, Pred: QSO, $z_{{DB}}={z_cand:.3f}$)")
        ax_obs.plot(WL_GRID, flux_qso, color='#d62728', lw=1.2, alpha=0.75, linestyle='--', label=f"Nearest Agreed QSO (Sim={best_sim:.3f}, $z_{{DB}}={z_qso:.3f}$)")
        
        # Plot Residual channel underneath
        ax_obs.plot(WL_GRID, res_cand - 1.5, color='#e6550d', lw=1.0, alpha=0.85, label=f"Ch 1: Residual (Offset -1.5)")
        ax_obs.plot(WL_GRID, res_qso - 1.5, color='#99000d', lw=1.0, alpha=0.75, linestyle='--', label=f"Nearest QSO Residual (Offset -1.5)")

        ax_obs.axhline(-1.5, color='gray', linestyle=':', lw=0.8)
        ax_obs.set_title(f"Pair #{row+1} Observed Frame Input to SimCLR\n[Cosine Similarity in 128D Latent Space = {best_sim:.4f}]", fontsize=11, fontweight='bold')
        ax_obs.set_xlabel("Observed Wavelength (nm)", fontsize=10)
        ax_obs.set_ylabel("Normalized Flux & Residual", fontsize=10)
        ax_obs.legend(loc='upper right', fontsize=8.8)
        ax_obs.grid(True, linestyle='--', alpha=0.35)

        # Right Column: Rest Frame (De-redshifted using their respective redshifts)
        ax_rest = axes[row, 1]
        rest_grid = np.linspace(110.0, 900.0, 3000)
        
        f_rest_cand = np.interp(rest_grid * (1.0 + z_cand), WL_GRID, flux_cand, left=np.nan, right=np.nan)
        f_rest_qso = np.interp(rest_grid * (1.0 + z_qso), WL_GRID, flux_qso, left=np.nan, right=np.nan)

        ax_rest.plot(rest_grid, f_rest_cand, color='#ff7f0e', lw=1.3, label=f"Candidate #{row+1} Rest-Frame ($z={z_cand:.3f}$)")
        ax_rest.plot(rest_grid, f_rest_qso, color='#d62728', lw=1.3, alpha=0.75, linestyle='--', label=f"Agreed QSO Rest-Frame ($z={z_qso:.3f}$)")

        for name, line_wl in spectral_lines.items():
            if 110.0 <= line_wl <= 900.0:
                ax_rest.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)
                ax_rest.text(line_wl, ax_rest.get_ylim()[1]*0.88 if ax_rest.get_ylim()[1] > 0 else 2.0, name, rotation=90, fontsize=7.5, color='#444444', ha='right')

        ax_rest.set_title(f"Pair #{row+1} Rest-Frame Alignment (De-redshifted by DB $z$)\nSpectroscopic Features (Mg II, C IV, H$\\alpha$, etc.)", fontsize=11, fontweight='bold')
        ax_rest.set_xlabel("Rest Wavelength (nm)", fontsize=10)
        ax_rest.set_ylabel("Rest Flux", fontsize=10)
        ax_rest.legend(loc='upper right', fontsize=8.8)
        ax_rest.grid(True, linestyle='--', alpha=0.35)

    plt.suptitle("Spectral Comparison of New QSO Candidates vs Their Nearest Latent Neighbor Agreed QSOs\n(Revealing why SimCLR maps them to the same latent coordinates)", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_fig = RESULTS_DIR / "latent_nearest_neighbor_spectra_comparison.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved to {out_fig}.")

    brain_fig = BRAIN_DIR / "latent_nearest_neighbor_spectra_comparison.png"
    shutil.copy2(out_fig, brain_fig)
    print(f"Copied to {brain_fig}.")


if __name__ == "__main__":
    main()

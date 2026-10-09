"""Plot ONLY the Missed QSO spectra (DB: QSO, Pred: GALAXY) from the previous comparison figure,
removing the overlapping green Agreed Galaxy spectra so the purple spectra can be clearly examined.
"""
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID
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

    db_classes = meta['db_classes']
    db_redshifts = meta['db_redshifts']
    fits_paths = meta['fits_paths']
    oof_preds = preds['oof_preds']
    oof_probs = preds['oof_probs']
    classes = list(preds['classes'])

    # The exact 2 objects from the user's uploaded figure:
    selected_indices = [1569, 689]

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

    fig, axes = plt.subplots(len(selected_indices), 2, figsize=(18, 10))

    for row, q_idx in enumerate(selected_indices):
        z_q = db_redshifts[q_idx]
        cls_q = db_classes[q_idx]
        pred_q = oof_preds[q_idx]
        probs = {c: oof_probs[q_idx, classes.index(c)] for c in classes}

        prep_q = preprocess_spectrum(load_raw_spectrum(fits_paths[q_idx]))
        flux_q = prep_q['flux_clean']
        cont_q, res_q = compute_continuum_subtracted(flux_q)

        # -----------------------------------------------------------------
        # Left Column: Observed Frame (Channel 0 Flux & Channel 1 Residual)
        # -----------------------------------------------------------------
        ax_obs = axes[row, 0]
        lbl_obs = (f"Missed QSO #{row+1} (Target ID: {q_idx})\n"
                   f"DB: {cls_q} ($z_{{DB}}={z_q:.3f}$) | 2ch-Pred: {pred_q} "
                   f"[P(GAL)={probs['GALAXY']:.2f}, P(QSO)={probs['QSO']:.2f}]")
        
        ax_obs.plot(WL_GRID, flux_q, color='#7b1fa2', lw=1.3, label=f"Ch 0: Normalized Flux")
        ax_obs.plot(WL_GRID, res_q - 1.5, color='#ab47bc', lw=1.0, alpha=0.85, label=f"Ch 1: Residual (Offset -1.5)")
        ax_obs.axhline(-1.5, color='gray', linestyle=':', lw=0.8)

        ax_obs.set_title(f"Target #{row+1} Observed Frame (2-Channel Input to Model)\n{lbl_obs}", 
                         fontsize=11, fontweight='bold', loc='left')
        ax_obs.set_xlabel("Observed Wavelength (nm)", fontsize=10.5)
        ax_obs.set_ylabel("Normalized Flux & Residual", fontsize=10.5)
        ax_obs.set_xlim(370.0, 1270.0)
        
        # Adaptive y-limits based on robust percentiles
        y_p1 = np.nanpercentile(flux_q, 1)
        y_p99 = np.nanpercentile(flux_q, 99)
        ax_obs.set_ylim(-3.0, max(y_p99 + 1.5, 5.0))
        ax_obs.legend(loc='upper right', fontsize=9.5)
        ax_obs.grid(True, linestyle='--', alpha=0.35)

        # -----------------------------------------------------------------
        # Right Column: Rest Frame (De-redshifted with DB z)
        # -----------------------------------------------------------------
        ax_rest = axes[row, 1]
        rest_grid = np.linspace(110.0, 900.0, 3000)
        f_rest_q = np.interp(rest_grid * (1.0 + z_q), WL_GRID, flux_q, left=np.nan, right=np.nan)

        ax_rest.plot(rest_grid, f_rest_q, color='#7b1fa2', lw=1.4, 
                     label=f"Rest-Frame Spectrum (De-redshifted with $z_{{DB}}={z_q:.3f}$)")

        y_max = 5.2
        for idx_l, (name, line_wl) in enumerate(spectral_lines.items()):
            if 110.0 <= line_wl <= (850.0 if row == 0 else 550.0):
                ax_rest.axvline(line_wl, color='#455a64', linestyle=':', alpha=0.6, lw=1.0)
                # Stagger text heights slightly to prevent label overlap
                y_pos = y_max if (idx_l % 2 == 0) else y_max - 0.7
                ax_rest.text(line_wl, y_pos, name, rotation=90, fontsize=8, color='#37474f', ha='right', va='top')

        ax_rest.set_title(f"Target #{row+1} Rest-Frame Spectrum ($z={z_q:.3f}$)\nExamination of Broad & Narrow Emission Lines", 
                          fontsize=11, fontweight='bold', loc='left')
        ax_rest.set_xlabel("Rest-Frame Wavelength (nm)", fontsize=10.5)
        ax_rest.set_ylabel("Rest Flux", fontsize=10.5)
        ax_rest.set_xlim(110.0, 850.0 if row == 0 else 550.0)
        ax_rest.set_ylim(-1.5, 6.0)
        ax_rest.legend(loc='upper right', fontsize=9.5)
        ax_rest.grid(True, linestyle='--', alpha=0.35)

    plt.suptitle("Detailed Inspection: Individual Missed QSO Spectra (DB: QSO, 2ch-Pred: GALAXY)\nDisplayed Alone (Green Agreed Galaxy Spectra Removed)", 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig = RESULTS_DIR / "missed_qso_individual_spectra_alone.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved figure to {out_fig}")

    brain_fig = BRAIN_DIR / "missed_qso_individual_spectra_alone.png"
    shutil.copy2(out_fig, brain_fig)
    print(f"Copied figure to {brain_fig}")


if __name__ == "__main__":
    main()

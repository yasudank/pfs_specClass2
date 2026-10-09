"""Plot the individual spectra of the 10 objects used in panel [C] of the stacking experiment.
Displays both Observed Frame (left) and Rest Frame de-redshifted by z_DB (right) for each of the 10 targets.
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
    t_start = time.time()
    print("Loading survey metadata and predictions...")
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    pred_path = RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz"

    meta = np.load(meta_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True)

    db_classes = meta['db_classes']
    db_redshifts = meta['db_redshifts']
    fits_paths = meta['fits_paths']
    oof_preds = preds['oof_preds']
    oof_probs = preds['oof_probs']
    classes = list(preds['classes'])

    m_missed = np.where((db_classes == 'QSO') & (oof_preds == 'GALAXY'))[0]
    top10_indices = m_missed[:10]

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

    # 10 rows, 2 columns
    fig, axes = plt.subplots(10, 2, figsize=(18, 25))

    rest_grid = np.linspace(100.0, 950.0, 3000)

    for i, idx in enumerate(top10_indices):
        z_val = db_redshifts[idx]
        cls_val = db_classes[idx]
        pred_val = oof_preds[idx]
        p_gal = oof_probs[idx, classes.index('GALAXY')]
        p_qso = oof_probs[idx, classes.index('QSO')]
        p_star = oof_probs[idx, classes.index('STAR')]

        raw = load_raw_spectrum(fits_paths[idx])
        prep = preprocess_spectrum(raw)
        flux = prep['flux_clean']
        cont, res = compute_continuum_subtracted(flux)

        # -------------------------------------------------------------
        # Left: Observed Frame
        # -------------------------------------------------------------
        ax_obs = axes[i, 0]
        color_obs = '#1f77b4' if (i % 2 == 0) else '#0288d1'
        ax_obs.plot(WL_GRID, flux, color=color_obs, lw=1.1, label="Normalized Flux")
        ax_obs.plot(WL_GRID, res - 1.2, color='#78909c', lw=0.8, alpha=0.75, label="Residual (-1.2)")
        ax_obs.axhline(-1.2, color='gray', linestyle=':', lw=0.7)

        title_obs = (f"#{i+1:02d} [ID {idx}] Observed Frame | $z_{{DB}}={z_val:.3f}$ | "
                     f"2ch-Pred: {pred_val} (P(G)={p_gal:.2f}, P(Q)={p_qso:.2f}, P(S)={p_star:.2f})")
        ax_obs.set_title(title_obs, fontsize=9.5, fontweight='bold', loc='left')
        ax_obs.set_ylabel("Obs Flux", fontsize=8.5)
        ax_obs.set_xlim(370.0, 1260.0)

        # Robust y-limits for observed frame
        finite_f = flux[np.isfinite(flux)]
        y_lo = max(float(np.nanpercentile(finite_f, 1)) - 1.5, -2.5) if len(finite_f) > 0 else -2.5
        y_hi = min(float(np.nanpercentile(finite_f, 99)) + 0.8, 6.0) if len(finite_f) > 0 else 4.0
        ax_obs.set_ylim(y_lo, y_hi)
        ax_obs.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax_obs.legend(loc='upper right', fontsize=8)
        if i == 9:
            ax_obs.set_xlabel("Observed Wavelength (nm)", fontsize=9.5)

        # -------------------------------------------------------------
        # Right: Rest Frame (De-redshifted with DB z)
        # -------------------------------------------------------------
        ax_rest = axes[i, 1]
        obs_wl_cand = rest_grid * (1.0 + z_val)
        f_rest = np.interp(obs_wl_cand, WL_GRID, flux, left=np.nan, right=np.nan)

        color_rest = '#d81b60' if (i % 2 == 0) else '#8e24aa'
        ax_rest.plot(rest_grid, f_rest, color=color_rest, lw=1.2, label=f"Rest-Frame ($z={z_val:.3f}$)")

        title_rest = f"#{i+1:02d} [ID {idx}] Rest-Frame Spectrum ($z={z_val:.3f}$)"
        ax_rest.set_title(title_rest, fontsize=9.5, fontweight='bold', loc='left')
        ax_rest.set_ylabel("Rest Flux", fontsize=8.5)
        
        # Max rest wavelength visible given PFS red limit (1260 / (1+z))
        max_rest_wl = 1260.0 / (1.0 + z_val)
        min_rest_wl = 380.0 / (1.0 + z_val)
        ax_rest.set_xlim(115.0, min(max_rest_wl + 30.0, 850.0))

        # Robust y-limits
        finite_rest = f_rest[np.isfinite(f_rest)]
        y_r_lo = max(float(np.nanpercentile(finite_rest, 1)) - 0.4, -1.5) if len(finite_rest) > 0 else -1.5
        y_r_hi = min(float(np.nanpercentile(finite_rest, 99)) + 0.8, 6.0) if len(finite_rest) > 0 else 4.0
        ax_rest.set_ylim(y_r_lo, y_r_hi)
        ax_rest.grid(True, linestyle='--', alpha=0.3)

        # Draw spectral lines that fall inside the observed coverage
        for l_idx, (name, line_wl) in enumerate(spectral_lines.items()):
            if min_rest_wl <= line_wl <= max_rest_wl:
                ax_rest.axvline(line_wl, color='#455a64', linestyle=':', alpha=0.6, lw=0.9)
                y_text_pos = y_r_hi * 0.85 if (l_idx % 2 == 0) else y_r_hi * 0.70
                ax_rest.text(line_wl, y_text_pos, name, rotation=90, fontsize=7.5, color='#37474f', ha='right')

        if i == 0:
            ax_rest.legend(loc='upper right', fontsize=8)
        if i == 9:
            ax_rest.set_xlabel("Rest-Frame Wavelength (nm)", fontsize=9.5)

    plt.suptitle("Individual Spectra of the 10 Missed QSOs Used in Stack [C]\n(Left: Observed Frame 2-Channel View | Right: Rest-Frame Spectrum De-redshifted with DB $z$)",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig = RESULTS_DIR / "top10_missed_qsos_individual_spectra.png"
    plt.savefig(out_fig, dpi=180)
    plt.close()
    print(f"Saved figure to {out_fig}")

    brain_fig = BRAIN_DIR / "top10_missed_qsos_individual_spectra.png"
    shutil.copy2(out_fig, brain_fig)
    print(f"Copied figure to {brain_fig}")
    print(f"Done in {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()

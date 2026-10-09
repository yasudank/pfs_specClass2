"""Generate Clean Observed Rest-Frame Median Composite Spectra (2-Channel Classification Version)
Replicating the original rf_disagreement_restframe_spectra.png layout cleanly using the new 2-channel predictions:
1. Agreed QSOs: DB=QSO & 2ch-Pred=QSO (N=451)
2. New QSO Candidates: DB=GALAXY & 2ch-Pred=QSO (N=145)
3. Missed/Host-Dominated QSOs: DB=QSO & 2ch-Pred=GALAXY (N=626)
4. Agreed Galaxies: DB=GALAXY & 2ch-Pred=GALAXY (N=600 sampled)
5. Agreed Stars: DB=STAR & 2ch-Pred=STAR (N=600 sampled)
"""
import os
import sys
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID, N_GRID
from dataset import load_raw_spectrum, preprocess_spectrum

BRAIN_DIR = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")


def load_prep_worker(item):
    fits_path = item
    raw = load_raw_spectrum(fits_path)
    if raw is None:
        return None
    prep = preprocess_spectrum(raw)
    if prep is None:
        return None
    return prep['flux_clean']


def main():
    print("Generating Clean Observed Rest-Frame Composite Spectra for 2-Channel Predictions...")
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    pred_path = RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz"
    meta = np.load(meta_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True)

    db_classes = meta['db_classes']
    db_redshifts = meta['db_redshifts']
    fits_paths = meta['fits_paths']
    oof_preds = preds['oof_preds']

    groups = [
        ("Agreed QSOs: DB=QSO & 2ch-Pred=QSO", (db_classes == 'QSO') & (oof_preds == 'QSO'), '#d62728'),
        ("New QSO Candidates: DB=GALAXY & 2ch-Pred=QSO", (db_classes == 'GALAXY') & (oof_preds == 'QSO'), '#ff7f0e'),
        ("Missed/Host-Dominated QSOs: DB=QSO & 2ch-Pred=GALAXY", (db_classes == 'QSO') & (oof_preds == 'GALAXY'), '#9467bd'),
        ("Agreed Galaxies: DB=GALAXY & 2ch-Pred=GALAXY", (db_classes == 'GALAXY') & (oof_preds == 'GALAXY'), '#2ca02c'),
        ("Agreed Stars: DB=STAR & 2ch-Pred=STAR", (db_classes == 'STAR') & (oof_preds == 'STAR'), '#1f77b4'),
    ]

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

    fig, axes = plt.subplots(len(groups), 1, figsize=(16, 16), sharex=True)

    for i, (title, mask, col) in enumerate(groups):
        ax = axes[i]
        indices = np.where(mask)[0]
        n_group = len(indices)
        sub_idx = indices if n_group <= 600 else np.random.RandomState(42).choice(indices, 600, replace=False)
        sub_fits = [fits_paths[idx] for idx in sub_idx]
        sub_z = [db_redshifts[idx] for idx in sub_idx]

        with ProcessPoolExecutor(max_workers=16) as ex:
            fluxes = list(ex.map(load_prep_worker, sub_fits))

        rest_fluxes = []
        valid_z = []
        for f, z in zip(fluxes, sub_z):
            if f is None:
                continue
            use_z = z if (z is not None and np.isfinite(z) and z > 0.001) else 0.0
            if use_z > 0.001:
                valid_z.append(use_z)
            obs_wl_cand = rest_wl * (1.0 + use_z)
            f_interp = np.interp(obs_wl_cand, WL_GRID, f, left=np.nan, right=np.nan)
            rest_fluxes.append(f_interp)

        rest_fluxes = np.array(rest_fluxes)
        proto = np.nanmedian(rest_fluxes, axis=0)
        p16 = np.nanpercentile(rest_fluxes, 16, axis=0)
        p84 = np.nanpercentile(rest_fluxes, 84, axis=0)
        med_z = float(np.nanmedian(valid_z)) if len(valid_z) > 0 else 0.0

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
                ax.text(line_wl, ax.get_ylim()[1]*0.88 if ax.get_ylim()[1] > 0 else 1.5, name, rotation=90, fontsize=7.5, color='#444444', ha='right')

        finite_p = proto[np.isfinite(proto)]
        y_lo = np.nanpercentile(finite_p, 2) - 0.2 if len(finite_p) > 0 else -0.5
        y_hi = np.nanpercentile(finite_p, 98) + 0.6 if len(finite_p) > 0 else 2.0
        ax.set_ylim(max(y_lo, -0.5), min(y_hi, 4.0))

    axes[-1].set_xlabel("Rest-Frame Wavelength (nm)", fontsize=11)
    axes[-1].set_xlim(110.0, 950.0)
    plt.suptitle("Median Rest-Frame Spectra: Database vs 2-Channel Random Forest Classifications & Disagreements", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_path = RESULTS_DIR / "rf_disagreement_restframe_spectra_2channel.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved to {out_path}.")

    brain_path = BRAIN_DIR / "rf_disagreement_restframe_spectra_2channel.png"
    shutil.copy2(out_path, brain_path)
    print(f"Copied to {brain_path}.")


if __name__ == "__main__":
    main()

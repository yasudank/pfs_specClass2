"""Divide the 626 Missed QSOs (DB=QSO, 2ch-Pred=GALAXY) into 6 batches of 100 objects each,
and generate the median rest-frame composite spectrum for each batch.
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
    idx, fits_path, z_db = item
    raw = load_raw_spectrum(fits_path)
    if raw is None:
        return idx, None, z_db
    prep = preprocess_spectrum(raw)
    if prep is None:
        return idx, None, z_db
    return idx, prep['flux_clean'], z_db


def main():
    t_start = time.time()
    print("=" * 70)
    print("Analyzing 6 Batches of 100 Missed QSOs (DB=QSO, Pred=GALAXY)")
    print("=" * 70)

    # 1. Load Data
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    pred_path = RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz"

    meta = np.load(meta_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True)

    db_classes = meta['db_classes']
    db_redshifts = meta['db_redshifts']
    fits_paths = meta['fits_paths']
    oof_preds = preds['oof_preds']

    m_missed = np.where((db_classes == 'QSO') & (oof_preds == 'GALAXY'))[0]
    n_total_missed = len(m_missed)
    print(f"Total Missed QSOs: {n_total_missed}")

    # Use first 600 objects, split into 6 subsets of 100
    n_batches = 6
    batch_size = 100
    batches_idx = [m_missed[b * batch_size : (b + 1) * batch_size] for b in range(n_batches)]

    # 2. Parallel Loading of all 600 spectra
    all_items = []
    for b_idx in range(n_batches):
        for idx in batches_idx[b_idx]:
            all_items.append((idx, fits_paths[idx], db_redshifts[idx]))

    print(f"Loading and preprocessing {len(all_items)} spectra across 16 CPU cores...")
    t0 = time.time()
    loaded_dict = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(load_prep_worker, all_items, chunksize=25))

    for idx, f_clean, z_db in results:
        if f_clean is not None:
            loaded_dict[idx] = (f_clean, z_db)
    print(f"Loaded {len(loaded_dict)} spectra in {time.time() - t0:.1f}s.")

    # 3. Compute Median Rest-Frame Spectra per Batch
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

    batch_results = []
    for b in range(n_batches):
        b_indices = [i for i in batches_idx[b] if i in loaded_dict]
        b_fluxes = [loaded_dict[i][0] for i in b_indices]
        b_z = [loaded_dict[i][1] for i in b_indices]

        rest_fluxes = []
        valid_z = []
        for f, z in zip(b_fluxes, b_z):
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

        batch_results.append({
            'batch_num': b + 1,
            'n_valid': len(b_indices),
            'med_z': med_z,
            'proto': proto,
            'p16': p16,
            'p84': p84
        })

    # 4. Generate Diagnostic Figure (6 subplots, one per batch)
    colors = ['#8856a7', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b']
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)

    for b in range(n_batches):
        ax = axes[b]
        res = batch_results[b]
        col = colors[b]

        lbl = f"Batch #{res['batch_num']} (Objects {b*100+1}–{(b+1)*100}) | Valid N={res['n_valid']} | Med $z_{{DB}}={res['med_z']:.2f}$"
        ax.fill_between(rest_wl, res['p16'], res['p84'], color=col, alpha=0.20)
        ax.plot(rest_wl, res['proto'], color=col, lw=1.4, label=lbl)

        ax.set_ylabel("Rest Flux", fontsize=9.5)
        ax.set_title(lbl, fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=9.5)
        ax.grid(True, linestyle='--', alpha=0.4)

        for name, line_wl in spectral_lines.items():
            ax.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)
            if b == 0:
                ax.text(line_wl, ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 1.8,
                        name, rotation=90, fontsize=7.5, color='#444444', ha='right')

        finite_p = res['proto'][np.isfinite(res['proto'])]
        y_lo = max(float(np.nanpercentile(finite_p, 2)) - 0.2, -0.5) if len(finite_p) > 0 else -0.5
        y_hi = min(float(np.nanpercentile(finite_p, 98)) + 0.6, 3.5) if len(finite_p) > 0 else 2.5
        ax.set_ylim(y_lo, y_hi)

    axes[-1].set_xlabel("Rest-Frame Wavelength (nm)", fontsize=11.5)
    axes[-1].set_xlim(110.0, 950.0)
    plt.suptitle("Missed QSOs (DB: QSO, 2ch-Pred: GALAXY) Divided into 6 Batches of 100 Objects Each\nMedian Rest-Frame Spectra by Batch (De-redshifted with DB $z$)",
                 fontsize=13.5, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig = RESULTS_DIR / "missed_qsos_6batches_restframe_composite.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved figure to {out_fig}.")

    brain_fig = BRAIN_DIR / "missed_qsos_6batches_restframe_composite.png"
    shutil.copy2(out_fig, brain_fig)
    print(f"Copied figure to {brain_fig}.")

    print(f"Completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

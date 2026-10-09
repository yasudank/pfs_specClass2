"""Stack varying numbers of spectra (N = 1, 3, 10, 30, 100) from Batch #1 of Missed QSOs
(DB: QSO, 2ch-Pred: GALAXY) de-redshifted to rest frame using DB redshifts.
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
    print("Stacking Missed QSOs with Varying N: [1, 3, 10, 30, 100]")
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
    batch1_indices = m_missed[:100]

    # 2. Parallel Loading of Batch #1 spectra
    all_items = [(idx, fits_paths[idx], db_redshifts[idx]) for idx in batch1_indices]
    print(f"Loading {len(all_items)} spectra of Batch #1 across 16 CPU cores...")
    
    loaded_dict = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(load_prep_worker, all_items, chunksize=10))

    for idx, f_clean, z_db in results:
        if f_clean is not None:
            loaded_dict[idx] = (f_clean, z_db)
    print(f"Successfully loaded {len(loaded_dict)} spectra.")

    # 3. De-redshift each spectrum to rest frame grid
    N_REST = 3000
    rest_wl = np.linspace(100.0, 1000.0, N_REST)
    valid_batch_indices = [idx for idx in batch1_indices if idx in loaded_dict]

    rest_flux_list = []
    z_list = []
    for idx in valid_batch_indices:
        f, z = loaded_dict[idx]
        use_z = z if (z is not None and np.isfinite(z) and z > 0.001) else 0.0
        z_list.append(use_z)
        obs_wl_cand = rest_wl * (1.0 + use_z)
        f_interp = np.interp(obs_wl_cand, WL_GRID, f, left=np.nan, right=np.nan)
        rest_flux_list.append(f_interp)

    rest_flux_matrix = np.array(rest_flux_list)  # (N_valid, N_REST)
    n_total_valid = len(rest_flux_matrix)
    print(f"Valid spectra matrix shape: {rest_flux_matrix.shape}")

    # 4. Compute median stack for N in [1, 3, 10, 30, 100]
    n_subsets = [1, 3, 10, 30, 100]
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

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#7b1fa2']
    fig, axes = plt.subplots(len(n_subsets), 1, figsize=(16, 15), sharex=True)

    for i, n_val in enumerate(n_subsets):
        ax = axes[i]
        actual_n = min(n_val, n_total_valid)
        sub_matrix = rest_flux_matrix[:actual_n]
        sub_z = z_list[:actual_n]
        med_z = np.nanmedian(sub_z)

        if actual_n == 1:
            proto = sub_matrix[0]
            lbl = f"N = 1 Spectrum (Single Target, $z_{{DB}}={sub_z[0]:.3f}$)"
            ax.plot(rest_wl, proto, color=colors[i], lw=1.2, label=lbl)
        else:
            proto = np.nanmedian(sub_matrix, axis=0)
            p16 = np.nanpercentile(sub_matrix, 16, axis=0)
            p84 = np.nanpercentile(sub_matrix, 84, axis=0)
            lbl = f"N = {actual_n} Stacked (Median Composite, Med $z_{{DB}}={med_z:.2f}$)"
            ax.fill_between(rest_wl, p16, p84, color=colors[i], alpha=0.18, label="16th–84th Percentile")
            ax.plot(rest_wl, proto, color=colors[i], lw=1.3, label=lbl)

        # Plot vertical lines for reference spectral lines
        for l_idx, (name, line_wl) in enumerate(spectral_lines.items()):
            ax.axvline(line_wl, color='#546e7a', linestyle=':', alpha=0.55, lw=0.9)
            if i == 0:
                y_text_pos = 4.2 if (l_idx % 2 == 0) else 3.5
                ax.text(line_wl, y_text_pos, name, rotation=90, fontsize=8, color='#37474f', ha='right')

        ax.set_ylabel("Rest Flux", fontsize=10)
        ax.set_title(f"[{chr(65+i)}] {lbl}", fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=9.5)
        ax.grid(True, linestyle='--', alpha=0.35)

        # Consistent y limits across panels to observe noise reduction
        ax.set_ylim(-1.0, 5.0)

    axes[-1].set_xlabel("Rest-Frame Wavelength (nm)", fontsize=11.5)
    axes[-1].set_xlim(115.0, 850.0)

    plt.suptitle("Emergence of QSO Broad Emission Lines via Rest-Frame Stacking\nMissed QSOs (DB: QSO, 2ch-Pred: GALAXY) Stacked with N = 1, 3, 10, 30, 100", 
                 fontsize=13.5, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig = RESULTS_DIR / "missed_qso_stack_varying_N.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved figure to {out_fig}")

    brain_fig = BRAIN_DIR / "missed_qso_stack_varying_N.png"
    shutil.copy2(out_fig, brain_fig)
    print(f"Copied figure to {brain_fig}")
    print(f"Finished in {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()

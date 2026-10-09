"""Generate Median Composite Rest-Frame Spectra: Model Reconstruction vs Observed Spectra
for Disagreement Groups:
1. DB: GALAXY, Pred: QSO (N=145, New QSO Candidates)
2. DB: QSO, Pred: GALAXY (N=626, Host-Dominated / Missed QSOs)
3. DB: QSO, Pred: QSO (N=451, Confirmed Agreed QSOs)
4. DB: GALAXY, Pred: GALAXY (N=500, Confirmed Agreed Galaxies)
"""
import os
import sys
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import matplotlib.pyplot as plt

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID, N_GRID
from dataset import load_raw_spectrum, preprocess_spectrum
from models.restframe_model import RestFrameAlignModel, REST_WL, N_REST

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
    print("Generating Model Rest-Frame Median Composite Spectra for Disagreements")
    print("=" * 70)

    # 1. Load Metadata & Predictions
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    pred_path = RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz"
    meta = np.load(meta_path, allow_pickle=True)
    preds = np.load(pred_path, allow_pickle=True)

    db_classes = meta['db_classes']
    db_redshifts = meta['db_redshifts']
    fits_paths = meta['fits_paths']
    oof_preds = preds['oof_preds']

    # Define groups
    m_gal_qso = np.where((db_classes == 'GALAXY') & (oof_preds == 'QSO'))[0]
    m_qso_gal = np.where((db_classes == 'QSO') & (oof_preds == 'GALAXY'))[0]
    m_qso_qso = np.where((db_classes == 'QSO') & (oof_preds == 'QSO'))[0]
    
    np.random.seed(42)
    m_gal_gal_all = np.where((db_classes == 'GALAXY') & (oof_preds == 'GALAXY'))[0]
    m_gal_gal = np.random.choice(m_gal_gal_all, 500, replace=False)

    groups = {
        'gal_qso': {
            'title': 'New QSO Candidates: DB=GALAXY & 2ch-Pred=QSO',
            'indices': m_gal_qso,
            'color': '#ff7f0e', # Orange
            'total_n': len(m_gal_qso)
        },
        'qso_gal': {
            'title': 'Host-Dominated / Missed QSOs: DB=QSO & 2ch-Pred=GALAXY',
            'indices': m_qso_gal,
            'color': '#9467bd', # Purple
            'total_n': len(m_qso_gal)
        },
        'qso_qso': {
            'title': 'Agreed QSOs: DB=QSO & 2ch-Pred=QSO',
            'indices': m_qso_qso,
            'color': '#d62728', # Red
            'total_n': len(m_qso_qso)
        },
        'gal_gal': {
            'title': 'Agreed Galaxies: DB=GALAXY & 2ch-Pred=GALAXY',
            'indices': m_gal_gal,
            'color': '#2ca02c', # Green
            'total_n': len(m_gal_gal)
        }
    }

    # 2. Parallel Loading of Spectra
    all_items = []
    for gkey, ginfo in groups.items():
        for idx in ginfo['indices']:
            all_items.append((idx, fits_paths[idx], db_redshifts[idx]))

    print(f"Loading and preprocessing {len(all_items)} spectra across 16 CPU cores...")
    t0 = time.time()
    loaded_dict = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(load_prep_worker, all_items, chunksize=32))

    for idx, f_clean, z_db in results:
        if f_clean is not None:
            loaded_dict[idx] = (f_clean, z_db)
    print(f"Loaded {len(loaded_dict)} spectra successfully in {time.time() - t0:.1f}s.")

    # 3. Load RestFrame Model on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading RestFrameAlignModel on {device}...")
    ckpt_path = RESULTS_DIR / "restframe_align_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    rf_model = RestFrameAlignModel().to(device)
    rf_model.load_state_dict(ckpt['model_state_dict'])
    rf_model.eval()

    # 4. Compute Model Rest-Frame Spectra and Observed Rest-Frame Spectra per Group
    # Standard rest wavelength grid for observed de-redshifted spectra
    rest_wl_common = np.linspace(110.0, 950.0, 3000)

    computed_groups = {}
    for gkey, ginfo in groups.items():
        print(f"Processing group {gkey}: {ginfo['title']}...")
        indices = [i for i in ginfo['indices'] if i in loaded_dict]
        
        fluxes_clean = np.array([loaded_dict[i][0] for i in indices], dtype=np.float32)
        z_vals = np.array([loaded_dict[i][1] for i in indices], dtype=np.float32)

        # Batch GPU inference for Model Rest-Frame Spectrum T_rest
        t_model_list = []
        batch_size = 64
        with torch.no_grad():
            for b in range(0, len(fluxes_clean), batch_size):
                b_x = torch.from_numpy(fluxes_clean[b:b+batch_size]).unsqueeze(1).to(device)
                h = rf_model.encode(b_x)
                t_rest = rf_model.decode_rest(h).squeeze(1).cpu().numpy()
                t_model_list.append(t_rest)
        t_model_arr = np.concatenate(t_model_list, axis=0) # (N, 9214)

        # Resample Model Rest-Frame Spectrum onto common rest grid
        t_model_interp = []
        for t_m in t_model_arr:
            t_interp = np.interp(rest_wl_common, REST_WL, t_m)
            t_model_interp.append(t_interp)
        t_model_interp = np.array(t_model_interp)

        # Compute Observed De-redshifted Spectrum onto common rest grid
        t_obs_interp = []
        valid_z_count = 0
        for f, z in zip(fluxes_clean, z_vals):
            use_z = z if (z is not None and np.isfinite(z) and z > 0.001) else 0.0
            if use_z > 0.001:
                valid_z_count += 1
            obs_wl_cand = rest_wl_common * (1.0 + use_z)
            f_interp = np.interp(obs_wl_cand, WL_GRID, f, left=np.nan, right=np.nan)
            t_obs_interp.append(f_interp)
        t_obs_interp = np.array(t_obs_interp)

        # Calculate medians and percentiles
        # 1) Model Rest-Frame
        med_model = np.median(t_model_interp, axis=0)
        p16_model = np.percentile(t_model_interp, 16, axis=0)
        p84_model = np.percentile(t_model_interp, 84, axis=0)

        # 2) Observed Rest-Frame
        med_obs = np.nanmedian(t_obs_interp, axis=0)
        p16_obs = np.nanpercentile(t_obs_interp, 16, axis=0)
        p84_obs = np.nanpercentile(t_obs_interp, 84, axis=0)

        med_z = float(np.nanmedian(z_vals[z_vals > 0])) if np.any(z_vals > 0) else 0.0

        computed_groups[gkey] = {
            'info': ginfo,
            'n_valid': len(indices),
            'med_z': med_z,
            'med_model': med_model,
            'p16_model': p16_model,
            'p84_model': p84_model,
            'med_obs': med_obs,
            'p16_obs': p16_obs,
            'p84_obs': p84_obs
        }

    # 5. Diagnostic Plot: Model Rest-Frame vs Observed Composite Spectra
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

    # --- FIGURE 1: 4-Panel Model Rest-Frame vs Observed Rest-Frame Overlay ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 17), sharex=True)
    plot_order = ['gal_qso', 'qso_gal', 'qso_qso', 'gal_gal']

    for i, gkey in enumerate(plot_order):
        ax = axes[i]
        cg = computed_groups[gkey]
        col = cg['info']['color']

        # Plot Model Rest-Frame Spectrum (Solid Line with shaded envelope)
        ax.plot(rest_wl_common, cg['med_model'], color='#111111', lw=1.8, label=f"Model Rest-Frame Reconstruction ($T_{{rest}}$) | Median")
        ax.fill_between(rest_wl_common, cg['p16_model'], cg['p84_model'], color='#333333', alpha=0.15, label="Model 16-84% envelope")

        # Plot Observed De-redshifted Composite Spectrum (Colored dashed line)
        ax.plot(rest_wl_common, cg['med_obs'], color=col, lw=1.6, linestyle='--', label=f"Observed Spectra (de-redshifted by $z_{{DB}}$) | Median")
        ax.fill_between(rest_wl_common, cg['p16_obs'], cg['p84_obs'], color=col, alpha=0.18, label="Observed 16-84% envelope")

        title_str = f"[{chr(65+i)}] {cg['info']['title']} (Total N={cg['info']['total_n']}, Valid N={cg['n_valid']}, Median $z_{{DB}}$={cg['med_z']:.2f})"
        ax.set_title(title_str, fontsize=11.5, fontweight='bold', loc='left')
        ax.set_ylabel("Rest-Frame Flux", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.45)
        ax.legend(loc='upper right', fontsize=9.2, framealpha=0.9, ncol=2)

        # Spectral lines
        for name, line_wl in spectral_lines.items():
            ax.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)
            if i == 0:
                ax.text(line_wl, ax.get_ylim()[1] * 0.90 if ax.get_ylim()[1] > 0 else 1.0, 
                        name, rotation=90, fontsize=7.5, color='#444444', ha='right')

        # Robust y-limits
        y_vals = np.concatenate([cg['med_model'], cg['med_obs']])
        valid_y = y_vals[np.isfinite(y_vals)]
        if len(valid_y) > 0:
            y_lo = max(float(np.percentile(valid_y, 2)) - 0.25, -0.6)
            y_hi = min(float(np.percentile(valid_y, 98)) + 0.6, 3.8)
            ax.set_ylim(y_lo, y_hi)

    axes[-1].set_xlabel("Rest-Frame Wavelength (nm)", fontsize=11.5)
    axes[-1].set_xlim(110.0, 950.0)
    plt.suptitle("Model Rest-Frame Reconstruction vs Observed De-Redshifted Composite Spectra\nComparison for Classification Disagreement Groups and Reference Sets", 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig1 = RESULTS_DIR / "model_restframe_disagreement_composite.png"
    plt.savefig(out_fig1, dpi=200)
    plt.close()
    print(f"Saved Figure 1 to {out_fig1}.")

    # Copy to Brain Artifacts
    brain_fig1 = BRAIN_DIR / "model_restframe_disagreement_composite.png"
    shutil.copy2(out_fig1, brain_fig1)
    print(f"Copied Figure 1 to {brain_fig1}.")

    # --- FIGURE 2: Direct Model-Only Rest-Frame Spectra Comparison (Side-by-Side Focus on Disagreements) ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    
    # Panel 1: The two disagreement groups compared directly
    ax1 = axes[0]
    c_gq = computed_groups['gal_qso']
    c_qg = computed_groups['qso_gal']

    ax1.plot(rest_wl_common, c_gq['med_model'], color='#ff7f0e', lw=2.0, 
             label=f"New QSO Candidates (DB: GALAXY, Pred: QSO, N={c_gq['n_valid']}) - Model $T_{{rest}}$")
    ax1.fill_between(rest_wl_common, c_gq['p16_model'], c_gq['p84_model'], color='#ff7f0e', alpha=0.18)

    ax1.plot(rest_wl_common, c_qg['med_model'], color='#9467bd', lw=2.0, 
             label=f"Host-Dominated / Missed QSOs (DB: QSO, Pred: GALAXY, N={c_qg['n_valid']}) - Model $T_{{rest}}$")
    ax1.fill_between(rest_wl_common, c_qg['p16_model'], c_qg['p84_model'], color='#9467bd', alpha=0.18)

    ax1.set_title("Direct Comparison of Model-Decoded Rest-Frame Spectra for Disagreement Groups", fontsize=11.5, fontweight='bold', loc='left')
    ax1.set_ylabel("Model Rest Flux", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.45)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

    for name, line_wl in spectral_lines.items():
        ax1.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)
        ax1.text(line_wl, ax1.get_ylim()[1] * 0.88, name, rotation=90, fontsize=7.5, color='#444444', ha='right')

    # Panel 2: Difference / Residual between the Model Templates of the two Disagreement Groups
    ax2 = axes[1]
    diff_model = c_gq['med_model'] - c_qg['med_model']
    ax2.plot(rest_wl_common, diff_model, color='#d62728', lw=1.8, label="Model Spectral Difference: [New QSO Candidates] - [Host-Dominated QSOs]")
    ax2.axhline(0, color='black', linestyle='--', lw=1.0, alpha=0.7)
    ax2.set_title("Intrinsic Spectral Contrast Decoded by the Model (Revealing Broad Line & Continuum Differences)", fontsize=11.5, fontweight='bold', loc='left')
    ax2.set_ylabel(r"$\Delta$ Model Flux", fontsize=10)
    ax2.set_xlabel("Rest-Frame Wavelength (nm)", fontsize=11.5)
    ax2.set_xlim(110.0, 950.0)
    ax2.grid(True, linestyle='--', alpha=0.45)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

    for name, line_wl in spectral_lines.items():
        ax2.axvline(line_wl, color='gray', linestyle=':', alpha=0.55, lw=0.9)

    plt.suptitle("Model-Decoded Rest-Frame Templates: Analysis of Disagreement Classes", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_fig2 = RESULTS_DIR / "model_restframe_disagreement_contrast.png"
    plt.savefig(out_fig2, dpi=200)
    plt.close()
    print(f"Saved Figure 2 to {out_fig2}.")

    brain_fig2 = BRAIN_DIR / "model_restframe_disagreement_contrast.png"
    shutil.copy2(out_fig2, brain_fig2)
    print(f"Copied Figure 2 to {brain_fig2}.")

    print(f"Completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

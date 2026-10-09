"""Step 3: Generate Canonical Rest-Frame Templates from Full Dataset (Scratch)
- Built from 20,000 Galaxies, 20,000 Stars, and all 1,374 real QSOs
- Discards all initial 1200 prototype templates
- Stored into canonical_templates_scratch.npz
"""
import os
import sys
import numpy as np

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID, N_GRID


def build_templates_from_scratch():
    print("=== Step 3: Building Canonical Rest-Frame Templates from Scratch ===")
    cache_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    print(f"Loading representative full dataset from {cache_path}...")
    cache = np.load(cache_path, allow_pickle=True)
    flux = cache['flux_clean']
    classes = cache['classes']
    redshifts = cache['redshifts']

    N_REST = 4096
    rest_wl = np.linspace(100.0, 1260.0, N_REST)
    log_rest = np.log(rest_wl)

    templates = {}

    # 1. STAR template (z = 0)
    star_mask = (classes == 'STAR')
    star_raw = np.nanmedian(flux[star_mask], axis=0)
    star_template = np.interp(rest_wl, WL_GRID, star_raw, left=0.0, right=0.0).astype(np.float32)
    templates['STAR'] = star_template
    print(f"STAR canonical template created (from {np.sum(star_mask)} stars)")

    # 2. GALAXY template (shift-aligned using z)
    gal_mask = (classes == 'GALAXY') & np.isfinite(redshifts) & (redshifts > 0.05) & (redshifts < 1.5)
    gal_indices = np.where(gal_mask)[0]
    print(f"Aligning {len(gal_indices)} galaxies into rest frame...")
    gal_fluxes = []
    # Sample up to 5,000 galaxies for fast, ultra-clean composite
    sample_gal_idx = gal_indices if len(gal_indices) <= 5000 else np.random.choice(gal_indices, 5000, replace=False)
    for i in sample_gal_idx:
        z = redshifts[i]
        obs_wl = rest_wl * (1.0 + z)
        f_int = np.interp(obs_wl, WL_GRID, flux[i], left=np.nan, right=np.nan)
        gal_fluxes.append(f_int)

    gal_template = np.nanmedian(np.array(gal_fluxes), axis=0).astype(np.float32)
    templates['GALAXY'] = np.nan_to_num(gal_template, nan=0.0)
    print(f"GALAXY canonical template created (from {len(gal_fluxes)} rest-frame aligned galaxies)")

    # 3. QSO template (shift-aligned using z)
    qso_mask = (classes == 'QSO') & np.isfinite(redshifts) & (redshifts > 0.3) & (redshifts < 3.8)
    qso_indices = np.where(qso_mask)[0]
    print(f"Aligning {len(qso_indices)} QSOs into rest frame...")
    qso_fluxes = []
    sample_qso_idx = qso_indices if len(qso_indices) <= 5000 else np.random.choice(qso_indices, 5000, replace=False)
    for i in sample_qso_idx:
        z = redshifts[i]
        obs_wl = rest_wl * (1.0 + z)
        f_int = np.interp(obs_wl, WL_GRID, flux[i], left=np.nan, right=np.nan)
        qso_fluxes.append(f_int)

    qso_template = np.nanmedian(np.array(qso_fluxes), axis=0).astype(np.float32)
    templates['QSO'] = np.nan_to_num(qso_template, nan=0.0)
    print(f"QSO canonical template created (from {len(qso_fluxes)} rest-frame aligned QSOs)")

    out_path = RESULTS_DIR / "canonical_templates_scratch.npz"
    np.savez_compressed(
        out_path,
        rest_wl=rest_wl,
        log_rest=log_rest,
        STAR=templates['STAR'],
        GALAXY=templates['GALAXY'],
        QSO=templates['QSO']
    )
    print(f"Saved fresh canonical templates to {out_path}.")


if __name__ == "__main__":
    build_templates_from_scratch()

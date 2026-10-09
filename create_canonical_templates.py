"""Generates high-fidelity canonical rest-frame templates for STAR, GALAXY, and QSO
from the clean PFS spectra and stores them for decoder / cross-correlation matching.
"""
import os
import sys
import numpy as np

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, WL_GRID, LOG_WL_GRID, WL_MIN, WL_MAX, N_GRID

def build_templates():
    print("Building canonical rest-frame templates...")
    cache_path = RESULTS_DIR / "dataset_cache_clean_v2.npz"
    cache = np.load(cache_path)
    flux = cache['flux_clean']
    classes = cache['classes']
    redshifts = cache['redshifts']

    # Common rest grid: rest wavelength from 100 nm to 1260 nm
    N_REST = 4096
    rest_wl = np.linspace(100.0, 1260.0, N_REST)
    log_rest = np.log(rest_wl)

    templates = {}

    # 1. STAR template (z = 0)
    star_mask = (classes == 'STAR')
    star_raw = np.median(flux[star_mask], axis=0)
    # interpolate onto rest_wl
    star_template = np.interp(rest_wl, WL_GRID, star_raw, left=0.0, right=0.0).astype(np.float32)
    templates['STAR'] = star_template
    print(f"STAR template created (N={np.sum(star_mask)} stars)")

    # 2. GALAXY template (shift-aligned using valid z)
    gal_mask = (classes == 'GALAXY') & np.isfinite(redshifts) & (redshifts > 0.05) & (redshifts < 1.5)
    gal_fluxes = []
    for i in np.where(gal_mask)[0]:
        z = redshifts[i]
        obs_wl = rest_wl * (1.0 + z)
        f_int = np.interp(obs_wl, WL_GRID, flux[i], left=np.nan, right=np.nan)
        gal_fluxes.append(f_int)
    gal_template = np.nanmedian(np.array(gal_fluxes), axis=0).astype(np.float32)
    templates['GALAXY'] = np.nan_to_num(gal_template, nan=0.0)
    print(f"GALAXY template created (N={len(gal_fluxes)} shift-aligned galaxies)")

    # 3. QSO template (shift-aligned using valid z)
    qso_mask = (classes == 'QSO') & np.isfinite(redshifts) & (redshifts > 0.3) & (redshifts < 3.8)
    qso_fluxes = []
    for i in np.where(qso_mask)[0]:
        z = redshifts[i]
        obs_wl = rest_wl * (1.0 + z)
        f_int = np.interp(obs_wl, WL_GRID, flux[i], left=np.nan, right=np.nan)
        qso_fluxes.append(f_int)
    qso_template = np.nanmedian(np.array(qso_fluxes), axis=0).astype(np.float32)
    templates['QSO'] = np.nan_to_num(qso_template, nan=0.0)
    print(f"QSO template created (N={len(qso_fluxes)} shift-aligned QSOs)")

    # Save templates
    out_path = RESULTS_DIR / "canonical_templates.npz"
    np.savez_compressed(
        out_path,
        rest_wl=rest_wl,
        log_rest=log_rest,
        STAR=templates['STAR'],
        GALAXY=templates['GALAXY'],
        QSO=templates['QSO']
    )
    print(f"Saved canonical templates to {out_path}.")

if __name__ == "__main__":
    build_templates()

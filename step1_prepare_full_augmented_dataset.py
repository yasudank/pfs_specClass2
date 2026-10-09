"""Step 1: Build Full-Survey Representative & QSO-Augmented Dataset from Scratch
- GALAXY: 20,000 targets randomly sampled from full survey (107k)
- STAR:   20,000 targets randomly sampled from full survey (108k)
- QSO:    ALL 1,374 targets in the survey, augmented by 10x (wavelength jitter, flux perturbation, noise) = 13,740
- Total:  ~53,740 high-quality clean spectra for representation learning & clustering
"""
import os
import sys
import time
import sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, DB_PATH, N_GRID, LOG_WL_GRID, WL_GRID
from dataset import load_raw_spectrum, preprocess_spectrum


def process_target_worker(item):
    obj_id, cls_name, z, fits_p = item
    raw = load_raw_spectrum(fits_p)
    if raw is None:
        return None
    prep = preprocess_spectrum(raw)
    if prep is None:
        return None
    return {
        'objId': obj_id,
        'class': cls_name,
        'redshift': float(z) if z is not None and np.isfinite(z) else 0.0,
        'fits_path': fits_p,
        'flux': prep['flux'],
        'flux_clean': prep['flux_clean'],
        'weight': prep['weight'],
        'mask': prep['mask']
    }


def augment_qso_spectrum(flux_clean, weight, mask, rng):
    """Augment QSO spectrum:
    - Small shift in log-wavelength: +/- 24 pixels (~0.007 in ln(lambda))
    - Small multiplicative tilt: 1 + slope * (lambda - center)
    - Small Gaussian noise proportional to 1/sqrt(weight)
    """
    shift = rng.randint(-24, 25)
    f_shifted = np.roll(flux_clean, shift)
    m_shifted = np.roll(mask, shift)
    w_shifted = np.roll(weight, shift)
    if shift > 0:
        f_shifted[:shift] = 0.0
        m_shifted[:shift] = 0.0
        w_shifted[:shift] = 0.0
    elif shift < 0:
        f_shifted[shift:] = 0.0
        m_shifted[shift:] = 0.0
        w_shifted[shift:] = 0.0

    # Multiplicative continuum tilt (+/- 5%)
    tilt_slope = rng.normal(0.0, 0.05)
    grid_norm = np.linspace(-1.0, 1.0, N_GRID)
    tilt = 1.0 + tilt_slope * grid_norm
    f_tilted = f_shifted * tilt

    # Small noise
    noise = rng.normal(0.0, 0.02, size=N_GRID)
    f_noisy = f_tilted + noise * (m_shifted > 0)
    f_noisy[m_shifted == 0] = 0.0

    # Renormalize
    valid_f = f_noisy[m_shifted > 0]
    p80 = np.percentile(valid_f, 80.0) if len(valid_f) > 0 else 1.0
    if p80 > 1e-4:
        f_noisy = f_noisy / p80
    return f_noisy.astype(np.float32), w_shifted.astype(np.float32), m_shifted.astype(np.float32)


def main():
    print("=== Step 1: Building Full-Scale Representative Dataset with QSO Augmentation ===")
    t0 = time.time()
    rng = np.random.RandomState(42)

    # 1. Query targets from DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("Querying all QSOs...")
    cur.execute("SELECT objId, classificationName, bestRedshift, fits_path FROM target_summary WHERE has_fits = 1 AND classificationName = 'QSO';")
    qso_targets = cur.fetchall()
    print(f"Total QSOs found: {len(qso_targets)}")

    print("Sampling 20,000 Galaxies...")
    cur.execute("SELECT objId, classificationName, bestRedshift, fits_path FROM target_summary WHERE has_fits = 1 AND classificationName = 'GALAXY' ORDER BY RANDOM() LIMIT 20000;")
    gal_targets = cur.fetchall()

    print("Sampling 20,000 Stars...")
    cur.execute("SELECT objId, classificationName, bestRedshift, fits_path FROM target_summary WHERE has_fits = 1 AND classificationName = 'STAR' ORDER BY RANDOM() LIMIT 20000;")
    star_targets = cur.fetchall()

    conn.close()

    all_raw_targets = qso_targets + gal_targets + star_targets
    print(f"Total queried targets: {len(all_raw_targets)} (QSO:{len(qso_targets)}, GAL:{len(gal_targets)}, STAR:{len(star_targets)})")

    # 2. Parallel Loading & Preprocessing
    print("Preprocessing spectra in parallel (16 workers)...")
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_target_worker, all_raw_targets))

    valid_results = [r for r in results if r is not None]
    print(f"Preprocessed {len(valid_results)} valid spectra in {time.time() - t0:.1f}s.")

    qso_valid = [r for r in valid_results if r['class'] == 'QSO']
    gal_valid = [r for r in valid_results if r['class'] == 'GALAXY']
    star_valid = [r for r in valid_results if r['class'] == 'STAR']
    print(f"Valid counts: QSO={len(qso_valid)}, GAL={len(gal_valid)}, STAR={len(star_valid)}")

    # 3. Augment QSOs by 10x
    print("Performing 10x spectral augmentation on all QSOs to prevent minority burial...")
    augmented_qso_list = []
    # Original QSOs
    augmented_qso_list.extend(qso_valid)

    # 9 additional augmented copies per QSO
    AUG_FACTOR = 9
    for copy_idx in range(AUG_FACTOR):
        for orig in qso_valid:
            aug_flux, aug_w, aug_m = augment_qso_spectrum(orig['flux_clean'], orig['weight'], orig['mask'], rng)
            augmented_qso_list.append({
                'objId': orig['objId'],
                'class': 'QSO',
                'redshift': orig['redshift'],
                'fits_path': orig['fits_path'],
                'flux': aug_flux,
                'flux_clean': aug_flux,
                'weight': aug_w,
                'mask': aug_m
            })

    print(f"Total QSOs after 10x augmentation: {len(augmented_qso_list)}")

    # 4. Assemble Final Full Dataset
    final_dataset = gal_valid + star_valid + augmented_qso_list
    rng.shuffle(final_dataset)
    print(f"Total assembled dataset: {len(final_dataset)} spectra (GAL:{len(gal_valid)}, STAR:{len(star_valid)}, QSO_AUG:{len(augmented_qso_list)})")

    # Pack into numpy arrays
    N = len(final_dataset)
    flux_matrix = np.empty((N, N_GRID), dtype=np.float32)
    weight_matrix = np.empty((N, N_GRID), dtype=np.float32)
    mask_matrix = np.empty((N, N_GRID), dtype=np.float32)
    obj_ids = np.empty(N, dtype=np.int64)
    classes = np.empty(N, dtype=object)
    redshifts = np.empty(N, dtype=np.float32)
    fits_paths = np.empty(N, dtype=object)

    for i, it in enumerate(final_dataset):
        flux_matrix[i] = it['flux_clean']
        weight_matrix[i] = it['weight']
        mask_matrix[i] = it['mask']
        obj_ids[i] = it['objId']
        classes[i] = it['class']
        redshifts[i] = it['redshift']
        fits_paths[i] = it['fits_path']

    out_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    np.savez_compressed(
        out_path,
        flux_clean=flux_matrix,
        weight=weight_matrix,
        mask=mask_matrix,
        objId=obj_ids,
        classes=classes,
        redshifts=redshifts,
        fits_paths=fits_paths
    )
    print(f"Saved complete augmented dataset to {out_path} ({os.path.getsize(out_path) / 1024**2:.1f} MB) in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()

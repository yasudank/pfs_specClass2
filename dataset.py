"""PFS Spectrum Dataset and Preprocessing Pipeline."""
import os
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from typing import Optional, Tuple, Dict, Any, List

from config import BASE_DIR, DB_PATH, LOG_WL_GRID, WL_GRID, N_GRID, WL_MIN, WL_MAX


def load_raw_spectrum(fits_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Loads raw spectrum from FITS file and performs basic sanity checks."""
    full_path = os.path.join(BASE_DIR, fits_path) if not os.path.isabs(fits_path) else fits_path
    if not os.path.exists(full_path):
        return None

    try:
        with fits.open(full_path, memmap=True) as hdul:
            wl = hdul['WAVELENGTH'].data.astype(np.float64)
            flux = hdul['FLUX'].data.astype(np.float32)
            mask = hdul['MASK'].data.astype(np.int32)
            covar = hdul['COVAR'].data.astype(np.float32)

            var = covar[0] if covar.ndim > 1 else covar
            return {
                'wl': wl,
                'flux': flux,
                'mask': mask,
                'var': var,
            }
    except Exception:
        return None


def preprocess_spectrum(
    raw_spec: Dict[str, np.ndarray],
    target_log_grid: np.ndarray = LOG_WL_GRID,
    norm_percentile: float = 80.0
) -> Optional[Dict[str, np.ndarray]]:
    """
    Preprocesses spectrum:
    1. Mask pixels where mask != 0 or non-finite flux/variance.
    2. Resample onto uniform log-wavelength grid ln(lambda).
    3. Normalize flux and calculate corresponding inverse-variance weights.
    """
    wl = raw_spec['wl']
    flux = raw_spec['flux']
    mask = raw_spec['mask']
    var = raw_spec['var']

    # 1. Identify good pixels from FITS mask and validity
    good = (mask == 0) & np.isfinite(flux) & np.isfinite(var) & (var > 0)
    # Filter out astronomical/pipeline variance explosion (e.g. var > 1e12)
    good = good & (var < 1e12)

    if np.sum(good) < 100:
        return None

    ln_wl = np.log(wl)

    # Filter strictly to valid ranges
    valid_range = (wl >= WL_MIN - 10.0) & (wl <= WL_MAX + 10.0)
    good = good & valid_range
    if np.sum(good) < 100:
        return None

    good_ln_wl = ln_wl[good]
    good_flux = flux[good]
    good_var = var[good]

    # Interpolate flux and variance onto target log-wavelength grid
    resampled_flux = np.interp(target_log_grid, good_ln_wl, good_flux, left=0.0, right=0.0)
    resampled_var = np.interp(target_log_grid, good_ln_wl, good_var, left=np.inf, right=np.inf)

    # Interpolate valid mask indicator
    good_indicator = np.zeros_like(wl, dtype=np.float32)
    good_indicator[good] = 1.0
    resampled_valid = np.interp(target_log_grid, ln_wl, good_indicator, left=0.0, right=0.0)
    mask_weight = (resampled_valid > 0.8).astype(np.float32)

    # 2. Adaptive Masking: Detect whether spectrum is BMN (medium-res) or BRN (low-res)
    from config import ARM_MASK_BRN, ARM_MASK_BMN, WL_GRID

    # Check for presence of gap between 670 and 690 nm to detect BMN
    raw_has_mid_red = np.any(good & (wl >= 670.0) & (wl <= 690.0))
    is_bmn = not raw_has_mid_red

    active_mask_regions = ARM_MASK_BMN if is_bmn else ARM_MASK_BRN
    for w_low, w_high in active_mask_regions:
        in_arm_join = (WL_GRID >= w_low) & (WL_GRID <= w_high)
        mask_weight[in_arm_join] = 0.0

    # Calculate weight (inverse variance)
    safe_var = np.where(resampled_var > 0, resampled_var, np.inf)
    inv_var = np.where((mask_weight > 0) & np.isfinite(safe_var), 1.0 / safe_var, 0.0).astype(np.float32)

    # 3. Robust normalization: normalize by percentile of valid flux
    valid_flux = resampled_flux[mask_weight > 0]
    if len(valid_flux) == 0:
        return None

    norm_val = np.percentile(valid_flux, norm_percentile)
    if not np.isfinite(norm_val) or norm_val <= 0:
        norm_val = np.median(np.abs(valid_flux))
    if not np.isfinite(norm_val) or norm_val <= 1e-12:
        norm_val = 1.0

    norm_flux = (resampled_flux / norm_val).astype(np.float32)
    norm_weight = (inv_var * (norm_val ** 2)).astype(np.float32)

    # 4. Zero out flux at masked pixels & clip unmasked runaway spikes (|flux| > 15)
    norm_flux[mask_weight == 0] = 0.0
    runaway_spikes = (np.abs(norm_flux) > 15.0)
    if np.any(runaway_spikes):
        norm_flux[runaway_spikes] = 0.0
        norm_weight[runaway_spikes] = 0.0
        mask_weight[runaway_spikes] = 0.0

    # Clip unphysical negative overshoots from sky oversubtraction
    norm_flux = np.clip(norm_flux, -2.0, 15.0)

    # 5. Variance-Suppressed Flux (OH airglow / Sky Line Residual Suppression):
    # Continuum regions have typical weight; sky lines have tiny weight.
    # Compute arm-aware continuum weight (Visible vs NIR) to accurately identify sky lines
    from config import WL_GRID
    is_nir = (WL_GRID > 900.0) & (mask_weight > 0)
    is_vis = (WL_GRID <= 900.0) & (mask_weight > 0)

    med_w_vis = np.median(norm_weight[is_vis]) if np.any(is_vis) else 1.0
    med_w_nir = np.median(norm_weight[is_nir]) if np.any(is_nir) else 1.0
    med_w_vis = max(med_w_vis, 1e-6)
    med_w_nir = max(med_w_nir, 1e-6)

    ref_w = np.where(WL_GRID > 900.0, med_w_nir, med_w_vis)
    w_ratio = np.where(mask_weight > 0, norm_weight / ref_w, 0.0)

    # Damp pixels where weight is much lower than continuum
    damp_factor = np.clip(2.0 * np.sqrt(np.maximum(w_ratio, 0.0)), 0.0, 1.0).astype(np.float32)
    clean_flux = (norm_flux * damp_factor).astype(np.float32)
    clean_flux = np.clip(clean_flux, -2.0, 15.0)

    return {
        'flux': norm_flux,             # (N_GRID,) float32 raw normalized
        'flux_clean': clean_flux,     # (N_GRID,) float32 noise/sky-dampened
        'weight': norm_weight,         # (N_GRID,) float32 inverse variance
        'mask': mask_weight,           # (N_GRID,) float32
        'norm_scale': np.float32(norm_val)
    }


def query_target_samples(
    db_path: str = str(DB_PATH),
    n_per_class: int = 1000,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """Queries balanced target samples across STAR, GALAXY, and QSO from SQLite efficiently."""
    np.random.seed(random_seed)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    targets = []
    classes = ['STAR', 'GALAXY', 'QSO']

    for cls in classes:
        # Get all matching rowids for the class
        c.execute("SELECT rowid FROM target_summary WHERE has_fits = 1 AND classificationName = ?;", (cls,))
        all_rowids = [r[0] for r in c.fetchall()]
        if not all_rowids:
            continue

        selected_count = min(len(all_rowids), n_per_class)
        chosen_rowids = np.random.choice(all_rowids, size=selected_count, replace=False).tolist()

        # Batch query in chunks
        chunk_size = 500
        for i in range(0, len(chosen_rowids), chunk_size):
            chunk = chosen_rowids[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
            SELECT objId, classificationName, bestRedshift, fits_path
            FROM target_summary
            WHERE rowid IN ({placeholders});
            """
            c.execute(query, chunk)
            for r in c.fetchall():
                targets.append({
                    'objId': r[0],
                    'classificationName': r[1],
                    'bestRedshift': r[2],
                    'fits_path': r[3]
                })

    conn.close()
    np.random.shuffle(targets)
    return targets


class PFSSpectrumDataset(Dataset):
    """PyTorch Dataset for PFS Spectra with preprocessed caching."""

    def __init__(self, targets: List[Dict[str, Any]], preloaded_data: Optional[Dict[str, np.ndarray]] = None):
        self.targets = targets
        self.preloaded = preloaded_data

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preloaded is not None:
            flux = torch.from_numpy(self.preloaded['flux'][idx])
            flux_clean = torch.from_numpy(self.preloaded.get('flux_clean', self.preloaded['flux'])[idx])
            weight = torch.from_numpy(self.preloaded['weight'][idx])
            mask = torch.from_numpy(self.preloaded['mask'][idx])
        else:
            meta = self.targets[idx]
            raw = load_raw_spectrum(meta['fits_path'])
            if raw is None:
                # Return dummy zeros if file error
                flux = torch.zeros(N_GRID, dtype=torch.float32)
                flux_clean = torch.zeros(N_GRID, dtype=torch.float32)
                weight = torch.zeros(N_GRID, dtype=torch.float32)
                mask = torch.zeros(N_GRID, dtype=torch.float32)
            else:
                proc = preprocess_spectrum(raw)
                if proc is None:
                    flux = torch.zeros(N_GRID, dtype=torch.float32)
                    flux_clean = torch.zeros(N_GRID, dtype=torch.float32)
                    weight = torch.zeros(N_GRID, dtype=torch.float32)
                    mask = torch.zeros(N_GRID, dtype=torch.float32)
                else:
                    flux = torch.from_numpy(proc['flux'])
                    flux_clean = torch.from_numpy(proc['flux_clean'])
                    weight = torch.from_numpy(proc['weight'])
                    mask = torch.from_numpy(proc['mask'])

        meta = self.targets[idx]
        return {
            'flux': flux,                # (N_GRID,)
            'flux_clean': flux_clean,    # (N_GRID,) sky-dampened
            'weight': weight,            # (N_GRID,)
            'mask': mask,                # (N_GRID,)
            'objId': meta['objId'],
            'class': meta['classificationName'],
            'redshift': meta['bestRedshift'] if meta['bestRedshift'] is not None else np.nan
        }

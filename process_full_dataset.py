"""High-Throughput Batch Processing and Inference for the Full PFS Dataset (~216,476 targets).

Pipeline per spectrum:
1. Load FITS file & apply adaptive arm masking (BRN / BMN detection).
2. Variance-suppressed flux resampling (370–1260 nm, N_GRID=4096).
3. 128-dimensional latent representation extraction via SpectralSimCLR (GPU).
4. Subtype clustering & broad class assignment (GMM).
5. Canonical rest-frame template matching for redshift estimation.
6. Writes results to SQLite database and stores embeddings.
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

import torch
from config import BASE_DIR, DB_PATH, RESULTS_DIR, WL_GRID, N_GRID
from dataset import load_raw_spectrum, preprocess_spectrum
from models.contrastive_model import SpectralSimCLR
from models.clustering import SpectralClusterClassifier


def process_spectrum_worker(fits_rel_path: str):
    """Worker function for parallel FITS loading and preprocessing."""
    try:
        raw = load_raw_spectrum(fits_rel_path)
        if raw is None:
            return None
        prep = preprocess_spectrum(raw)
        if prep is None:
            return None
        return {
            'flux_clean': prep['flux_clean'],
            'mask': prep['mask'],
            'norm_scale': prep['norm_scale']
        }
    except Exception:
        return None


def run_full_processing(
    batch_size: int = 5000,
    max_targets: int = None,
    num_workers: int = 16,
    db_out_path: str = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting PFS Full Dataset Processing on {device} (workers: {num_workers}) ===")

    if db_out_path is None:
        db_out_path = RESULTS_DIR / "pfs_classifications_216k.sqlite3"
    db_out_path = Path(db_out_path)

    # 1. Initialize SQLite results database
    conn_out = sqlite3.connect(db_out_path)
    cur_out = conn_out.cursor()
    cur_out.execute("""
    CREATE TABLE IF NOT EXISTS classifications (
        objId INTEGER PRIMARY KEY,
        fits_path TEXT,
        db_class TEXT,
        db_redshift REAL,
        pred_class TEXT,
        pred_confidence REAL,
        cluster_id INTEGER,
        cluster_name TEXT,
        cluster_confidence REAL,
        anomaly_score REAL,
        est_redshift REAL,
        est_redshift_quality REAL,
        arm_mode TEXT,
        valid_pixel_frac REAL,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cur_out.execute("CREATE INDEX IF NOT EXISTS idx_pred_class ON classifications (pred_class);")
    cur_out.execute("CREATE INDEX IF NOT EXISTS idx_cluster ON classifications (cluster_id);")
    conn_out.commit()

    # Get already processed objIds to enable resuming
    cur_out.execute("SELECT objId FROM classifications;")
    processed_set = set(r[0] for r in cur_out.fetchall())
    print(f"Resuming: {len(processed_set)} targets already processed in {db_out_path}.")

    # 2. Query target list from source SQLite
    conn_src = sqlite3.connect(DB_PATH)
    cur_src = conn_src.cursor()
    query = "SELECT objId, classificationName, bestRedshift, fits_path FROM target_summary WHERE has_fits = 1 ORDER BY objId;"
    cur_src.execute(query)
    all_targets = cur_src.fetchall()
    conn_src.close()

    # Filter out already processed
    targets_to_process = [t for t in all_targets if t[0] not in processed_set]
    if max_targets is not None:
        targets_to_process = targets_to_process[:max_targets]

    total_count = len(targets_to_process)
    print(f"Targets remaining to process: {total_count} / {len(all_targets)} total.")
    if total_count == 0:
        print("All targets already processed.")
        conn_out.close()
        return

    # 3. Load Models
    print("Loading models...")
    model_path = RESULTS_DIR / "spectral_simclr_clean_v2.pt"
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    encoder = SpectralSimCLR(embed_dim=128, proj_dim=64).to(device)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    clf = SpectralClusterClassifier.load(str(RESULTS_DIR / "spectral_cluster_classifier.joblib"))

    # Canonical templates for redshift cross-correlation
    tmpl_data = np.load(RESULTS_DIR / "canonical_templates.npz")
    rest_wl = tmpl_data['rest_wl']
    templates = {
        'STAR': tmpl_data['STAR'],
        'GALAXY': tmpl_data['GALAXY'],
        'QSO': tmpl_data['QSO']
    }

    # Redshift grids
    gal_z_grid = np.linspace(0.0, 1.5, 200)
    qso_z_grid = np.linspace(0.2, 3.8, 250)

    # Precompute redshifted template banks for ultra-fast vector correlation
    print("Precomputing redshift template banks...")
    gal_bank = np.array([np.interp(WL_GRID, rest_wl * (1.0 + z), templates['GALAXY'], left=0.0, right=0.0) for z in gal_z_grid])
    gal_bank = (gal_bank - np.mean(gal_bank, axis=1, keepdims=True)) / np.maximum(np.std(gal_bank, axis=1, keepdims=True), 1e-6)

    qso_bank = np.array([np.interp(WL_GRID, rest_wl * (1.0 + z), templates['QSO'], left=0.0, right=0.0) for z in qso_z_grid])
    qso_bank = (qso_bank - np.mean(qso_bank, axis=1, keepdims=True)) / np.maximum(np.std(qso_bank, axis=1, keepdims=True), 1e-6)

    # Convert template banks to GPU tensors for millisecond batch matching
    gal_bank_t = torch.from_numpy(gal_bank).float().to(device)  # (N_z, 4096)
    qso_bank_t = torch.from_numpy(qso_bank).float().to(device)  # (N_z, 4096)

    # 4. Process in Chunks
    start_time = time.time()
    n_done = 0

    for chunk_start in range(0, total_count, batch_size):
        chunk_targets = targets_to_process[chunk_start:chunk_start + batch_size]
        c_t0 = time.time()

        # Step A: Parallel FITS reading & preprocessing
        fits_paths = [t[3] for t in chunk_targets]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            proc_results = list(executor.map(process_spectrum_worker, fits_paths))

        # Filter valid
        valid_items = []
        for meta, res in zip(chunk_targets, proc_results):
            if res is not None:
                valid_items.append((meta, res))

        if not valid_items:
            continue

        n_valid = len(valid_items)
        flux_matrix = np.array([item[1]['flux_clean'] for item in valid_items], dtype=np.float32)
        masks = [item[1]['mask'] for item in valid_items]

        # Step B: GPU forward pass for embeddings
        with torch.no_grad():
            x_gpu = torch.from_numpy(flux_matrix).unsqueeze(1).to(device)
            embeds = encoder.encode(x_gpu).cpu().numpy()  # (n_valid, 128)

            # Step C: GPU template cross-correlation for redshifts
            # Normalize flux matrix along spectral axis
            f_mean = flux_matrix.mean(axis=1, keepdims=True)
            f_std = np.maximum(flux_matrix.std(axis=1, keepdims=True), 1e-6)
            f_norm_t = torch.from_numpy((flux_matrix - f_mean) / f_std).float().to(device)  # (n_valid, 4096)

            # Dot products: (n_valid, N_z)
            gal_corrs = torch.matmul(f_norm_t, gal_bank_t.T) / N_GRID
            qso_corrs = torch.matmul(f_norm_t, qso_bank_t.T) / N_GRID

            best_gal_val, best_gal_idx = gal_corrs.max(dim=1)
            best_qso_val, best_qso_idx = qso_corrs.max(dim=1)

            best_gal_z = gal_z_grid[best_gal_idx.cpu().numpy()]
            best_gal_r = best_gal_val.cpu().numpy()

            best_qso_z = qso_z_grid[best_qso_idx.cpu().numpy()]
            best_qso_r = best_qso_val.cpu().numpy()

        # Step D: Clustering & Metadata Assembly
        db_rows = []
        for i, (meta, prep) in enumerate(valid_items):
            obj_id, db_cls, db_z, fits_p = meta
            emb = embeds[i]
            mask = masks[i]

            # GMM prediction
            pred = clf.predict(emb)
            pred_cls = pred['predicted_broad_class']

            # Redshift selection
            if pred_cls == 'STAR':
                est_z = 0.0
                est_r = 1.0
            elif pred_cls == 'QSO':
                est_z = float(best_qso_z[i])
                est_r = float(max(best_qso_r[i], 0.0))
            else:
                est_z = float(best_gal_z[i])
                est_r = float(max(best_gal_r[i], 0.0))

            # Arm mode
            in_gap = (WL_GRID >= 670.0) & (WL_GRID <= 690.0)
            is_bmn = bool(np.all(mask[in_gap] == 0))
            arm_mode = "BMN" if is_bmn else "BRN"

            valid_frac = float(np.mean(mask > 0))

            db_rows.append((
                obj_id,
                fits_p,
                db_cls,
                float(db_z) if db_z is not None and np.isfinite(db_z) else None,
                pred_cls,
                float(pred['broad_class_confidence']),
                int(pred['cluster_id']),
                pred['cluster_name'],
                float(pred['cluster_confidence']),
                float(pred['anomaly_score']),
                round(est_z, 4),
                round(est_r, 3),
                arm_mode,
                round(valid_frac, 4)
            ))

        # Step E: Insert into SQLite
        cur_out.executemany("""
        INSERT OR REPLACE INTO classifications (
            objId, fits_path, db_class, db_redshift,
            pred_class, pred_confidence,
            cluster_id, cluster_name, cluster_confidence,
            anomaly_score, est_redshift, est_redshift_quality,
            arm_mode, valid_pixel_frac
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, db_rows)
        conn_out.commit()

        n_done += len(db_rows)
        c_time = time.time() - c_t0
        rate = len(db_rows) / max(c_time, 0.001)
        total_elapsed = time.time() - start_time
        overall_rate = n_done / max(total_elapsed, 0.001)
        est_remaining_min = (total_count - n_done) / max(overall_rate, 0.001) / 60.0

        print(f"[{n_done:6d}/{total_count}] Batch processed: {len(db_rows)} in {c_time:.2f}s ({rate:.1f} spec/s) | ETA: {est_remaining_min:.1f} min")

    conn_out.close()
    print(f"=== Completed processing {n_done} spectra in {time.time() - start_time:.1f}s ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    run_full_processing(
        batch_size=args.batch_size,
        max_targets=args.max_targets,
        num_workers=args.workers
    )

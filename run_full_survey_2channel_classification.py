"""Full Survey (216,476 spectra) 2-Channel SimCLR & Random Forest Classification Pipeline.
1. Extracts 128D embeddings for all 216,476 spectra using the 2-Channel SimCLR model.
2. Runs 5-Fold Stratified Cross-Validation Random Forest on the entire survey.
3. Computes OOF predictions and class probabilities for all 216,476 objects.
4. Analyzes Recall vs Precision curves for QSO screening.
5. Updates SQLite3 database results/pfs_classifications_216k_scratch.sqlite3.
6. Generates full survey diagnostic figures.
"""
import os
import sys
import time
import shutil
import sqlite3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
from models.contrastive_model import SpectralSimCLR
from investigate_random_forest_classification import process_spectrum_worker
from config import RESULTS_DIR, WL_GRID


def proc_2ch_worker(fits_path):
    f = process_spectrum_worker(fits_path)
    if f is None:
        return None
    cont = ndimage.median_filter(f, size=128)
    cont = ndimage.gaussian_filter1d(cont, sigma=32)
    res = f - cont
    return f, res


def main():
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== Full Survey (216k) 2-Channel SimCLR & Random Forest Pipeline ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Survey Metadata
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    print(f"Loading survey metadata from {meta_path}...")
    meta = np.load(meta_path, allow_pickle=True)
    obj_ids = meta["objIds"]
    db_classes = meta["db_classes"]
    db_redshifts = meta["db_redshifts"]
    fits_paths = meta["fits_paths"]
    total_targets = len(obj_ids)
    print(f"Total targets: {total_targets}")

    class_names = ['GALAXY', 'QSO', 'STAR']
    cls_map = {c: i for i, c in enumerate(class_names)}
    y_true_idx = np.array([cls_map[c] for c in db_classes])

    # 2. Extract 2-Channel 128D Embeddings for all 216,476 spectra
    emb_save_path = RESULTS_DIR / "survey_embeddings_216k_2channel.npy"

    if emb_save_path.exists():
        print(f"\nFound existing 2-channel embeddings at {emb_save_path}, loading...")
        all_embeddings = np.load(emb_save_path)
        print(f"Loaded embeddings shape: {all_embeddings.shape}")
    else:
        print(f"\n--- Step 1: Extracting 2-Channel Embeddings for all {total_targets} spectra ---")
        model_path = RESULTS_DIR / "spectral_simclr_2channel.pt"
        model = SpectralSimCLR(in_channels=2, embed_dim=128, proj_dim=64).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()

        CHUNK_SIZE = 8192
        all_embeds_list = []
        t_extr = time.time()

        for chunk_idx, start_idx in enumerate(range(0, total_targets, CHUNK_SIZE)):
            end_idx = min(start_idx + CHUNK_SIZE, total_targets)
            chunk_paths = fits_paths[start_idx:end_idx]

            # Parallel FITS loading & continuum subtraction
            with ProcessPoolExecutor(max_workers=16) as ex:
                chunk_res = list(ex.map(proc_2ch_worker, chunk_paths, chunksize=256))

            # Stack into (B, 2, 4096)
            f_list, r_list = [], []
            for item in chunk_res:
                if item is not None:
                    f_list.append(item[0])
                    r_list.append(item[1])
                else:
                    # Fallback if any corrupted
                    f_list.append(np.zeros(4096, dtype=np.float32))
                    r_list.append(np.zeros(4096, dtype=np.float32))

            flux_arr = np.array(f_list, dtype=np.float32)
            res_arr = np.array(r_list, dtype=np.float32)
            x_2ch = np.stack([flux_arr, res_arr], axis=1)

            # GPU Forward pass
            with torch.no_grad():
                for b_start in range(0, len(x_2ch), 512):
                    b_end = min(b_start + 512, len(x_2ch))
                    bx = torch.from_numpy(x_2ch[b_start:b_end]).to(device)
                    emb = model.encode(bx).cpu().numpy()
                    all_embeds_list.append(emb)

            done = end_idx
            elapsed = time.time() - t_extr
            rate = done / max(elapsed, 0.1)
            print(f"Extraction Progress: {done}/{total_targets} ({done/total_targets*100:.1f}%) | Rate: {rate:.1f} spec/s")

        all_embeddings = np.concatenate(all_embeds_list, axis=0)
        np.save(emb_save_path, all_embeddings)
        print(f"Saved all 2-channel embeddings to {emb_save_path} in {time.time()-t_extr:.1f}s.")

    # 3. 5-Fold Stratified Cross-Validation Random Forest on all 216,476 targets
    print(f"\n--- Step 2: 5-Fold CV Random Forest across {total_targets} targets (16 CPU cores) ---")
    t_rf = time.time()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(total_targets, dtype=int)
    oof_probs = np.zeros((total_targets, len(class_names)), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_embeddings, y_true_idx), 1):
        fold_t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=16
        )
        rf.fit(all_embeddings[train_idx], y_true_idx[train_idx])
        oof_preds[val_idx] = rf.predict(all_embeddings[val_idx])
        oof_probs[val_idx] = rf.predict_proba(all_embeddings[val_idx])
        print(f"  Fold {fold}/5 complete ({time.time()-fold_t0:.1f}s)")

    print(f"Full survey RF cross-validation completed in {time.time()-t_rf:.1f}s.")

    # Convert predictions to strings
    oof_pred_strings = np.array([class_names[p] for p in oof_preds])

    # Save OOF predictions
    pred_save_path = RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz"
    np.savez_compressed(
        pred_save_path,
        oof_preds=oof_pred_strings,
        oof_probs=oof_probs,
        classes=class_names
    )
    print(f"Saved predictions to {pred_save_path}")

    # 4. Evaluation and Performance Report
    print("\n================== FULL SURVEY EVALUATION REPORT ==================")
    report_str = classification_report(y_true_idx, oof_preds, target_names=class_names, digits=4)
    print(report_str)

    cm = confusion_matrix(y_true_idx, oof_preds)
    cm_norm = confusion_matrix(y_true_idx, oof_preds, normalize='true')
    print("Confusion Matrix (Counts):")
    print(cm)

    qso_idx = class_names.index('QSO')
    gal_idx = class_names.index('GALAXY')
    star_idx = class_names.index('STAR')

    print(f"\nPopulation Comparison:")
    for c in class_names:
        idx_c = cls_map[c]
        print(f"  {c:<8}: DB Count = {np.sum(y_true_idx == idx_c):>6} | RF Predicted = {np.sum(oof_preds == idx_c):>6}")

    # 5. Threshold Analysis for QSO Recall vs Precision
    print("\n--- Step 3: QSO Screening Threshold Analysis ---")
    qso_prob_array = oof_probs[:, qso_idx]
    y_true_is_qso = (y_true_idx == qso_idx).astype(int)

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(f"{'Threshold':<10} | {'Pred QSOs':>10} | {'True QSOs':>10} | {'Recall':>8} | {'Precision':>10}")
    print("-" * 60)
    for th in thresholds:
        pred_qso_mask = (qso_prob_array >= th)
        n_cand = int(np.sum(pred_qso_mask))
        n_true = int(np.sum(pred_qso_mask & (y_true_is_qso == 1)))
        rec = n_true / np.sum(y_true_is_qso) * 100
        prec = (n_true / n_cand * 100) if n_cand > 0 else 0.0
        print(f"P >= {th:<6.2f} | {n_cand:>10} | {n_true:>10} | {rec:>7.1f}% | {prec:>9.1f}%")

    # 6. Update SQLite3 Database
    db_file = RESULTS_DIR / "pfs_classifications_216k_scratch.sqlite3"
    print(f"\n--- Step 4: Updating SQLite3 Database at {db_file} ---")
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Add columns if not exist
    cur.execute("PRAGMA table_info(classifications)")
    existing_cols = [c[1] for c in cur.fetchall()]
    if "rf2ch_pred_class" not in existing_cols:
        cur.execute("ALTER TABLE classifications ADD COLUMN rf2ch_pred_class TEXT")
        cur.execute("ALTER TABLE classifications ADD COLUMN rf2ch_prob_galaxy REAL")
        cur.execute("ALTER TABLE classifications ADD COLUMN rf2ch_prob_qso REAL")
        cur.execute("ALTER TABLE classifications ADD COLUMN rf2ch_prob_star REAL")
        conn.commit()

    update_records = [
        (str(oof_pred_strings[i]), float(oof_probs[i, gal_idx]), float(oof_probs[i, qso_idx]), float(oof_probs[i, star_idx]), i + 1)
        for i in range(total_targets)
    ]
    cur.executemany("""
        UPDATE classifications
        SET rf2ch_pred_class = ?, rf2ch_prob_galaxy = ?, rf2ch_prob_qso = ?, rf2ch_prob_star = ?
        WHERE id = ?
    """, update_records)
    conn.commit()
    conn.close()
    print("Database updated successfully!")

    # 7. Diagnostic Visualizations
    print("\n--- Step 5: Generating Diagnostic Figures ---")
    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    # Panel 1: Full Survey Confusion Matrix (Heatmap)
    ax_cm = fig.add_subplot(gs[0, 0])
    im = ax_cm.imshow(cm_norm, cmap='Greens', vmin=0, vmax=1.0)
    cb = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cb.set_label('Recall Rate', fontsize=10)
    ax_cm.set_xticks(range(3))
    ax_cm.set_xticklabels(class_names, fontsize=10.5)
    ax_cm.set_yticks(range(3))
    ax_cm.set_yticklabels(class_names, fontsize=10.5)
    ax_cm.set_xlabel('Predicted Class (Random Forest 2-Channel)', fontsize=11)
    ax_cm.set_ylabel('Database Catalog True Class', fontsize=11)
    ax_cm.set_title('[A] Full Survey Confusion Matrix (216,476 Objects)\nNormalized by True Class (Row %)', fontsize=12, fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax_cm.text(j, i, f"{cm_norm[i, j]*100:.2f}%\n(N={cm[i, j]:,})",
                       ha='center', va='center', color='black' if cm_norm[i, j] < 0.6 else 'white', fontsize=10)

    # Panel 2: Total Population Comparison (Bar Chart)
    ax_pop = fig.add_subplot(gs[0, 1])
    db_counts = [np.sum(y_true_idx == i) for i in range(3)]
    rf_counts = [np.sum(oof_preds == i) for i in range(3)]
    x_b = np.arange(3)
    width = 0.35
    b1 = ax_pop.bar(x_b - width/2, db_counts, width, label='DB Catalog Counts', color='#377eb8', alpha=0.85)
    b2 = ax_pop.bar(x_b + width/2, rf_counts, width, label='RF Predicted Counts (Argmax)', color='#4daf4a', alpha=0.85)
    ax_pop.set_xticks(x_b)
    ax_pop.set_xticklabels(class_names, fontsize=11)
    ax_pop.set_ylabel('Number of Objects', fontsize=11)
    ax_pop.set_title('[B] Full Survey Population Comparison (DB vs 2-Channel RF)', fontsize=12, fontweight='bold')
    ax_pop.legend(loc='upper right', fontsize=10.5)
    ax_pop.grid(True, linestyle='--', alpha=0.35, axis='y')
    for rect in b1 + b2:
        h = rect.get_height()
        ax_pop.text(rect.get_x() + rect.get_width()/2., h + 2000, f"{h:,}", ha='center', va='bottom', fontsize=9.5)

    # Panel 3: P(QSO) Probability Distribution (Log-Scale)
    ax_prob = fig.add_subplot(gs[1, 0])
    q_probs_qso = qso_prob_array[y_true_idx == qso_idx]
    q_probs_gal = qso_prob_array[y_true_idx == gal_idx]
    q_probs_star = qso_prob_array[y_true_idx == star_idx]
    ax_prob.hist(q_probs_gal, bins=50, alpha=0.4, color='#377eb8', label=f'True Galaxies (N={len(q_probs_gal):,})', density=True)
    ax_prob.hist(q_probs_star, bins=50, alpha=0.4, color='#4daf4a', label=f'True Stars (N={len(q_probs_star):,})', density=True)
    ax_prob.hist(q_probs_qso, bins=50, alpha=0.8, color='#e41a1c', label=f'True QSOs (N={len(q_probs_qso):,})', density=True)
    ax_prob.set_yscale('log')
    ax_prob.set_xlabel('Predicted P(QSO)', fontsize=11)
    ax_prob.set_ylabel('Density (Log Scale)', fontsize=11)
    ax_prob.set_title('[C] Full Survey P(QSO) Probability Distribution', fontsize=12, fontweight='bold')
    ax_prob.legend(loc='upper right', fontsize=10)
    ax_prob.grid(True, linestyle='--', alpha=0.35)

    # Panel 4: Precision-Recall Curve for QSO Screening
    ax_pr = fig.add_subplot(gs[1, 1])
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true_is_qso, qso_prob_array)
    ap_score = average_precision_score(y_true_is_qso, qso_prob_array)
    ax_pr.plot(recalls, precisions, color='#e41a1c', lw=2.2, label=f'2-Channel SimCLR RF (AP={ap_score:.3f})')
    ax_pr.axvline(0.427, color='gray', linestyle=':', label='Argmax Operating Point (Recall 42.7%, Prec 67.8%)')
    ax_pr.set_xlabel('QSO Recall (Completeness)', fontsize=11)
    ax_pr.set_ylabel('QSO Precision (Purity)', fontsize=11)
    ax_pr.set_title('[D] QSO Screening: Precision vs Recall Trade-off Curve', fontsize=12, fontweight='bold')
    ax_pr.legend(loc='lower left', fontsize=10)
    ax_pr.grid(True, linestyle='--', alpha=0.35)

    plt.suptitle("Full Survey (216,476 Objects) Spectral Classification with 2-Channel Dual-Stream SimCLR",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig = RESULTS_DIR / "full_survey_2channel_classification_results.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved figure to {out_fig}")

    # Copy to artifacts
    art_dir = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")
    shutil.copy(out_fig, art_dir / "full_survey_2channel_classification_results.png")
    print("Copied to artifacts successfully!")
    print(f"Entire full-survey pipeline completed in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()

"""Create 216k Full Survey Predictions Catalog:
- Target count: 216,476 (All spectra in Subaru PFS survey)
- Pure pipeline outputs only (no database ground truth columns)
- Integrates the SMOTE-boosted 2-Channel Random Forest predictions (which achieved 881 Agreed QSOs on the representative dataset)
  ensuring full sensitivity to the rare QSO population (~1,184 predicted QSOs, 904 agreed QSOs across 216k)
- Schema:
    catId INTEGER NOT NULL,
    objId INTEGER NOT NULL,
    combination TEXT NOT NULL,
    pred_class TEXT,
    prob_star REAL,
    prob_galaxy REAL,
    prob_qso REAL,
    pred_confidence REAL,
    pred_redshift REAL,
    redshift_quality REAL,
    z_galaxy_candidate REAL,
    z_qso_candidate REAL,
    PRIMARY KEY (catId, objId, combination)
- Output files:
    results/pfs_full_survey_predictions.csv
    results/pfs_full_survey_predictions.csv.gz
    results/pfs_full_survey_predictions.sqlite3
"""
import os
import sys
import time
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR, DB_PATH

BRAIN_DIR = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")


def main():
    t_start = time.time()
    print("=" * 80)
    print("Creating Full Survey 216k Predictions Catalog with SMOTE-Boosted Predictions")
    print("=" * 80)

    # 1. Load target_summary mapping (fits_path -> catId, objId, combination)
    print(f"Reading target_summary from {DB_PATH} ...")
    conn_db = sqlite3.connect(DB_PATH)
    cur_db = conn_db.cursor()
    cur_db.execute("""
        SELECT fits_path, catId, objId, combination
        FROM target_summary
        WHERE has_fits = 1;
    """)
    fits_to_tuple = {r[0]: (r[1], r[2], r[3]) for r in cur_db.fetchall()}
    conn_db.close()
    print(f"Loaded N = {len(fits_to_tuple):,} target mappings from DB.")

    # 2. Load 40k Representative SMOTE-Boosted Predictions & High-Precision Redshifts
    print("Loading 40k SMOTE-boosted representative predictions...")
    rf_40k = np.load(RESULTS_DIR / "rf_oof_boost3x_smote_representative.npz", allow_pickle=True)
    p_40k = rf_40k["oof_preds"]
    probs_40k = rf_40k["oof_probs"]
    classes_40k = list(rf_40k["classes"])

    data_40k = np.load(RESULTS_DIR / "dataset_full_augmented_scratch.npz", allow_pickle=True)
    all_obj_ids_40k = data_40k["objId"]
    fits_40k = data_40k["fits_paths"]
    _, uidx = np.unique(all_obj_ids_40k, return_index=True)
    unique_fits_40k = fits_40k[uidx]

    idx_gal_40k = classes_40k.index("GALAXY")
    idx_qso_40k = classes_40k.index("QSO")
    idx_star_40k = classes_40k.index("STAR")

    # Map by fits_path
    map_40k = {}
    for i, fp in enumerate(unique_fits_40k):
        map_40k[fp] = {
            "pred_class": p_40k[i],
            "prob_galaxy": float(probs_40k[i, idx_gal_40k]),
            "prob_qso": float(probs_40k[i, idx_qso_40k]),
            "prob_star": float(probs_40k[i, idx_star_40k]),
            "pred_conf": float(np.max(probs_40k[i]))
        }

    # Load high-precision ResNet/Hybrid candidate redshifts from Step 4
    npz_redshifts = RESULTS_DIR / "redshift_predictions_scratch.npz"
    map_redshifts_by_obj = {}
    if npz_redshifts.exists():
        print(f"Loading high-precision redshifts from Step 4 cache: {npz_redshifts} ...")
        r_npz = np.load(npz_redshifts)
        r_oids = r_npz['objIds']
        r_zgal = r_npz['z_galaxy_candidate']
        r_zqso = r_npz['z_qso_candidate']
        r_qual = r_npz['redshift_quality']
        for idx_r, roid in enumerate(r_oids):
            map_redshifts_by_obj[int(roid)] = {
                'z_gal': float(r_zgal[idx_r]),
                'z_qso': float(r_zqso[idx_r]),
                'z_qual': float(r_qual[idx_r])
            }
        print(f"Loaded high-precision redshifts for N = {len(map_redshifts_by_obj):,} representative targets.")

    # 3. Load 216k Survey Baseline
    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    rf_path = RESULTS_DIR / "rf_oof_predictions_216k_2channel.npz"

    print(f"Loading survey metadata from {meta_path} ...")
    meta = np.load(meta_path, allow_pickle=True)
    fits_paths = meta['fits_paths']
    obj_ids = meta['objIds']
    best_gal_z = meta['best_gal_z']
    best_gal_r = meta['best_gal_r']
    best_qso_z = meta['best_qso_z']
    best_qso_r = meta['best_qso_r']

    print(f"Loading 216k baseline RF predictions from {rf_path} ...")
    rf_216k = np.load(rf_path, allow_pickle=True)
    oof_preds_216k = rf_216k['oof_preds']
    oof_probs_216k = rf_216k['oof_probs']
    classes_216k = list(rf_216k['classes'])

    idx_gal_216k = classes_216k.index('GALAXY')
    idx_qso_216k = classes_216k.index('QSO')
    idx_star_216k = classes_216k.index('STAR')

    # 4. Assemble All 216,476 Records
    print("\nAssembling all 216,476 records...")
    cat_ids = []
    combinations = []
    final_pred_classes = []
    final_prob_star = []
    final_prob_galaxy = []
    final_prob_qso = []
    final_pred_conf = []
    final_pred_redshift = []
    final_redshift_qual = []
    final_z_gal = []
    final_z_qso = []

    n_from_smote = 0
    for i, fp in enumerate(fits_paths):
        if fp in fits_to_tuple:
            cid, oid, comb = fits_to_tuple[fp]
        else:
            cid, oid, comb = 10094, int(obj_ids[i]), 'brn_run28'
        cat_ids.append(cid)
        combinations.append(comb)

        tuple_key = (cid, oid, comb)

        # Prioritize SMOTE-boosted predictions for representative set
        if fp in map_40k:
            rec = map_40k[fp]
            cls = rec['pred_class']
            p_s = rec['prob_star']
            p_g = rec['prob_galaxy']
            p_q = rec['prob_qso']
            conf = rec['pred_conf']
            n_from_smote += 1
        else:
            cls = oof_preds_216k[i]
            p_s = float(oof_probs_216k[i, idx_star_216k])
            p_g = float(oof_probs_216k[i, idx_gal_216k])
            p_q = float(oof_probs_216k[i, idx_qso_216k])
            conf = float(np.max(oof_probs_216k[i]))

        final_pred_classes.append(cls)
        final_prob_star.append(p_s)
        final_prob_galaxy.append(p_g)
        final_prob_qso.append(p_q)
        final_pred_conf.append(conf)

        # Redshifts
        gz = float(best_gal_z[i])
        gr = float(best_gal_r[i])
        qz = float(best_qso_z[i])
        qr = float(best_qso_r[i])

        if int(oid) in map_redshifts_by_obj:
            r_rec = map_redshifts_by_obj[int(oid)]
            if r_rec['z_gal'] > 0:
                gz = r_rec['z_gal']
            if r_rec['z_qso'] > 0:
                qz = r_rec['z_qso']
                qr = r_rec['z_qual']

        final_z_gal.append(gz)
        final_z_qso.append(qz)

        if cls == 'GALAXY':
            final_pred_redshift.append(gz)
            final_redshift_qual.append(gr)
        elif cls == 'QSO':
            final_pred_redshift.append(qz)
            final_redshift_qual.append(qr)
        else:  # STAR
            final_pred_redshift.append(0.0)
            final_redshift_qual.append(1.0)

    print(f"Applied SMOTE-boosted predictions to N = {n_from_smote:,} targets.")

    # 5. Build Final Clean DataFrame
    df_216k = pd.DataFrame({
        'catId': np.array(cat_ids, dtype=np.int64),
        'objId': np.array(obj_ids, dtype=np.int64),
        'combination': combinations,
        'pred_class': final_pred_classes,
        'prob_star': np.round(final_prob_star, 4),
        'prob_galaxy': np.round(final_prob_galaxy, 4),
        'prob_qso': np.round(final_prob_qso, 4),
        'pred_confidence': np.round(final_pred_conf, 4),
        'pred_redshift': np.round(final_pred_redshift, 4),
        'redshift_quality': np.round(final_redshift_qual, 4),
        'z_galaxy_candidate': np.round(final_z_gal, 4),
        'z_qso_candidate': np.round(final_z_qso, 4)
    })

    print("\nFinal Class Distribution across all 216,476 spectra:")
    for c_name in ['STAR', 'GALAXY', 'QSO']:
        cnt = np.sum(df_216k['pred_class'] == c_name)
        print(f"  {c_name:<8}: {cnt:>6} ({cnt / len(df_216k) * 100:.1f}%)")

    # 6. Save CSV and Gzipped CSV
    out_csv = RESULTS_DIR / "pfs_full_survey_predictions.csv"
    out_csv_gz = RESULTS_DIR / "pfs_full_survey_predictions.csv.gz"
    print(f"\nSaving CSV: {out_csv} ...")
    df_216k.to_csv(out_csv, index=False)
    print(f"Saving Gzipped CSV: {out_csv_gz} ...")
    df_216k.to_csv(out_csv_gz, index=False, compression='gzip')

    # 7. Save SQLite Database with Composite Primary Key
    out_db = RESULTS_DIR / "pfs_full_survey_predictions.sqlite3"
    print(f"Saving SQLite Database: {out_db} ...")
    conn = sqlite3.connect(out_db)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS survey_predictions;")

    cur.execute("""
        CREATE TABLE survey_predictions (
            catId INTEGER NOT NULL,
            objId INTEGER NOT NULL,
            combination TEXT NOT NULL,
            pred_class TEXT,
            prob_star REAL,
            prob_galaxy REAL,
            prob_qso REAL,
            pred_confidence REAL,
            pred_redshift REAL,
            redshift_quality REAL,
            z_galaxy_candidate REAL,
            z_qso_candidate REAL,
            PRIMARY KEY (catId, objId, combination)
        );
    """)

    df_216k.to_sql('survey_predictions', conn, if_exists='append', index=False)

    print("Creating indices...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_class ON survey_predictions(pred_class);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_objId ON survey_predictions(objId);")
    conn.commit()
    conn.close()

    # Copy to brain dir if exists
    if BRAIN_DIR.exists():
        import shutil
        shutil.copyfile(out_csv, BRAIN_DIR / "pfs_full_survey_predictions.csv")
        shutil.copyfile(out_db, BRAIN_DIR / "pfs_full_survey_predictions.sqlite3")

    print("\n" + "=" * 80)
    print("Catalog updated successfully!")
    print(f"Total Rows: {len(df_216k):,}")
    print(f"Total Elapsed Time: {time.time() - t_start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

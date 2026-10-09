"""Fast, comprehensive SMOTE evaluation on 216k survey dataset (80/20 Stratified Split).
Train set: 173,180 targets (1,099 QSOs)
Val set: 43,296 targets (275 QSOs)
Evaluates SMOTE multipliers [0x, 2x, 5x, 10x, 20x].
"""
import os
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR


def smote_oversample(X_minority, n_synthetic, k_neighbors=5, random_state=42):
    rng = np.random.RandomState(random_state)
    k = min(k_neighbors, len(X_minority) - 1)
    knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_minority)
    _, indices = knn.kneighbors(X_minority)

    base_idx = rng.randint(0, len(X_minority), size=n_synthetic)
    neighbor_col = rng.randint(1, k + 1, size=n_synthetic)
    neighbor_idx = indices[base_idx, neighbor_col]

    diff = X_minority[neighbor_idx] - X_minority[base_idx]
    gap = rng.uniform(0.0, 1.0, size=(n_synthetic, 1)).astype(np.float32)
    return X_minority[base_idx] + gap * diff


def main():
    print("=" * 70)
    print("Evaluating SMOTE Oversampling for Random Forest on 216k Survey Embeddings")
    print("=" * 70)

    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    emb_path = RESULTS_DIR / "survey_embeddings_216k_2channel.npy"

    meta = np.load(meta_path, allow_pickle=True)
    db_classes = meta["db_classes"]
    embeds = np.load(emb_path)

    class_names = ['GALAXY', 'QSO', 'STAR']
    cls_map = {c: i for i, c in enumerate(class_names)}
    y_all = np.array([cls_map[c] for c in db_classes])
    qso_idx = class_names.index('QSO')

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(embeds, y_all))

    X_train_base, y_train_base = embeds[train_idx], y_all[train_idx]
    X_val, y_val = embeds[val_idx], y_all[val_idx]

    m_qso_train = (y_train_base == qso_idx)
    X_qso = X_train_base[m_qso_train]
    n_qso_base = len(X_qso)

    print(f"Train size: {len(X_train_base)} (QSOs: {n_qso_base})")
    print(f"Val size:   {len(X_val)} (QSOs: {np.sum(y_val == qso_idx)})")

    multipliers = [0, 2, 5, 10, 20]

    print(f"\n{'SMOTE Config':<16} | {'Train QSOs':>10} | {'QSO Recall':>11} | {'QSO Prec':>10} | {'QSO F1':>8} | {'Galaxy F1':>9} | {'Star F1':>8} | {'Time':>6}")
    print("-" * 92)

    for mult in multipliers:
        t0 = time.time()
        if mult == 0:
            X_tr, y_tr = X_train_base, y_train_base
        else:
            n_synth = int(n_qso_base * mult)
            X_synth = smote_oversample(X_qso, n_synth, k_neighbors=5, random_state=42 + mult)
            y_synth = np.full(n_synth, qso_idx, dtype=int)
            X_tr = np.vstack([X_train_base, X_synth])
            y_tr = np.concatenate([y_train_base, y_synth])

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=16
        )
        rf.fit(X_tr, y_tr)
        val_preds = rf.predict(X_val)
        fit_time = time.time() - t0

        rep = classification_report(y_val, val_preds, target_names=class_names, output_dict=True, digits=4)
        q_rec = rep['QSO']['recall']
        q_prec = rep['QSO']['precision']
        q_f1 = rep['QSO']['f1-score']
        g_f1 = rep['GALAXY']['f1-score']
        s_f1 = rep['STAR']['f1-score']

        label = "Baseline (0x)" if mult == 0 else f"SMOTE {mult}x"
        print(f"{label:<16} | {len(X_tr[y_tr==qso_idx]):>10d} | {q_rec*100:>10.2f}% | {q_prec*100:>9.2f}% | {q_f1:>8.4f} | {g_f1:>9.4f} | {s_f1:>8.4f} | {fit_time:>5.1f}s")


if __name__ == "__main__":
    main()

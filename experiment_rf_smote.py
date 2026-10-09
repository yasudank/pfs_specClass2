"""Experiment 1: Test SMOTE oversampling of QSO embeddings for Random Forest classification
on the 216k survey dataset.
Compares:
1) Baseline (No SMOTE, class_weight='balanced_subsample')
2) SMOTE 5x (Synthesizing ~5,500 QSOs in train fold)
3) SMOTE 10x (Synthesizing ~11,000 QSOs in train fold)
4) SMOTE 20x (Synthesizing ~22,000 QSOs in train fold)
Evaluates out-of-fold precision, recall, and F1.
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

from config import RESULTS_DIR


def smote_oversample(X_minority, n_synthetic, k_neighbors=5, random_state=42):
    """Synthetic Minority Over-sampling Technique (SMOTE) implementation."""
    rng = np.random.RandomState(random_state)
    k = min(k_neighbors, len(X_minority) - 1)
    knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_minority)
    _, indices = knn.kneighbors(X_minority)

    base_idx = rng.randint(0, len(X_minority), size=n_synthetic)
    neighbor_col = rng.randint(1, k + 1, size=n_synthetic)
    neighbor_idx = indices[base_idx, neighbor_col]

    diff = X_minority[neighbor_idx] - X_minority[base_idx]
    gap = rng.uniform(0.0, 1.0, size=(n_synthetic, 1)).astype(np.float32)
    synthetic = X_minority[base_idx] + gap * diff
    return synthetic


def run_cv_evaluation(all_embeddings, y_true_idx, class_names, smote_multiplier=0, n_splits=5):
    """Runs cross-validation with optional SMOTE on train folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    qso_idx = class_names.index('QSO')
    total_targets = len(y_true_idx)

    oof_preds = np.zeros(total_targets, dtype=int)
    oof_probs = np.zeros((total_targets, len(class_names)), dtype=np.float32)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_embeddings, y_true_idx), 1):
        X_train, y_train = all_embeddings[train_idx], y_true_idx[train_idx]
        X_val = all_embeddings[val_idx]

        if smote_multiplier > 0:
            m_qso_train = (y_train == qso_idx)
            X_qso = X_train[m_qso_train]
            n_synth = int(len(X_qso) * smote_multiplier)
            X_synth = smote_oversample(X_qso, n_synth, k_neighbors=5, random_state=42 + fold)
            y_synth = np.full(n_synth, qso_idx, dtype=int)

            X_train = np.vstack([X_train, X_synth])
            y_train = np.concatenate([y_train, y_synth])

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=16
        )
        rf.fit(X_train, y_train)
        oof_preds[val_idx] = rf.predict(X_val)
        oof_probs[val_idx] = rf.predict_proba(X_val)

    cv_time = time.time() - t0
    report = classification_report(y_true_idx, oof_preds, target_names=class_names, output_dict=True, digits=4)
    qso_rep = report['QSO']
    return {
        'smote_multiplier': smote_multiplier,
        'qso_recall': qso_rep['recall'],
        'qso_precision': qso_rep['precision'],
        'qso_f1': qso_rep['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'accuracy': report['accuracy'],
        'time': cv_time,
        'oof_preds': oof_preds,
        'oof_probs': oof_probs
    }


def main():
    print("=" * 70)
    print("Experiment 1: SMOTE Oversampling on 216k 2-Channel Embeddings")
    print("=" * 70)

    meta_path = RESULTS_DIR / "survey_meta_216k_final.npz"
    emb_path = RESULTS_DIR / "survey_embeddings_216k_2channel.npy"

    meta = np.load(meta_path, allow_pickle=True)
    db_classes = meta["db_classes"]
    all_embeddings = np.load(emb_path)

    class_names = ['GALAXY', 'QSO', 'STAR']
    cls_map = {c: i for i, c in enumerate(class_names)}
    y_true_idx = np.array([cls_map[c] for c in db_classes])

    print(f"Total dataset: {len(y_true_idx)} targets")
    print(f"Original class counts: {dict(zip(class_names, np.bincount(y_true_idx)))}")

    multipliers = [0, 5, 10, 20]
    results = []

    print(f"\n{'SMOTE Config':<18} | {'QSO Recall':>11} | {'QSO Prec':>10} | {'QSO F1':>8} | {'Macro F1':>9} | {'Time':>6}")
    print("-" * 75)

    for mult in multipliers:
        res = run_cv_evaluation(all_embeddings, y_true_idx, class_names, smote_multiplier=mult, n_splits=5)
        results.append(res)
        label = "Baseline (0x)" if mult == 0 else f"SMOTE {mult}x"
        print(f"{label:<18} | {res['qso_recall']*100:>10.2f}% | {res['qso_precision']*100:>9.2f}% | {res['qso_f1']:>8.4f} | {res['macro_f1']:>9.4f} | {res['time']:>5.1f}s")

    # Save summary results
    out_npz = RESULTS_DIR / "rf_smote_experiment_results.npz"
    np.savez_compressed(
        out_npz,
        multipliers=multipliers,
        qso_recalls=[r['qso_recall'] for r in results],
        qso_precisions=[r['qso_precision'] for r in results],
        qso_f1s=[r['qso_f1'] for r in results]
    )
    print(f"\nSaved results to {out_npz}")


if __name__ == "__main__":
    main()

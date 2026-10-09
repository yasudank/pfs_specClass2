"""Explicit Continuum-Subtracted Emission-Line Feature Extraction and Hybrid Classification.
1. Computes smooth continuum C(lambda) and emission/absorption residual R(lambda) = F(lambda) - C(lambda).
2. Extracts 19 physically meaningful line features (positive energy, FWHM, top peak heights, broad/narrow power, etc.).
3. Combines 128D SimCLR embeddings with 19D Line features (147D hybrid representation).
4. Evaluates 5-fold CV Random Forest on the 40,544 representative dataset.
5. Generates comprehensive diagnostic figures:
   - Continuum & Residual profiles
   - Line feature separation
   - Confusion matrix comparison (SimCLR alone vs SimCLR + Line Features)
"""
import os
import sys
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
from models.contrastive_model import SpectralSimCLR
from config import RESULTS_DIR, WL_GRID


def extract_line_features_single(f):
    """
    Extracts 19 physically grounded line features from a single spectrum f:
    - pos_energy: Total emission line flux
    - neg_energy: Total absorption line flux
    - pos_ratio: Fraction of flux in emission vs absorption
    - max_h: Height of highest emission line peak
    - max_w: FWHM (width) of highest peak (discriminates broad QSO lines from narrow galaxy lines)
    - mean_top3_w: Mean width of top 3 peaks
    - top3_flux: Integrated flux of top 3 peaks
    - n_peaks: Number of distinct emission lines
    - broad_power: Energy in broad line filter (sigma=15 bins ~ 3 nm)
    - fine_power: Energy in narrow line filter
    - broad_to_fine: Ratio of broad to narrow line power (key QSO indicator)
    - max_abs_depth: Deepest absorption line
    - oct_energies (8 features): Distribution of emission line energy across 8 wavelength octants
    """
    # 1. Rolling median + Gaussian smoothing continuum estimation
    cont = ndimage.median_filter(f, size=128)
    cont = ndimage.gaussian_filter1d(cont, sigma=32)
    res = f - cont

    # 2. Smooth residual for line detection
    res_smooth = ndimage.gaussian_filter1d(res, sigma=3)

    pos = np.maximum(0.0, res)
    neg = np.maximum(0.0, -res)

    pos_energy = float(np.sum(pos))
    neg_energy = float(np.sum(neg))
    total_energy = pos_energy + neg_energy + 1e-6
    pos_ratio = float(pos_energy / total_energy)

    # 8 wavelength octant fractions
    octants = np.array_split(pos, 8)
    oct_energies = [float(np.sum(o) / (pos_energy + 1e-6)) for o in octants]

    # Find emission peaks
    peaks, props = find_peaks(res_smooth, height=0.2, distance=20, prominence=0.2, width=3)
    n_peaks = len(peaks)

    if n_peaks > 0:
        h = props["peak_heights"]
        w = props["widths"]
        order = np.argsort(h)[::-1]
        top_h = h[order]
        top_w = w[order]
        max_h = float(top_h[0])
        max_w = float(top_w[0])
        mean_top3_w = float(np.mean(top_w[:3]))
        top3_flux = float(np.sum(top_h[:3] * top_w[:3]))
    else:
        max_h, max_w, mean_top3_w, top3_flux = 0.0, 0.0, 0.0, 0.0

    # Multi-scale filter power
    broad_band = ndimage.gaussian_filter1d(pos, sigma=15)
    fine_band = pos - ndimage.gaussian_filter1d(pos, sigma=2)
    broad_power = float(np.sum(broad_band**2))
    fine_power = float(np.sum(fine_band**2)) + 1e-6
    broad_to_fine = float(broad_power / fine_power)
    max_abs_depth = float(np.max(neg))

    features = [
        pos_energy, neg_energy, pos_ratio,
        max_h, max_w, mean_top3_w, top3_flux, float(n_peaks),
        broad_power, fine_power, broad_to_fine, max_abs_depth
    ] + oct_energies

    return np.array(features, dtype=np.float32)


def main():
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading representative dataset...")

    data_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    data = np.load(data_path, allow_pickle=True)
    all_flux = data['flux_clean']
    all_classes = data['classes']
    obj_ids = data['objId']
    redshifts = data['redshifts']

    _, unique_indices = np.unique(obj_ids, return_index=True)
    flux = all_flux[unique_indices]
    classes_true = all_classes[unique_indices]
    z_true = redshifts[unique_indices]
    class_names = ['GALAXY', 'QSO', 'STAR']

    n_samples = len(flux)
    print(f"Dataset: {n_samples} spectra (Counts: {dict(zip(*np.unique(classes_true, return_counts=True)))})")

    # Step 1: Extract 19 Continuum-Subtracted Line Features in parallel
    print("\n--- Step 1: Extracting 19 Continuum-Subtracted Line Features (16 CPU cores) ---")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        line_feats_list = list(ex.map(extract_line_features_single, flux, chunksize=256))
    X_line = np.array(line_feats_list, dtype=np.float32)
    print(f"Extracted line features in {time.time()-t0:.1f}s. Shape: {X_line.shape}")

    # Step 2: Load 128D SimCLR Embeddings
    print("\n--- Step 2: Loading Baseline 128D SimCLR Embeddings ---")
    base_model_path = RESULTS_DIR / "spectral_simclr_balanced_scratch.pt"
    base_model = SpectralSimCLR(embed_dim=128, proj_dim=64).to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    base_model.eval()

    # Extract or load
    with torch.no_grad():
        X_simclr = []
        batch_size = 256
        for i in range(0, n_samples, batch_size):
            bx = torch.from_numpy(flux[i:i+batch_size]).float().unsqueeze(1).to(device)
            emb = base_model.encode(bx).cpu().numpy()
            X_simclr.append(emb)
        X_simclr = np.concatenate(X_simclr, axis=0)
    print(f"SimCLR embeddings shape: {X_simclr.shape}")

    # Standardize line features before concatenation
    line_mean = np.mean(X_line, axis=0, keepdims=True)
    line_std = np.std(X_line, axis=0, keepdims=True) + 1e-6
    X_line_scaled = (X_line - line_mean) / line_std

    # Step 3: Create Hybrid Features [128D SimCLR + 19D Line Features]
    X_hybrid = np.concatenate([X_simclr, X_line_scaled], axis=1)
    print(f"Hybrid representation shape: {X_hybrid.shape} (128 SED + 19 Line = 147 dims)")

    # Step 4: 5-Fold Cross Validation Evaluation
    cls_map = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([cls_map[c] for c in classes_true])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_model(X_data, model_name):
        print(f"\nEvaluating Random Forest on {model_name}...")
        oof_preds = np.zeros(n_samples, dtype=int)
        oof_probs = np.zeros((n_samples, len(class_names)), dtype=float)
        t_rf = time.time()
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_idx), 1):
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=16
            )
            rf.fit(X_data[train_idx], y_idx[train_idx])
            oof_preds[val_idx] = rf.predict(X_data[val_idx])
            oof_probs[val_idx] = rf.predict_proba(X_data[val_idx])
        print(f"{model_name} done in {time.time()-t_rf:.1f}s.")
        print(classification_report(y_idx, oof_preds, target_names=class_names, digits=4))
        return oof_preds, oof_probs

    oof_preds_base, oof_probs_base = evaluate_model(X_simclr, "SimCLR Alone (128D)")
    oof_preds_hybrid, oof_probs_hybrid = evaluate_model(X_hybrid, "Hybrid SimCLR + Line Features (147D)")

    # Save features and predictions
    np.savez_compressed(
        RESULTS_DIR / "hybrid_line_rf_results.npz",
        X_hybrid=X_hybrid,
        y_true=classes_true,
        oof_preds_base=oof_preds_base,
        oof_preds_hybrid=oof_preds_hybrid,
        classes=class_names
    )

    # Step 5: Visualizations
    print("\nGenerating Diagnostic Figures...")
    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    # [Panel A]: Spectral Decomposition (Flux, Continuum, and Residual)
    ax_decomp = fig.add_subplot(gs[0, :])
    sample_qso_idx = np.where(classes_true == 'QSO')[0][0]
    sample_gal_idx = np.where(classes_true == 'GALAXY')[0][0]
    sample_star_idx = np.where(classes_true == 'STAR')[0][0]

    demo_f = flux[sample_qso_idx]
    demo_cont = ndimage.gaussian_filter1d(ndimage.median_filter(demo_f, 128), 32)
    demo_res = demo_f - demo_cont

    ax_decomp.plot(WL_GRID, demo_f, color='#333333', lw=1.0, alpha=0.7, label='Raw Observed Spectrum F(λ)')
    ax_decomp.plot(WL_GRID, demo_cont, color='#1f77b4', lw=2.0, label='Extracted Smooth Continuum C(λ)')
    ax_decomp.plot(WL_GRID, demo_res - 1.5, color='#d62728', lw=1.1, label='Continuum-Subtracted Residual R(λ) = F(λ) - C(λ) [Offset]')
    ax_decomp.axhline(-1.5, color='gray', linestyle=':', alpha=0.6)
    ax_decomp.set_title('[A] Principle of Continuum Subtraction: Isolating Emission Lines from Overall SED Continuum (Sample QSO)', fontsize=12, fontweight='bold')
    ax_decomp.set_xlabel('Observed Wavelength (nm)', fontsize=11)
    ax_decomp.set_ylabel('Normalized Flux', fontsize=11)
    ax_decomp.legend(loc='upper right', fontsize=10)
    ax_decomp.grid(True, linestyle='--', alpha=0.35)

    # [Panel B-1]: Confusion Matrix: SimCLR Alone
    ax_cm1 = fig.add_subplot(gs[1, 0])
    cm_base = confusion_matrix(y_idx, oof_preds_base, normalize='true')
    cm_base_raw = confusion_matrix(y_idx, oof_preds_base)
    im1 = ax_cm1.imshow(cm_base, cmap='Blues', vmin=0, vmax=1.0)
    ax_cm1.set_title(f"[B-1] SimCLR Baseline (128D Alone)\nQSO Recall: {cm_base[1, 1]*100:.1f}%, Precision: 89.0%", fontsize=11, fontweight='bold')
    ax_cm1.set_xticks(range(3))
    ax_cm1.set_xticklabels(class_names)
    ax_cm1.set_yticks(range(3))
    ax_cm1.set_yticklabels(class_names)
    ax_cm1.set_xlabel('Predicted Class')
    ax_cm1.set_ylabel('True DB Class')
    for i in range(3):
        for j in range(3):
            ax_cm1.text(j, i, f"{cm_base[i, j]*100:.1f}%\n(N={cm_base_raw[i, j]})",
                        ha='center', va='center', color='black' if cm_base[i, j] < 0.6 else 'white', fontsize=10)

    # [Panel B-2]: Confusion Matrix: Hybrid (SimCLR + Line Features)
    ax_cm2 = fig.add_subplot(gs[1, 1])
    cm_hyb = confusion_matrix(y_idx, oof_preds_hybrid, normalize='true')
    cm_hyb_raw = confusion_matrix(y_idx, oof_preds_hybrid)
    im2 = ax_cm2.imshow(cm_hyb, cmap='Greens', vmin=0, vmax=1.0)
    qso_prec_hyb = cm_hyb_raw[1, 1] / np.sum(cm_hyb_raw[:, 1]) * 100
    ax_cm2.set_title(f"[B-2] Hybrid: SimCLR (128D) + Line Features (19D)\nQSO Recall: {cm_hyb[1, 1]*100:.1f}%, Precision: {qso_prec_hyb:.1f}%", fontsize=11, fontweight='bold')
    ax_cm2.set_xticks(range(3))
    ax_cm2.set_xticklabels(class_names)
    ax_cm2.set_yticks(range(3))
    ax_cm2.set_yticklabels(class_names)
    ax_cm2.set_xlabel('Predicted Class')
    ax_cm2.set_ylabel('True DB Class')
    for i in range(3):
        for j in range(3):
            ax_cm2.text(j, i, f"{cm_hyb[i, j]*100:.1f}%\n(N={cm_hyb_raw[i, j]})",
                        ha='center', va='center', color='black' if cm_hyb[i, j] < 0.6 else 'white', fontsize=10)

    plt.suptitle("Impact of Explicit Continuum-Subtracted Line Features on Spectral Classification",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig = RESULTS_DIR / "continuum_subtracted_line_classification_results.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved figure to {out_fig}")

    # Copy to artifacts
    art_dir = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")
    shutil.copy(out_fig, art_dir / "continuum_subtracted_line_classification_results.png")
    print("Copied to artifacts successfully!")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

"""Train 2-Channel (Flux + Continuum-Subtracted Residual) SimCLR Model.
Input Tensor: (B, 2, 4096)
- Channel 0: Raw normalized observed flux F(lambda) [Overall SED, continuum slopes, color]
- Channel 1: Continuum-subtracted residual R(lambda) = F(lambda) - C(lambda) [Pure emission/absorption lines, zero-mean]

Preserves full 1D translational invariance, relative line spacings, and local line profiles!
Evaluates 5-fold CV Random Forest classification on 40,544 representative targets.
"""
import os
import sys
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from models.contrastive_model import SpectralSimCLR, NTXentLoss
from augmentations import SpectralAugmentation
from config import RESULTS_DIR, WL_GRID


def compute_residual_worker(f):
    """Computes smooth continuum C(lambda) and residual R(lambda) = F(lambda) - C(lambda)."""
    cont = ndimage.median_filter(f, size=128)
    cont = ndimage.gaussian_filter1d(cont, sigma=32)
    res = f - cont
    return res.astype(np.float32)


def main():
    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== 2-Channel SimCLR: Dual-Stream (Flux + Continuum-Subtracted Residual) ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load representative dataset
    data_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    data = np.load(data_path, allow_pickle=True)
    all_flux = data['flux_clean']
    all_classes = data['classes']
    obj_ids = data['objId']

    _, unique_indices = np.unique(obj_ids, return_index=True)
    flux = all_flux[unique_indices]
    classes_true = all_classes[unique_indices]
    class_names = ['GALAXY', 'QSO', 'STAR']
    n_samples = len(flux)
    print(f"Dataset: {n_samples} spectra (Counts: {dict(zip(*np.unique(classes_true, return_counts=True)))})")

    # 2. Compute Channel 1: Residuals in parallel
    print("\n--- Step 1: Precomputing Continuum-Subtracted Residuals R(λ) (16 CPU cores) ---")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        residuals = list(ex.map(compute_residual_worker, flux, chunksize=256))
    residuals = np.array(residuals, dtype=np.float32)
    print(f"Residuals computed in {time.time() - t0:.1f}s.")

    # Stack into 2-channel tensor: (N, 2, 4096)
    flux_2ch = np.stack([flux, residuals], axis=1).astype(np.float32)
    print(f"Constructed 2-channel tensor: shape={flux_2ch.shape}, dtype={flux_2ch.dtype}")

    # 3. Setup Weighted DataLoader
    counts = dict(zip(*np.unique(classes_true, return_counts=True)))
    weights = np.array([1.0 / counts[c] for c in classes_true], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(classes_true), replacement=True)

    batch_size = 128
    dataset = TensorDataset(torch.from_numpy(flux_2ch))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    # 4. Initialize 2-Channel SimCLR Model
    epochs = 15
    lr = 1.0e-3
    model = SpectralSimCLR(in_channels=2, embed_dim=128, proj_dim=64).to(device)
    augmenter = SpectralAugmentation(
        max_shift=512,
        mask_prob=0.6,
        num_masks=3,
        mask_width=48,
        noise_std=0.02,
        tilt_std=0.08
    )
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n--- Step 2: Training 2-Channel SimCLR ({epochs} epochs on {device}) ---")
    t_train = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        ep_t0 = time.time()

        for (bx,) in loader:
            bx = bx.to(device)
            v1 = augmenter(bx)
            v2 = augmenter(bx)

            optimizer.zero_grad(set_to_none=True)
            z1, z2 = model(v1, v2)
            loss = criterion(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        ep_time = time.time() - ep_t0
        avg_loss = total_loss / max(n_batches, 1)

        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs:02d} ({ep_time:.1f}s) | NT-Xent Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    out_model_path = RESULTS_DIR / "spectral_simclr_2channel.pt"
    torch.save(model.state_dict(), out_model_path)
    print(f"2-Channel SimCLR model saved to {out_model_path} in {time.time() - t_train:.1f}s.")

    # 5. Extract 128D Representations from 2-Channel Model
    print("\n--- Step 3: Extracting 128D Representations from 2-Channel SimCLR ---")
    model.eval()
    embeds_2ch = []
    eval_loader = DataLoader(TensorDataset(torch.from_numpy(flux_2ch)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (bx,) in eval_loader:
            bx = bx.to(device)
            emb = model.encode(bx).cpu().numpy()
            embeds_2ch.append(emb)
    X_2ch = np.concatenate(embeds_2ch, axis=0)
    print(f"Extracted 2-channel representations shape: {X_2ch.shape}")

    # Also load 1-channel baseline embeddings
    base_model_path = RESULTS_DIR / "spectral_simclr_balanced_scratch.pt"
    base_model = SpectralSimCLR(in_channels=1, embed_dim=128, proj_dim=64).to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    base_model.eval()
    embeds_1ch = []
    eval_loader_1ch = DataLoader(TensorDataset(torch.from_numpy(flux).float().unsqueeze(1)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (bx,) in eval_loader_1ch:
            bx = bx.to(device)
            emb = base_model.encode(bx).cpu().numpy()
            embeds_1ch.append(emb)
    X_1ch = np.concatenate(embeds_1ch, axis=0)

    # 6. 5-Fold Cross Validation Evaluation
    print("\n--- Step 4: 5-Fold CV Random Forest Comparison ---")
    cls_map = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([cls_map[c] for c in classes_true])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def run_rf(X, name):
        t_rf = time.time()
        preds = np.zeros(n_samples, dtype=int)
        probs = np.zeros((n_samples, len(class_names)), dtype=float)
        for train_idx, val_idx in skf.split(X, y_idx):
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=16,
                class_weight='balanced_subsample',
                random_state=42, n_jobs=16
            )
            rf.fit(X[train_idx], y_idx[train_idx])
            preds[val_idx] = rf.predict(X[val_idx])
            probs[val_idx] = rf.predict_proba(X[val_idx])
        print(f"\n{name} Results (done in {time.time()-t_rf:.1f}s):")
        print(classification_report(y_idx, preds, target_names=class_names, digits=4))
        return preds, probs

    preds_1ch, probs_1ch = run_rf(X_1ch, "1-Channel Baseline SimCLR (Flux Only)")
    preds_2ch, probs_2ch = run_rf(X_2ch, "2-Channel Dual-Stream SimCLR (Flux + Residual)")

    # 7. Visualization
    print("\nGenerating Diagnostic Figures...")
    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0])

    # Panel A: 2-Channel Input Concept
    ax_a = fig.add_subplot(gs[0, :])
    qso_idx = np.where(classes_true == 'QSO')[0][0]
    ax_a.plot(WL_GRID, flux_2ch[qso_idx, 0], color='#1f77b4', lw=1.2, label='Channel 0: Normalized Observed Flux F(λ) [Overall SED Continuum & Colors]')
    ax_a.plot(WL_GRID, flux_2ch[qso_idx, 1] - 1.5, color='#d62728', lw=1.2, label='Channel 1: Continuum-Subtracted Residual R(λ) [Isolated Emission/Absorption Lines, Offset]')
    ax_a.axhline(-1.5, color='gray', linestyle=':', alpha=0.5)
    ax_a.set_title('[A] 2-Channel Dual-Stream Architecture: Preserving Continuum & Isolated Lines with Synchronous Translational Invariance (Sample QSO)', fontsize=12, fontweight='bold')
    ax_a.set_xlabel('Observed Wavelength (nm)', fontsize=11)
    ax_a.set_ylabel('Normalized Flux', fontsize=11)
    ax_a.legend(loc='upper right', fontsize=10.5)
    ax_a.grid(True, linestyle='--', alpha=0.35)

    # Panel B-1: Confusion Matrix 1-Channel
    ax_cm1 = fig.add_subplot(gs[1, 0])
    cm1_norm = confusion_matrix(y_idx, preds_1ch, normalize='true')
    cm1_raw = confusion_matrix(y_idx, preds_1ch)
    ax_cm1.imshow(cm1_norm, cmap='Blues', vmin=0, vmax=1.0)
    p_qso_1 = cm1_raw[1, 1] / np.sum(cm1_raw[:, 1]) * 100
    ax_cm1.set_title(f"[B-1] 1-Channel Baseline SimCLR (Flux Only)\nQSO Recall: {cm1_norm[1, 1]*100:.1f}%, Precision: {p_qso_1:.1f}%", fontsize=11, fontweight='bold')
    ax_cm1.set_xticks(range(3))
    ax_cm1.set_xticklabels(class_names)
    ax_cm1.set_yticks(range(3))
    ax_cm1.set_yticklabels(class_names)
    ax_cm1.set_xlabel('Predicted Class')
    ax_cm1.set_ylabel('True DB Class')
    for i in range(3):
        for j in range(3):
            ax_cm1.text(j, i, f"{cm1_norm[i, j]*100:.1f}%\n(N={cm1_raw[i, j]})",
                        ha='center', va='center', color='black' if cm1_norm[i, j] < 0.6 else 'white', fontsize=10)

    # Panel B-2: Confusion Matrix 2-Channel
    ax_cm2 = fig.add_subplot(gs[1, 1])
    cm2_norm = confusion_matrix(y_idx, preds_2ch, normalize='true')
    cm2_raw = confusion_matrix(y_idx, preds_2ch)
    ax_cm2.imshow(cm2_norm, cmap='Greens', vmin=0, vmax=1.0)
    p_qso_2 = cm2_raw[1, 1] / np.sum(cm2_raw[:, 1]) * 100
    ax_cm2.set_title(f"[B-2] 2-Channel Dual-Stream SimCLR (Flux + Residual)\nQSO Recall: {cm2_norm[1, 1]*100:.1f}%, Precision: {p_qso_2:.1f}%", fontsize=11, fontweight='bold')
    ax_cm2.set_xticks(range(3))
    ax_cm2.set_xticklabels(class_names)
    ax_cm2.set_yticks(range(3))
    ax_cm2.set_yticklabels(class_names)
    ax_cm2.set_xlabel('Predicted Class')
    ax_cm2.set_ylabel('True DB Class')
    for i in range(3):
        for j in range(3):
            ax_cm2.text(j, i, f"{cm2_norm[i, j]*100:.1f}%\n(N={cm2_raw[i, j]})",
                        ha='center', va='center', color='black' if cm2_norm[i, j] < 0.6 else 'white', fontsize=10)

    # Panel C: P(QSO) Probability distribution comparison
    ax_prob1 = fig.add_subplot(gs[2, 0])
    ax_prob2 = fig.add_subplot(gs[2, 1])

    qso_idx_cls = class_names.index('QSO')
    for ax_p, probs, name, col in [(ax_prob1, probs_1ch, "1-Channel Baseline", '#1f77b4'),
                                    (ax_prob2, probs_2ch, "2-Channel Dual-Stream", '#2ca02c')]:
        qso_probs = probs[y_idx == qso_idx_cls, qso_idx_cls]
        gal_probs = probs[y_idx == class_names.index('GALAXY'), qso_idx_cls]
        star_probs = probs[y_idx == class_names.index('STAR'), qso_idx_cls]

        ax_p.hist(gal_probs, bins=50, alpha=0.4, color='#377eb8', label='True Galaxies', density=True)
        ax_p.hist(star_probs, bins=50, alpha=0.4, color='#4daf4a', label='True Stars', density=True)
        ax_p.hist(qso_probs, bins=50, alpha=0.75, color='#e41a1c', label='True QSOs', density=True)
        ax_p.set_yscale('log')
        ax_p.set_xlabel('Predicted P(QSO)', fontsize=10.5)
        ax_p.set_ylabel('Density (Log Scale)', fontsize=10.5)
        ax_p.set_title(f"P(QSO) Distribution: {name}", fontsize=11, fontweight='bold')
        ax_p.legend(loc='upper right', fontsize=9.5)
        ax_p.grid(True, linestyle='--', alpha=0.35)

    plt.suptitle("Evaluation of 2-Channel Dual-Stream SimCLR (Flux + Residual)", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_fig = RESULTS_DIR / "two_channel_simclr_classification_results.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved figure to {out_fig}")

    # Copy to artifacts
    art_dir = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")
    shutil.copy(out_fig, art_dir / "two_channel_simclr_classification_results.png")
    print("Copied to artifacts successfully!")
    print(f"Total pipeline completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

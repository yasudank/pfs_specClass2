"""Train 2-Channel SimCLR Model with Channel 1 (Residual) Amplitude Boost (alpha = 3.0).
Input Tensor: (B, 2, 4096)
- Channel 0: Raw normalized observed flux F(lambda) [Overall SED, continuum shape]
- Channel 1: 3.0 * Continuum-subtracted residual 3.0 * R(lambda) [Amplified emission/absorption lines]

Trains on RTX 4090 for 15 epochs, saves model checkpoint, and evaluates 5-fold CV classification.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

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
    alpha_boost = 3.0
    print("=" * 70)
    print(f"Training 2-Channel SimCLR with Channel 1 Boost alpha = {alpha_boost:.1f}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

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

    # 2. Compute Channel 1: Residuals
    print("\n--- Step 1: Precomputing Residuals & Applying alpha=3.0 Boost ---")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        residuals = list(ex.map(compute_residual_worker, flux, chunksize=256))
    residuals = np.array(residuals, dtype=np.float32)
    print(f"Residuals computed in {time.time() - t0:.1f}s.")

    # Stack into 2-channel tensor with 3x boosted residuals: (N, 2, 4096)
    boosted_residuals = alpha_boost * residuals
    flux_2ch = np.stack([flux, boosted_residuals], axis=1).astype(np.float32)
    print(f"Constructed boosted 2-channel tensor: shape={flux_2ch.shape}")
    print(f"Channel 0 std: {np.std(flux):.4f} | Channel 1 (boosted 3x) std: {np.std(boosted_residuals):.4f}")

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

    print(f"\n--- Step 2: Training Boosted 2-Channel SimCLR ({epochs} epochs on {device}) ---")
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

    out_model_path = RESULTS_DIR / "spectral_simclr_2channel_boost3x.pt"
    torch.save(model.state_dict(), out_model_path)
    print(f"\nModel saved to {out_model_path} in {time.time() - t_train:.1f}s.")

    # 5. Extract 128D Representations
    print("\n--- Step 3: Extracting 128D Representations ---")
    model.eval()
    embeds = []
    eval_loader = DataLoader(TensorDataset(torch.from_numpy(flux_2ch)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (bx,) in eval_loader:
            bx = bx.to(device)
            emb = model.encode(bx).cpu().numpy()
            embeds.append(emb)
    embeds = np.vstack(embeds)
    print(f"Extracted embeddings shape: {embeds.shape}")

    # 6. Evaluate 5-fold CV Random Forest on representative dataset
    print("\n--- Step 4: 5-Fold CV Random Forest Classification ---")
    cls_map = {c: i for i, c in enumerate(class_names)}
    y_true_idx = np.array([cls_map[c] for c in classes_true])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_true_idx), dtype=int)
    oof_probs = np.zeros((len(y_true_idx), len(class_names)), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeds, y_true_idx), 1):
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=16
        )
        rf.fit(embeds[train_idx], y_true_idx[train_idx])
        oof_preds[val_idx] = rf.predict(embeds[val_idx])
        oof_probs[val_idx] = rf.predict_proba(embeds[val_idx])

    print("\n================== BOOSTED 2-CHANNEL EVALUATION REPORT ==================")
    report_str = classification_report(y_true_idx, oof_preds, target_names=class_names, digits=4)
    print(report_str)

    # Save representative embeddings and results
    np.save(RESULTS_DIR / "embeddings_representative_boost3x.npy", embeds)
    np.savez_compressed(
        RESULTS_DIR / "rf_oof_boost3x_representative.npz",
        oof_preds=oof_preds,
        oof_probs=oof_probs,
        classes=class_names,
        y_true_idx=y_true_idx
    )
    print(f"Completed in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

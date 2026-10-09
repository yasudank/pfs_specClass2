"""Compare Classification Performance: SimCLR with Attention vs SimCLR without Attention.
Evaluates on the 40,544 representative dataset (1,309 QSOs, 19,721 Galaxies, 19,514 Stars).
1. Extracts 128D embeddings using Attentive SimCLR.
2. Extracts 128D embeddings using original SimCLR (without Attention).
3. Evaluates 5-fold CV Random Forest on both.
4. Generates side-by-side confusion matrix & P(QSO) recall comparison plot.
"""
import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
from torch.utils.data import DataLoader, TensorDataset
from models.contrastive_model import SpectralSimCLR, SpectralAttentionSimCLR
from config import RESULTS_DIR


def extract_features(model, flux_array, device, batch_size=256):
    loader = DataLoader(TensorDataset(torch.from_numpy(flux_array).float().unsqueeze(1)),
                        batch_size=batch_size, shuffle=False)
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            emb = model.encode(bx)
            all_embeds.append(emb.cpu().numpy())
    return np.concatenate(all_embeds, axis=0)


def run_rf_cv(X, y_labels, classes):
    cls_map = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_map[c] for c in y_labels])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y_labels), dtype=int)
    oof_probs = np.zeros((len(y_labels), len(classes)), dtype=float)

    for train_idx, val_idx in skf.split(X, y_idx):
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=16
        )
        rf.fit(X[train_idx], y_idx[train_idx])
        oof_preds[val_idx] = rf.predict(X[val_idx])
        oof_probs[val_idx] = rf.predict_proba(X[val_idx])

    return oof_preds, oof_probs, y_idx


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading data...")

    data_path = RESULTS_DIR / "dataset_full_augmented_scratch.npz"
    data = np.load(data_path, allow_pickle=True)
    all_flux = data['flux_clean']
    all_classes = data['classes']
    obj_ids = data['objId']

    _, unique_indices = np.unique(obj_ids, return_index=True)
    flux = all_flux[unique_indices]
    classes_true = all_classes[unique_indices]
    class_names = ['GALAXY', 'QSO', 'STAR']

    print(f"Dataset size: {len(flux)} targets")

    # 1. Feature extraction with Baseline SimCLR (No Attention)
    base_model_path = RESULTS_DIR / "spectral_simclr_balanced_scratch.pt"
    print(f"\n1. Extracting Baseline SimCLR embeddings (No Attention)...")
    base_model = SpectralSimCLR(embed_dim=128, proj_dim=64).to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    X_base = extract_features(base_model, flux, device)

    # 2. Feature extraction with Attentive SimCLR (With Attention)
    attn_model_path = RESULTS_DIR / "spectral_simclr_attention_model.pt"
    print(f"\n2. Extracting Attentive SimCLR embeddings (With Attention)...")
    attn_model = SpectralAttentionSimCLR(embed_dim=128, proj_dim=64).to(device)
    attn_model.load_state_dict(torch.load(attn_model_path, map_location=device))
    X_attn = extract_features(attn_model, flux, device)

    # 3. RF on Baseline
    print("\n3. Training RF on Baseline SimCLR (5-fold CV)...")
    t0 = time.time()
    preds_base, probs_base, y_idx = run_rf_cv(X_base, classes_true, class_names)
    print(f"Baseline RF done in {time.time()-t0:.1f}s")
    print(classification_report(y_idx, preds_base, target_names=class_names, digits=4))

    # 4. RF on Attentive
    print("\n4. Training RF on Attentive SimCLR (5-fold CV)...")
    t0 = time.time()
    preds_attn, probs_attn, _ = run_rf_cv(X_attn, classes_true, class_names)
    print(f"Attentive RF done in {time.time()-t0:.1f}s")
    print(classification_report(y_idx, preds_attn, target_names=class_names, digits=4))

    # 5. Side-by-side visualization
    cm_base = confusion_matrix(y_idx, preds_base, normalize='true')
    cm_attn = confusion_matrix(y_idx, preds_attn, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Confusion matrix comparison
    im1 = axes[0].imshow(cm_base, cmap='Blues', vmin=0, vmax=1.0)
    axes[0].set_title(f"Baseline SimCLR (Without Attention / GAP)\nQSO Recall: {cm_base[1, 1]*100:.1f}%", fontsize=11, fontweight='bold')
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(class_names)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel('Predicted Class')
    axes[0].set_ylabel('True DB Class')
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{cm_base[i, j]*100:.1f}%\n(N={confusion_matrix(y_idx, preds_base)[i, j]})",
                         ha='center', va='center', color='black' if cm_base[i, j] < 0.6 else 'white', fontsize=9.5)

    im2 = axes[1].imshow(cm_attn, cmap='Greens', vmin=0, vmax=1.0)
    axes[1].set_title(f"Attentive SimCLR (With Self-Attention + Attentive Pooling)\nQSO Recall: {cm_attn[1, 1]*100:.1f}%", fontsize=11, fontweight='bold')
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(class_names)
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('True DB Class')
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{cm_attn[i, j]*100:.1f}%\n(N={confusion_matrix(y_idx, preds_attn)[i, j]})",
                         ha='center', va='center', color='black' if cm_attn[i, j] < 0.6 else 'white', fontsize=9.5)

    plt.suptitle("Impact of Attention Mechanism on Astronomical Object Classification", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_fig = RESULTS_DIR / "attention_vs_baseline_comparison.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved comparison figure to {out_fig}")

    # Copy to artifacts
    art_dir = Path("/home/yasuda/.gemini/antigravity-ide/brain/d5096d3e-909d-4738-8a2b-70fbd53adf8d")
    shutil.copy(out_fig, art_dir / "attention_vs_baseline_comparison.png")
    print("Copied to artifacts successfully!")


if __name__ == "__main__":
    main()

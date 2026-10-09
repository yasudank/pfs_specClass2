"""Training and Evaluation Script for Step A: Self-Supervised Contrastive Learning (SpectralSimCLR)."""
import os
import sys
import time
from pathlib import Path
import numpy as np

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

from config import RESULTS_DIR, N_GRID
from augmentations import SpectralAugmentation
from models.contrastive_model import SpectralSimCLR, NTXentLoss
from evaluate import plot_latent_space


def train_simclr(
    data: dict,
    epochs: int = 40,
    batch_size: int = 64,
    embed_dim: int = 128,
    proj_dim: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.1,
    max_shift: int = 384  # Shift in log-wavelength grid
):
    """Trains SpectralSimCLR using random shift and cutout augmentations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training SpectralSimCLR on {device} ---")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}, embed_dim={embed_dim}, max_shift={max_shift}")

    X = torch.from_numpy(data['flux'][:, None, :])  # (N, 1, 4096)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SpectralSimCLR(embed_dim=embed_dim, proj_dim=proj_dim).to(device)
    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    augmenter = SpectralAugmentation(
        max_shift=max_shift,
        mask_prob=0.6,
        num_masks=3,
        mask_width=48,
        noise_std=0.02,
        tilt_std=0.08
    )

    history = {'loss': []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (bx,) in loader:
            bx = bx.to(device)
            # Create two augmented views
            v1 = augmenter(bx)
            v2 = augmenter(bx)

            optimizer.zero_grad()
            z1, z2 = model(v1, v2)
            loss = criterion(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | NT-Xent Loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

    print(f"Training completed in {time.time() - t0:.1f}s.")

    # Extract embeddings for all data without augmentation
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeds = []
    with torch.no_grad():
        for (bx,) in eval_loader:
            bx = bx.to(device)
            emb = model.encode(bx)
            all_embeds.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeds, axis=0)

    # Save model checkpoint
    ckpt_path = RESULTS_DIR / "simclr_spectral.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'embed_dim': embed_dim,
        'history': history
    }, ckpt_path)
    print(f"Model saved to {ckpt_path}.")

    return model, embeddings, history


def evaluate_contrastive_representations(
    simclr_embeds: np.ndarray,
    ae_latents: np.ndarray,
    classes: np.ndarray,
    redshifts: np.ndarray
):
    """
    Evaluates representation quality:
    1. 2D projection comparison (Visual inspection of class separation & redshift invariance)
    2. k-NN Classification benchmark (Self-supervised feature quality)
    """
    print("\n--- Evaluating Representation Quality (k-NN Benchmark) ---")

    # Split train/test indices (70/30) for k-NN probe
    np.random.seed(42)
    n_samples = len(classes)
    perm = np.random.permutation(n_samples)
    n_train = int(0.7 * n_samples)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    y_train, y_test = classes[train_idx], classes[test_idx]

    # Benchmark 1: Phase 1 Autoencoder features
    knn_ae = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_ae.fit(ae_latents[train_idx], y_train)
    pred_ae = knn_ae.predict(ae_latents[test_idx])
    acc_ae = accuracy_score(y_test, pred_ae)
    print(f"[Phase 1 1D-Autoencoder] 5-NN Test Accuracy: {acc_ae * 100:.2f}%")

    # Benchmark 2: Phase 2 SimCLR contrastive features
    knn_simclr = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn_simclr.fit(simclr_embeds[train_idx], y_train)
    pred_simclr = knn_simclr.predict(simclr_embeds[test_idx])
    acc_simclr = accuracy_score(y_test, pred_simclr)
    print(f"[Phase 2 SimCLR (Step A)] 5-NN Test Accuracy: {acc_simclr * 100:.2f}%")

    print("\nSimCLR Classification Report:")
    print(classification_report(y_test, pred_simclr, digits=3))

    # Visualizations
    plot_latent_space(
        simclr_embeds,
        classes,
        redshifts,
        method_name="SimCLR (Shift-Invariant, dim=128)",
        save_prefix="simclr"
    )

    # Plot Comparison Chart
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ['1D-Autoencoder\n(Phase 1)', 'SpectralSimCLR\n(Step A: Shift-Invariant)']
    accs = [acc_ae * 100, acc_simclr * 100]
    bars = ax.bar(methods, accs, color=['#4a90e2', '#50e3c2'], width=0.5)
    ax.set_ylabel("5-NN Unsupervised Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Unsupervised Representation Quality Comparison")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    comp_path = RESULTS_DIR / "model_comparison_knn.png"
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to {comp_path}")

    return {
        'acc_ae': acc_ae,
        'acc_simclr': acc_simclr,
        'report_simclr': classification_report(y_test, pred_simclr, output_dict=True)
    }


def main():
    print("=" * 60)
    print("  PFS Spectra Unsupervised Learning - Step A: SimCLR SSL  ")
    print("=" * 60)

    # 1. Load dataset cache from Phase 1
    cache_path = RESULTS_DIR / "dataset_cache_400.npz"
    if not os.path.exists(cache_path):
        print(f"Cache {cache_path} not found. Please run Phase 1 first.")
        return

    data = np.load(cache_path, allow_pickle=True)
    print(f"Loaded {len(data['flux'])} cached spectra from {cache_path}.")

    # Load Phase 1 Autoencoder latents for comparison
    ae_ckpt = RESULTS_DIR / "autoencoder_baseline.pt"
    from models.autoencoder import SpectrumAutoencoder1D
    ae_model = SpectrumAutoencoder1D(latent_dim=32, seq_len=N_GRID)
    ckpt = torch.load(ae_ckpt, map_location='cpu')
    ae_model.load_state_dict(ckpt['model_state_dict'])
    ae_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model.to(device)

    with torch.no_grad():
        X_tensor = torch.from_numpy(data['flux'][:, None, :]).to(device)
        ae_latents = ae_model.encode(X_tensor).cpu().numpy()

    # 2. Train SpectralSimCLR
    model, simclr_embeds, history = train_simclr(
        data=data,
        epochs=40,
        batch_size=64,
        embed_dim=128,
        proj_dim=64,
        lr=1e-3,
        temperature=0.1,
        max_shift=384
    )

    # 3. Comprehensive Evaluation & Comparison
    results = evaluate_contrastive_representations(
        simclr_embeds=simclr_embeds,
        ae_latents=ae_latents,
        classes=data['classes'],
        redshifts=data['redshifts']
    )

    print("\n" * 1 + "=" * 60)
    print("Step A (Contrastive Learning) finished successfully!")
    print(f"Results and plots saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

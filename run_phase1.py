import os
import sys

if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = sys.prefix

import numpy as np

from train_baseline import prepare_dataset, train_empca, train_autoencoder
from evaluate import plot_latent_space, plot_anomaly_distribution_and_samples
from config import RESULTS_DIR


def main():
    print("=" * 60)
    print("       PFS Spectra Unsupervised Learning - Phase 1       ")
    print("=" * 60)

    # 1. Prepare Dataset (Balanced sampling across STAR, GALAXY, QSO)
    # n_per_class=400 gives 1,200 spectra for fast and thorough baseline analysis
    data = prepare_dataset(n_per_class=400, cache_file=str(RESULTS_DIR / "dataset_cache_400.npz"))

    print(f"\nDataset summary:")
    print(f"  Total spectra: {len(data['flux'])}")
    for cls in ['STAR', 'GALAXY', 'QSO']:
        count = np.sum(data['classes'] == cls)
        print(f"    {cls}: {count}")

    # 2. Run EMPCA baseline
    empca, empca_coeffs, empca_scores = train_empca(data, n_components=10)
    plot_latent_space(
        empca_coeffs,
        data['classes'],
        data['redshifts'],
        method_name="EMPCA (10 components)",
        save_prefix="empca"
    )

    # 3. Train 1D Convolutional Autoencoder baseline
    model, ae_latents, ae_scores, ae_recons, history = train_autoencoder(
        data, epochs=30, batch_size=64, latent_dim=32, lr=1e-3
    )

    # 4. Evaluate Autoencoder latent space
    plot_latent_space(
        ae_latents,
        data['classes'],
        data['redshifts'],
        method_name="1D-Autoencoder (dim=32)",
        save_prefix="autoencoder"
    )

    # 5. Evaluate Anomaly Scores & Outlier Spectra
    plot_anomaly_distribution_and_samples(
        flux=data['flux'],
        recons=ae_recons,
        mask=data['mask'],
        scores=ae_scores,
        obj_ids=data['objId'],
        classes=data['classes'],
        redshifts=data['redshifts'],
        method_name="1D-Autoencoder",
        save_prefix="autoencoder"
    )

    print("\n" + "=" * 60)
    print("Phase 1 Baseline execution finished successfully!")
    print(f"Results and figures saved in: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/bin/bash
# ==============================================================================
# Subaru PFS Spectral Classification Pipeline
# Architecture: 2-Channel SimCLR (Channel 1 Boost alpha=3.0) + SMOTE 3x Random Forest
# ==============================================================================

set -e

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"

# Locate dedicated virtual environment: pfs_specclass
if [ -f "/home/yasuda/work/lsst_stack/conda/envs/pfs_specclass/bin/python" ]; then
    PYTHON_EXEC="/home/yasuda/work/lsst_stack/conda/envs/pfs_specclass/bin/python"
elif [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" = "pfs_specclass" ]; then
    PYTHON_EXEC="$(which python)"
elif command -v conda &> /dev/null && conda env list | grep -q "pfs_specclass"; then
    PYTHON_EXEC="$(conda env list | grep 'pfs_specclass' | awk '{print $NF}')/bin/python"
else
    echo "Warning: Dedicated environment 'pfs_specclass' not detected. Running ./setup_env.sh..."
    bash ./setup_env.sh
    PYTHON_EXEC="/home/yasuda/work/lsst_stack/conda/envs/pfs_specclass/bin/python"
fi

echo "========================================================================"
echo "Subaru PFS Spectral Classification Full Pipeline"
echo "Environment: pfs_specclass"
echo "Python: $PYTHON_EXEC"
echo "Work Directory: $WORK_DIR"
echo "========================================================================"

# Step 1: Data Preparation from raw FITS (skip if dataset already exists)
if [ ! -f "results/dataset_full_augmented_scratch.npz" ]; then
    echo ""
    echo ">>> [Step 1/4] Preparing Survey Representative & Augmented Dataset..."
    "$PYTHON_EXEC" step1_prepare_full_augmented_dataset.py
else
    echo ""
    echo ">>> [Step 1/4] Dataset results/dataset_full_augmented_scratch.npz exists. Skipping generation."
fi

# Step 2: Train 2-Channel Boosted SimCLR (alpha=3.0) & Extract 128D Embeddings
echo ""
echo ">>> [Step 2/6] Training Boosted 2-Channel SimCLR (alpha=3.0) & Extracting 128D Embeddings..."
"$PYTHON_EXEC" train_2channel_boosted_simclr.py

# Step 3: Stage 1 Classification: SMOTE 3x Random Forest 5-Fold Cross-Validation
echo ""
echo ">>> [Step 3/6] Running Stage 1 Classification (SMOTE 3x + Random Forest 5-Fold CV)..."
"$PYTHON_EXEC" run_experiment3_full_analysis.py

# Step 4: Stage 2 Redshift Estimation: Train Galaxy ResNet & QSO Hybrid CCF Models
echo ""
echo ">>> [Step 4/6] Training Stage 2 Redshift Models (Galaxy ResNet & QSO Hybrid CCF) & Inferring Redshifts..."
"$PYTHON_EXEC" step4_train_and_estimate_redshifts.py

# Step 5: Generate Final Full Survey Predictions Catalog (216,476 targets)
echo ""
echo ">>> [Step 5/6] Generating Final Full Survey Predictions Catalog (CSV, Gzip, SQLite3)..."
"$PYTHON_EXEC" create_216k_full_survey_predictions_catalog.py

# Step 6: Run Latent Space Diagnostics & Spectral Visualizations
echo ""
echo ">>> [Step 6/6] Generating Latent Space and Spectral Diagnostics..."
echo "  - Running 128D Cosine Similarity & Remaining Missed QSOs Analysis..."
"$PYTHON_EXEC" visualize_remaining_missed_qsos_latent.py

echo "  - Plotting Individual Representative Spectra of Missed QSOs..."
"$PYTHON_EXEC" plot_remaining_missed_qso_spectra.py

echo "  - Mapping Individual Spectra onto Latent Space & PCA/t-SNE Manifolds..."
"$PYTHON_EXEC" plot_individual_missed_qsos_latent_positions.py

echo ""
echo "========================================================================"
echo "Pipeline execution finished successfully!"
echo "Final catalog, model checkpoints, and diagnostic plots are saved in results/"
echo "========================================================================"

#!/bin/bash
# ==============================================================================
# Setup dedicated virtual environment for PFS Spectral Classification
# Environment Name: pfs_specclass
# ==============================================================================

set -e

ENV_NAME="pfs_specclass"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "Creating dedicated virtual environment: $ENV_NAME"
echo "Work Directory: $WORK_DIR"
echo "========================================================================"

# Check for conda
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
elif [ -f "/home/yasuda/work/lsst_stack/conda/bin/conda" ]; then
    CONDA_CMD="/home/yasuda/work/lsst_stack/conda/bin/conda"
else
    echo "ERROR: conda executable not found. Please ensure conda is installed."
    exit 1
fi

echo "Using conda at: $CONDA_CMD"

# Check if environment already exists
if $CONDA_CMD env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    echo "Updating packages from environment.yml..."
    $CONDA_CMD env update -n "$ENV_NAME" -f "$WORK_DIR/environment.yml" --prune
else
    echo "Creating new conda environment '$ENV_NAME' from environment.yml..."
    $CONDA_CMD env create -f "$WORK_DIR/environment.yml"
fi

echo ""
echo "Installing/verifying PyTorch with CUDA in '$ENV_NAME'..."
# Find env python/pip
ENV_PYTHON=$($CONDA_CMD info --base)/envs/$ENV_NAME/bin/python
ENV_PIP=$($CONDA_CMD info --base)/envs/$ENV_NAME/bin/pip

if [ ! -f "$ENV_PYTHON" ]; then
    # Fallback to local user env search
    ENV_PYTHON=$(conda env list | grep "$ENV_NAME" | awk '{print $NF}')/bin/python
    ENV_PIP=$(dirname "$ENV_PYTHON")/pip
fi

"$ENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || true

echo ""
echo "Verifying environment:"
"$ENV_PYTHON" -c "
import torch, astropy, sklearn, scipy, numpy
print('Environment check PASSED:')
print('  Python:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU Device:', torch.cuda.get_device_name(0))
print('  Astropy:', astropy.__version__)
print('  Scikit-Learn:', sklearn.__version__)
"

echo ""
echo "========================================================================"
echo "Virtual environment '$ENV_NAME' is ready!"
echo "To activate:"
echo "    conda activate $ENV_NAME"
echo "========================================================================"

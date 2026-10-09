"""Configuration for PFS Spectral Unsupervised Learning."""
from pathlib import Path

# Paths
BASE_DIR = Path("/mnt/ugnas/work/PFS/CommFiller/pfs-target-viewer/extracted_targets")
DB_PATH = Path("/mnt/ugnas/work/PFS/CommFiller/pfs-target-viewer/pfs_metadata.sqlite3")
WORK_DIR = Path("/mnt/ugnas/work/PFS/pfs_specClass2")
RESULTS_DIR = WORK_DIR / "results"

# Spectral grid configuration
# PFS coverage extended down to ~370 nm to preserve valid Blue arm pixels (Ly-alpha / UV features)
WL_MIN = 370.0   # nm
WL_MAX = 1260.0  # nm
N_GRID = 4096    # power of 2 for CNN/FFT efficiency

# Log-wavelength grid: ln(lambda)
import numpy as np
LOG_WL_GRID = np.linspace(np.log(WL_MIN), np.log(WL_MAX), N_GRID)
WL_GRID = np.exp(LOG_WL_GRID)

# Rest-frame canonical grid (spans rest-UV ~80nm up to rest-optical/NIR ~1260nm)
WL_REST_MIN = 80.0   # nm (covers Ly-alpha 121.6 nm, C IV 154.9 nm for z up to ~3.6)
WL_REST_MAX = 1260.0 # nm
N_GRID_REST = 4096
LOG_WL_GRID_REST = np.linspace(np.log(WL_REST_MIN), np.log(WL_REST_MAX), N_GRID_REST)
WL_GRID_REST = np.exp(LOG_WL_GRID_REST)

# Arm boundary & detector edge exclusion zones (nm)
# Optimized based on empirical flux variance profiles:
# - Blue extreme edge: < 373 nm (where valid data starts dropping off)
# - NIR extreme edge: > 1255 nm
# - BRN: Red-NIR transition around 932 - 952 nm
# - BMN: Blue-m gap (660.0 - 700.5 nm), m-NIR gap (897.5 - 948.0 nm)
ARM_MASK_BRN = [
    (370.0, 373.0),    # Blue extreme edge
    (932.0, 952.0),    # Red-NIR transition
    (1255.0, 1260.0),  # NIR extreme edge
]

ARM_MASK_BMN = [
    (370.0, 373.0),    # Blue extreme edge
    (660.0, 700.5),    # Blue-m physical gap (preserves H-alpha at 656.3 nm!)
    (897.5, 948.0),    # m-NIR physical gap & NIR edge spike (preserves Ca II triplet in m arm!)
    (1255.0, 1260.0),  # NIR extreme edge
]

# Training defaults
BATCH_SIZE = 64
LATENT_DIM = 32
NUM_WORKERS = 4


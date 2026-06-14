"""
config.py — Global Configuration for SisFall Project
All constants, paths, and experimental settings are centralized here.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_data_dir() -> str:
    """Automatically find the SisFall_dataset directory in common locations.

    Priority:
      1. Environment variable SISFALL_DIR (if set by user)
      2. data/SisFall_dataset       (standard repo, per data/README.md)
      3. 9.5Dataset/SisFall_dataset (local root structure)
    If not found, return standard path (data/) to report a clear error.
    """
    candidates = [
        os.environ.get("SISFALL_DIR"),
        os.path.join(BASE_DIR, "data", "SisFall_dataset"),
        os.path.join(BASE_DIR, "SisFall Dataset", "SisFall_dataset"),
        os.path.join(BASE_DIR, "9.5Dataset", "SisFall_dataset"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return os.path.join(BASE_DIR, "data", "SisFall_dataset")


DATA_DIR    = _resolve_data_dir()
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
MODEL_DIR   = os.path.join(RESULTS_DIR, "models")
METRIC_DIR  = os.path.join(RESULTS_DIR, "metrics")

# ─── Sensors ────────────────────────────────────────────────────────────────
SENSOR_COLS = ['ADXL_x', 'ADXL_y', 'ADXL_z',
               'ITG_x',  'ITG_y',  'ITG_z',
               'MMA_x',  'MMA_y',  'MMA_z']

# Physical unit conversion factors
ADXL_FACTOR = 32.0 / 8192.0    # -> g
ITG_FACTOR  = 1.0  / 14.375    # -> degree/s
MMA_FACTOR  = 16.0 / 16384.0   # -> g

# ─── 7 Sensor Combinations (Main contribution of the project) ───────────────────────
SENSOR_VARIANTS = {
    "ADXL":     ['ADXL_x', 'ADXL_y', 'ADXL_z'],
    "MMA":      ['MMA_x',  'MMA_y',  'MMA_z'],
    "ITG":      ['ITG_x',  'ITG_y',  'ITG_z'],
    "ADXL_ITG": ['ADXL_x', 'ADXL_y', 'ADXL_z', 'ITG_x', 'ITG_y', 'ITG_z'],
    "MMA_ITG":  ['MMA_x',  'MMA_y',  'MMA_z',  'ITG_x', 'ITG_y', 'ITG_z'],
    "ADXL_MMA": ['ADXL_x', 'ADXL_y', 'ADXL_z', 'MMA_x', 'MMA_y', 'MMA_z'],
    "ALL9":     ['ADXL_x', 'ADXL_y', 'ADXL_z',
                 'ITG_x',  'ITG_y',  'ITG_z',
                 'MMA_x',  'MMA_y',  'MMA_z'],
}

# ─── Signal Processing ──────────────────────────────────────────────────────────
FS          = 200       # sampling frequency (Hz)
# Cutoff 15Hz: retains fall impact spike (5-15Hz), significantly higher than old 5Hz
# (Old 5Hz cuts out the most important fall characteristic -> limits model)
CUTOFF_HZ   = 15.0
FILTER_ORDER = 4

# ─── Windowing ───────────────────────────────────────────────────────────────
WINDOW_SIZE = 400       # 2 seconds * 200Hz
STEP_SIZE   = 200       # sliding step 1 second (50% overlap)

# ─── Label refinement ────────────────────────────────────────────────────────
# In Fall files (F*), windows with AVM_max < FALL_AVM_THRESHOLD will be discarded
# (these are "walking before fall" or "lying after fall" windows - not the fall event)
# Unit: g (after multiplying with ADXL_FACTOR/MMA_FACTOR)
# Update: switch to ADL_LABEL_MODE (drop / keep_as_adl)
USE_LABEL_REFINEMENT = True
FALL_AVM_THRESHOLD   = 1.8       # g - from EDA analysis: fall peak usually > 2g
FALL_LABEL_MODE      = "drop"    # 'drop' = discard non-fall windows / 'keep_as_adl' = label=0

# ─── Labels ────────────────────────────────────────────────────────────────────
LABEL_FALL = 1
LABEL_ADL  = 0

# ─── Subject-wise split ───────────────────────────────────────────────
# SA01-SA18: train | SA19-SA21: val | SA22-SA23: test (intra-age)
# SE01-SE15: test  (cross-age evaluation)
TRAIN_SUBJECTS = [f"SA{i:02d}" for i in range(1, 19)]   # SA01-SA18
VAL_SUBJECTS   = [f"SA{i:02d}" for i in range(19, 22)]  # SA19-SA21
TEST_SA        = [f"SA{i:02d}" for i in range(22, 24)]  # SA22-SA23
TEST_SE        = [f"SE{i:02d}" for i in range(1, 16)]   # SE01-SE15

# ─── ML GridSearchCV ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5
N_JOBS       = 2       # use 2 CPU cores to avoid MemoryError on Windows

# Extended grid v2: added 1-2 new candidates per model. ML features already include jerk+SMA+tilt.
RF_PARAM_GRID = {
    'n_estimators':      [300],
    'max_depth':         [None, 30],
    'min_samples_split': [2, 5],
    'class_weight':      ['balanced'],
}

SVM_PARAM_GRID = {
    'C':      [1, 10, 30],
    'kernel': ['rbf'],
    'gamma':  ['scale'],
    'class_weight': ['balanced'],
}

KNN_PARAM_GRID = {
    'n_neighbors': [5, 7, 11],
    'weights':     ['distance'],
    'metric':      ['euclidean'],
}

# SVM/KNN parameter sets after label refinement (data will be smaller)
ML_MAX_SAMPLES = {'SVM': 30000, 'RF': None, 'KNN': 30000}

# ─── DL (PyTorch CNN-LSTM on GPU) ──────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 60
PATIENCE    = 10        # early stopping
LR          = 1e-3
DL_DEVICE   = "cuda"    # "cuda" | "cpu" - auto fallback if CUDA is unavailable
DL_ARCHS    = ['cnn_lstm', 'cnn_bilstm_attention']

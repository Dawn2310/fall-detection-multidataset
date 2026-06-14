"""
data_loader.py — Read, convert physical units, filter SisFall signals
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from config import (DATA_DIR, SENSOR_COLS, ADXL_FACTOR, ITG_FACTOR,
                    MMA_FACTOR, FS, CUTOFF_HZ, FILTER_ORDER)


# ─── Read a single file ──────────────────────────────────────────────────────
def load_raw_file(filepath: str) -> pd.DataFrame:
    """Read SisFall .txt file, convert bits -> physical units."""
    df = pd.read_csv(filepath, names=SENSOR_COLS, sep=',', engine='python')
    df['MMA_z'] = (df['MMA_z'].astype(str)
                   .str.replace(';', '', regex=False)
                   .astype(float))
    df[['ADXL_x','ADXL_y','ADXL_z']] *= ADXL_FACTOR
    df[['ITG_x', 'ITG_y', 'ITG_z']]  *= ITG_FACTOR
    df[['MMA_x', 'MMA_y', 'MMA_z']]  *= MMA_FACTOR
    return df


# ─── Butterworth Filter ──────────────────────────────────────────────────────
def butterworth_filter(data: np.ndarray,
                       cutoff: float = CUTOFF_HZ,
                       fs: float = FS,
                       order: int = FILTER_ORDER) -> np.ndarray:
    """4th-order low-pass Butterworth filter."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


# ─── Load entire dataset ─────────────────────────────────────────────────────
def load_dataset(subjects: list[str] | None = None,
                 apply_filter: bool = True,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Read all files in DATA_DIR, return DataFrame with added columns:
        subject, activity_code, activity_type (Fall/ADL), label (0/1),
        trial, filepath

    subjects: None = read all, or pass a list ['SA01','SA02',...]
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data not found at: {DATA_DIR}")

    all_subjects = [d for d in sorted(os.listdir(DATA_DIR))
                    if os.path.isdir(os.path.join(DATA_DIR, d))
                    and (d.startswith('SA') or d.startswith('SE'))]

    if subjects is not None:
        all_subjects = [s for s in all_subjects if s in subjects]

    records = []
    for subj in all_subjects:
        subj_path = os.path.join(DATA_DIR, subj)
        files = sorted(f for f in os.listdir(subj_path) if f.endswith('.txt'))
        for fname in files:
            # File name format: F01_SA01_R01.txt or D01_SA01_R01.txt
            parts = fname.replace('.txt', '').split('_')
            if len(parts) != 3:
                continue
            act_code, subj_id, trial = parts
            act_type = 'Fall' if act_code.startswith('F') else 'ADL'
            label    = 1 if act_type == 'Fall' else 0

            fpath = os.path.join(subj_path, fname)
            try:
                df = load_raw_file(fpath)
                if apply_filter:
                    filtered = butterworth_filter(df[SENSOR_COLS].values)
                    df[SENSOR_COLS] = filtered

                df['subject']       = subj_id
                df['activity_code'] = act_code
                df['activity_type'] = act_type
                df['label']         = label
                df['trial']         = trial
                df['filepath']      = fpath
                df['age_group']     = 'SA' if subj_id.startswith('SA') else 'SE'
                records.append(df)
            except Exception as e:
                if verbose:
                    print(f"  Error reading {fname}: {e}")

    if not records:
        raise ValueError("Could not read any files!")

    full_df = pd.concat(records, ignore_index=True)
    if verbose:
        n_files = full_df.groupby(['subject','activity_code','trial']).ngroups
        print(f"Loaded: {n_files} files from {len(all_subjects)} subjects")
        print(f"  Fall samples: {(full_df['label']==1).sum():,}")
        print(f"  ADL  samples: {(full_df['label']==0).sum():,}")
    return full_df


# ─── Utilities ────────────────────────────────────────────────────────────────
def get_file_metadata(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Return metadata table for all files (without reading content)."""
    rows = []
    for subj in sorted(os.listdir(data_dir)):
        subj_path = os.path.join(data_dir, subj)
        if not os.path.isdir(subj_path):
            continue
        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith('.txt'):
                continue
            parts = fname.replace('.txt','').split('_')
            if len(parts) != 3:
                continue
            act_code, subj_id, trial = parts
            fpath = os.path.join(subj_path, fname)
            n_samples = sum(1 for _ in open(fpath))
            rows.append({
                'subject':       subj_id,
                'age_group':     'SA' if subj_id.startswith('SA') else 'SE',
                'activity_code': act_code,
                'activity_type': 'Fall' if act_code.startswith('F') else 'ADL',
                'label':         1 if act_code.startswith('F') else 0,
                'trial':         trial,
                'n_samples':     n_samples,
                'duration_s':    n_samples / FS,
                'filepath':      fpath,
            })
    return pd.DataFrame(rows)


def compute_avm(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Calculate Acceleration Vector Magnitude (AVM) for a sensor."""
    return np.sqrt(df[f'{prefix}_x']**2 +
                   df[f'{prefix}_y']**2 +
                   df[f'{prefix}_z']**2)

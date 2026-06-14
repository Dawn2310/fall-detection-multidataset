"""
upfall_loader.py — Load UP-Fall dataset for cross-dataset evaluation.

UP-Fall (Martínez-Villaseñor et al., Sensors 2019):
  - 17 subjects (ages 18-24)
  - 5 IMU sensors (Ankle, RightPocket, Belt, Neck, Wrist) at ~20 Hz
  - 5 Fall activities (Activity 1-5) + 6 ADL activities (Activity 6-11)
  - Tag column: for fall recordings, Tag == Activity number marks the
    actual fall moment; other tags are surrounding ADL context
  - 17 subjects × 11 activities × 3 trials = 561 recordings total

Sensor mapping to SisFall (Belt-worn IMU):
  UP-Fall BeltAccelerometer X/Y/Z -> MMA_x/y/z (already in g)
  UP-Fall BeltAngularVelocity X/Y/Z -> ITG_x/y/z (already in deg/s)

Belt columns in CompleteDataSet.csv:
  col 15: BeltAccelerometer   (x-axis, g)
  col 16: Unnamed: 16         (y-axis, g)
  col 17: Unnamed: 17         (z-axis, g)
  col 18: BeltAngularVelocity (x-axis, deg/s)
  col 19: Unnamed: 19         (y-axis, deg/s)
  col 20: Unnamed: 20         (z-axis, deg/s)

Usage:
    from upfall_loader import load_upfall_dataset
    X, y, meta = load_upfall_dataset('UP-Fall dataset/')
"""
import os
import gc
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

from config import WINDOW_SIZE, STEP_SIZE, CUTOFF_HZ, FS, FILTER_ORDER
from data_loader import butterworth_filter


# ─── UP-Fall Specifications ──────────────────────────────────────────────────
UPFALL_FS = 18          # actual ~18-20 Hz (median interval ~52ms)
SISFALL_FS = FS         # = 200 Hz (config.py)
RESAMPLE_UP = 10        # 18*10 = 180, then down=9 -> ~200 Hz
RESAMPLE_DOWN = 9       # 18 * 10/9 = 20 -> then use 200/20 = 10

# Belt sensor columns in the CSV
BELT_ACCEL_COLS = ['BeltAccelerometer', 'Unnamed: 16', 'Unnamed: 17']
BELT_GYRO_COLS  = ['BeltAngularVelocity', 'Unnamed: 19', 'Unnamed: 20']

# Activity mapping: 1-5 = Fall types, 6-11 = ADL types
FALL_ACTIVITIES = {1, 2, 3, 4, 5}
ADL_ACTIVITIES  = {6, 7, 8, 9, 10, 11}

# Activity descriptions (from UP-Fall paper)
ACTIVITY_NAMES = {
    1:  'Falling forward using hands',
    2:  'Falling forward using knees',
    3:  'Falling backwards',
    4:  'Falling sitting in empty chair',
    5:  'Falling sideways',
    6:  'Walking',
    7:  'Standing',
    8:  'Sitting',
    9:  'Picking up an object',
    10: 'Jumping',
    11: 'Laying down',
}


def _compute_actual_fs(timestamps: pd.Series) -> float:
    """Compute actual sampling frequency from timestamps."""
    ts = pd.to_datetime(timestamps)
    diffs = ts.diff().dropna().dt.total_seconds()
    median_dt = diffs.median()
    return 1.0 / median_dt if median_dt > 0 else 20.0


# ─── Resample ~18 Hz -> 200 Hz ───────────────────────────────────────────────
def upsample_upfall(data: np.ndarray, actual_fs: float = None) -> np.ndarray:
    """Resample UP-Fall data from actual_fs to 200 Hz using rational resampling.

    Uses resample_poly with appropriate up/down ratios to achieve ~200Hz.
    """
    if actual_fs is None:
        actual_fs = UPFALL_FS

    # Find best rational approximation for 200/actual_fs
    # For ~18-20 Hz, use up=200, down=round(actual_fs) to be precise
    target_fs = SISFALL_FS  # 200
    # Simplify ratio: 200/18 ≈ 100/9, 200/20 = 10/1
    # Use GCD approach for cleaner ratio
    from math import gcd
    actual_rounded = round(actual_fs)
    if actual_rounded <= 0:
        actual_rounded = 18
    g = gcd(target_fs, actual_rounded)
    up = target_fs // g
    down = actual_rounded // g

    return resample_poly(data, up=up, down=down, axis=0).astype(np.float32)


# ─── Build windows from one recording ────────────────────────────────────────
def recording_to_windows(data: np.ndarray,
                         window_size: int = WINDOW_SIZE,
                         step_size: int = STEP_SIZE,
                         apply_filter: bool = True) -> np.ndarray:
    """
    Take (N, 6) sensor data (already resampled to 200Hz)
    -> sliding windows (M, 400, 6).
    Output channels: [AccX, AccY, AccZ, GyrX, GyrY, GyrZ]
    """
    # Butterworth filter (same 15Hz cutoff as SisFall)
    if apply_filter and len(data) > window_size:
        data = butterworth_filter(data, cutoff=CUTOFF_HZ,
                                   fs=SISFALL_FS, order=FILTER_ORDER).astype(np.float32)

    # Sliding windows
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[start:start + window_size])
    return np.array(windows) if windows else np.empty((0, window_size, 6), dtype=np.float32)


# ─── Label refinement for Fall windows ────────────────────────────────────────
def _refine_fall_windows_avm(windows: np.ndarray,
                              threshold: float = 1.8) -> tuple:
    """Keep Fall windows with AVM peak >= threshold (g)."""
    avm = np.sqrt(np.sum(windows[:, :, :3]**2, axis=2))  # accel only
    peaks = avm.max(axis=1)
    keep = peaks >= threshold
    return windows[keep], keep


# ─── Load entire UP-Fall dataset ─────────────────────────────────────────────
def load_upfall_dataset(upfall_dir: str,
                        use_tag_labels: bool = False,
                        apply_label_refinement: bool = True,
                        fall_avm_threshold: float = 1.8,
                        verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load UP-Fall dataset from CompleteDataSet.csv.

    Args:
        upfall_dir: Path to folder containing CompleteDataSet.csv
        use_tag_labels: If True, use Tag column for precise fall labeling
                       (Tag == Activity for fall activities = actual fall moment).
                       If False, treat entire Fall recordings as fall events.
        apply_label_refinement: Apply AVM-based refinement on fall windows
        fall_avm_threshold: AVM peak threshold (g) for fall refinement
        verbose: Print progress

    Returns:
        X:    (N, 400, 6) windows [accX, accY, accZ, gyrX, gyrY, gyrZ]
        y:    (N,) labels 0=ADL, 1=Fall
        meta: list of dicts {subject, activity_code, activity_type, trial, ...}
    """
    csv_path = os.path.join(upfall_dir, 'CompleteDataSet.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"UP-Fall CSV not found: {csv_path}")

    if verbose:
        print(f"[UP-Fall] Loading {csv_path} ...")

    # Read CSV, skip the unit-description row (row index 1 in raw = sub-header)
    df = pd.read_csv(csv_path, skiprows=[1])

    # Compute actual sampling frequency
    actual_fs = _compute_actual_fs(df['TimeStamps'].head(200))
    if verbose:
        print(f"[UP-Fall] Detected sampling rate: {actual_fs:.1f} Hz -> resample to {SISFALL_FS} Hz")
        print(f"[UP-Fall] Total samples: {len(df):,}")
        print(f"[UP-Fall] Subjects: {sorted(df['Subject'].dropna().unique().astype(int))}")
        print(f"[UP-Fall] Tag-based fall labeling: {use_tag_labels}")

    # Extract Belt sensor data
    belt_data = df[BELT_ACCEL_COLS + BELT_GYRO_COLS].values.astype(np.float32)

    all_X, all_y, all_meta = [], [], []
    n_fall_raw, n_fall_kept, n_adl = 0, 0, 0
    skipped = 0

    # Group by (Subject, Activity, Trial) = one recording
    groups = df.groupby(['Subject', 'Activity', 'Trial'])
    n_groups = len(groups)

    if verbose:
        print(f"[UP-Fall] {n_groups} recordings (Subject×Activity×Trial)")

    iterator = tqdm(groups, total=n_groups, desc="UP-Fall recordings") if verbose else groups
    for (subj, act, trial), group_df in iterator:
        subj = int(subj)
        act = int(act)
        trial = int(trial)
        is_fall_activity = act in FALL_ACTIVITIES

        # Get indices for this group
        idx = group_df.index
        recording_belt = belt_data[idx]

        if len(recording_belt) < 3:
            skipped += 1
            continue

        # Check for NaN
        if np.isnan(recording_belt).any():
            recording_belt = np.nan_to_num(recording_belt, nan=0.0)

        # For fall activities with tag-based labeling:
        # Split into fall-tagged and ADL-tagged portions
        if is_fall_activity and use_tag_labels:
            tags = group_df['Tag'].values
            fall_mask = tags == act  # Tag matches Activity = actual fall moment

            # Process fall portion
            fall_data = recording_belt[fall_mask]
            if len(fall_data) >= 3:
                fall_resampled = upsample_upfall(fall_data, actual_fs)
                fall_windows = recording_to_windows(fall_resampled)

                if len(fall_windows) > 0:
                    n_fall_raw += len(fall_windows)

                    if apply_label_refinement:
                        fall_windows, _ = _refine_fall_windows_avm(
                            fall_windows, fall_avm_threshold)

                    if len(fall_windows) > 0:
                        n_fall_kept += len(fall_windows)
                        all_X.append(fall_windows)
                        all_y.append(np.full(len(fall_windows), 1, dtype=np.int8))
                        for _ in range(len(fall_windows)):
                            all_meta.append({
                                'subject': f"S{subj:02d}",
                                'activity_code': f"A{act:02d}_fall",
                                'activity_type': 'Fall',
                                'activity_name': ACTIVITY_NAMES.get(act, f'Activity_{act}'),
                                'trial': trial,
                                'source_file': f'S{subj:02d}_A{act:02d}_T{trial}_fall',
                            })
                    del fall_windows
                del fall_data

            # Process ADL context portion (non-fall-tagged samples)
            adl_data = recording_belt[~fall_mask]
            if len(adl_data) >= 3:
                adl_resampled = upsample_upfall(adl_data, actual_fs)
                adl_windows = recording_to_windows(adl_resampled)

                if len(adl_windows) > 0:
                    n_adl += len(adl_windows)
                    all_X.append(adl_windows)
                    all_y.append(np.full(len(adl_windows), 0, dtype=np.int8))
                    for _ in range(len(adl_windows)):
                        all_meta.append({
                            'subject': f"S{subj:02d}",
                            'activity_code': f"A{act:02d}_adl_context",
                            'activity_type': 'ADL',
                            'activity_name': ACTIVITY_NAMES.get(act, f'Activity_{act}') + ' (context)',
                            'trial': trial,
                            'source_file': f'S{subj:02d}_A{act:02d}_T{trial}_adl',
                        })
                del adl_windows
                del adl_data

        elif is_fall_activity and not use_tag_labels:
            # Treat entire recording as fall
            resampled = upsample_upfall(recording_belt, actual_fs)
            windows = recording_to_windows(resampled)

            if len(windows) > 0:
                n_fall_raw += len(windows)

                if apply_label_refinement:
                    windows, _ = _refine_fall_windows_avm(windows, fall_avm_threshold)

                if len(windows) > 0:
                    n_fall_kept += len(windows)
                    all_X.append(windows)
                    all_y.append(np.full(len(windows), 1, dtype=np.int8))
                    for _ in range(len(windows)):
                        all_meta.append({
                            'subject': f"S{subj:02d}",
                            'activity_code': f"A{act:02d}",
                            'activity_type': 'Fall',
                            'activity_name': ACTIVITY_NAMES.get(act, f'Activity_{act}'),
                            'trial': trial,
                            'source_file': f'S{subj:02d}_A{act:02d}_T{trial}',
                        })
            del windows

        else:
            # ADL activity (6-11): entire recording is ADL
            resampled = upsample_upfall(recording_belt, actual_fs)
            windows = recording_to_windows(resampled)

            if len(windows) > 0:
                n_adl += len(windows)
                all_X.append(windows)
                all_y.append(np.full(len(windows), 0, dtype=np.int8))
                for _ in range(len(windows)):
                    all_meta.append({
                        'subject': f"S{subj:02d}",
                        'activity_code': f"A{act:02d}",
                        'activity_type': 'ADL',
                        'activity_name': ACTIVITY_NAMES.get(act, f'Activity_{act}'),
                        'trial': trial,
                        'source_file': f'S{subj:02d}_A{act:02d}_T{trial}',
                    })
            del windows

        del recording_belt
        gc.collect()

    if not all_X:
        raise RuntimeError("No valid recordings loaded from UP-Fall.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    del all_X, all_y
    gc.collect()

    if verbose:
        print(f"\n[UP-Fall] Total windows: {len(X):,}")
        print(f"  Fall: {(y==1).sum():,} (kept from {n_fall_raw:,} raw)")
        print(f"  ADL:  {(y==0).sum():,}")
        print(f"  Skipped recordings: {skipped}")
        print(f"  Shape: {X.shape}")
        if apply_label_refinement:
            dropped = n_fall_raw - n_fall_kept
            pct = dropped / max(n_fall_raw, 1) * 100
            print(f"  AVM refinement: dropped {dropped:,} ({pct:.1f}%)")
        subjects_found = sorted(set(m['subject'] for m in all_meta))
        print(f"  Subjects: {subjects_found}")

    return X, y, all_meta


# ─── CLI test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Load UP-Fall dataset.")
    parser.add_argument("--dir", default="UP-Fall dataset",
                        help="Path to UP-Fall folder containing CompleteDataSet.csv")
    parser.add_argument("--no_refine", action="store_true",
                        help="Disable AVM label refinement")
    parser.add_argument("--no_tag", action="store_true",
                        help="Don't use Tag column for fall labeling")
    args = parser.parse_args()

    X, y, meta = load_upfall_dataset(
        args.dir,
        use_tag_labels=not args.no_tag,
        apply_label_refinement=not args.no_refine,
    )
    print(f"\nDone! X.shape={X.shape}, y.shape={y.shape}")
    subjects = sorted(set(m['subject'] for m in meta))
    print(f"Subjects: {subjects}")
    print(f"Sample meta: {meta[0]}")

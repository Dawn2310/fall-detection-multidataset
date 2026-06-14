"""
kfall_loader.py — Read KFall dataset (Yu et al., Frontiers in Aging Neuroscience 2021)
and map to SisFall format to reuse the pipeline.

KFall specs (Yu et al. 2021):
  - 32 male subjects (SA06-SA38, no SA01-05/SA34), age 24.9 +/- 3.7
  - 9-axis IMU (accel + gyro + euler) worn at lower back, 100 Hz
  - 21 ADL (T01-T19, T35-T36) + 15 Fall types (T20-T34 = F01-F15)
  - 5075 motion files (2443 ADL + 2632 Fall)
  - Has label "Fall_onset_frame" + "Fall_impact_frame" from video sync

File naming: S{subj_num}T{task:02d}R{trial:02d}.csv
  e.g.: S06T01R01.csv = Subject SA06, Task 01, Trial 01
  Task 01-19, 35-36 = ADL
  Task 20-34 = Fall (F01-F15)

Columns: TimeStamp(s), FrameCounter, AccX, AccY, AccZ, GyrX, GyrY, GyrZ,
         EulerX, EulerY, EulerZ
  AccX/Y/Z: unit g
  GyrX/Y/Z: unit deg/s

Map to SisFall format:
  KFall AccX/Y/Z -> MMA_x/y/z (SisFall accel, already in unit g)
  KFall GyrX/Y/Z -> ITG_x/y/z (SisFall gyro)
  -> Fits sensor variant MMA_ITG (6 channels)

Usage:
    from kfall_loader import load_kfall_dataset
    X, y, meta = load_kfall_dataset('K-Fall dataset/')
"""
import os
import gc
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

from config import WINDOW_SIZE, STEP_SIZE, CUTOFF_HZ, FS, FILTER_ORDER
from data_loader import butterworth_filter


# ─── KFall Specifications ────────────────────────────────────────────────────
KFALL_FS = 100                  # KFall sampling rate (Hz)
SISFALL_FS = FS                 # = 200 Hz (config.py)
RESAMPLE_RATIO = SISFALL_FS // KFALL_FS  # = 2

KFALL_ACCEL_COLS = ['AccX', 'AccY', 'AccZ']
KFALL_GYRO_COLS  = ['GyrX', 'GyrY', 'GyrZ']

FALL_TASKS = set(range(20, 35))  # T20-T34 = F01-F15


def _parse_filename(fname: str) -> dict:
    """Parse S06T01R01.csv -> {subject_num: 6, task: 1, trial: 1}"""
    name = fname.replace('.csv', '')
    s_idx = name.find('S')
    t_idx = name.find('T')
    r_idx = name.find('R')
    return {
        'subject_num': int(name[s_idx+1:t_idx]),
        'task': int(name[t_idx+1:r_idx]),
        'trial': int(name[r_idx+1:]),
    }


def _is_fall_task(task_num: int) -> bool:
    return task_num in FALL_TASKS


# ─── Load label data (Fall onset/impact frames) ─────────────────────────────
def load_labels(label_dir: str, subject: str) -> dict:
    """Load Fall_onset_frame/Fall_impact_frame from label Excel.
    Returns dict: (task_num, trial) -> (onset_frame, impact_frame)
    """
    label_path = os.path.join(label_dir, f"{subject}_label.xlsx")
    if not os.path.exists(label_path):
        return {}

    df = pd.read_excel(label_path)
    df['Task Code (Task ID)'] = df['Task Code (Task ID)'].ffill()
    df['Description'] = df['Description'].ffill()

    labels = {}
    for _, row in df.iterrows():
        tc = str(row['Task Code (Task ID)'])
        # Extract task ID number: "F01 (20)" -> 20
        try:
            task_id = int(tc.split('(')[1].replace(')', '').strip())
        except (IndexError, ValueError):
            continue
        trial = int(row['Trial ID'])
        onset = int(row['Fall_onset_frame']) if pd.notna(row['Fall_onset_frame']) else None
        impact = int(row['Fall_impact_frame']) if pd.notna(row['Fall_impact_frame']) else None
        labels[(task_id, trial)] = (onset, impact)
    return labels


# ─── Resample 100 -> 200 Hz ──────────────────────────────────────────────────
def upsample_kfall(data: np.ndarray) -> np.ndarray:
    return resample_poly(data, up=RESAMPLE_RATIO, down=1, axis=0).astype(np.float32)


# ─── Build windows from a single KFall file ─────────────────────────────────
def file_to_windows_kfall(filepath: str,
                          window_size: int = WINDOW_SIZE,
                          step_size: int = STEP_SIZE,
                          apply_filter: bool = True) -> np.ndarray:
    """
    Read 1 KFall CSV file -> sliding windows (N, T, 6).
    Output: [AccX, AccY, AccZ, GyrX, GyrY, GyrZ]
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    needed = KFALL_ACCEL_COLS + KFALL_GYRO_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"KFall file missing columns: {missing}. Has: {list(df.columns)}")

    data = df[KFALL_ACCEL_COLS + KFALL_GYRO_COLS].values.astype(np.float32)

    # Upsample 100 -> 200 Hz
    data = upsample_kfall(data)

    # Butterworth filter (same 15Hz cutoff as SisFall)
    if apply_filter and len(data) > window_size:
        data = butterworth_filter(data, cutoff=CUTOFF_HZ,
                                   fs=SISFALL_FS, order=FILTER_ORDER).astype(np.float32)

    # Sliding windows
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[start:start + window_size])
    return np.array(windows) if windows else np.empty((0, window_size, 6), dtype=np.float32)


# ─── Label refinement for Fall windows ───────────────────────────────────────
def _refine_fall_windows_avm(windows: np.ndarray,
                              threshold: float = 1.8) -> np.ndarray:
    """Keep Fall windows with AVM peak >= threshold (g)."""
    avm = np.sqrt(np.sum(windows[:, :, :3]**2, axis=2))  # accel only
    peaks = avm.max(axis=1)
    keep = peaks >= threshold
    return windows[keep], keep


def _refine_fall_windows_label(windows: np.ndarray,
                                onset_frame: int,
                                impact_frame: int,
                                original_fs: int = KFALL_FS,
                                window_size: int = WINDOW_SIZE,
                                step_size: int = STEP_SIZE) -> np.ndarray:
    """Keep Fall windows containing impact moment (from label).
    onset_frame/impact_frame are frames at 100Hz -> convert to sample index at 200Hz.
    """
    onset_200 = onset_frame * RESAMPLE_RATIO
    impact_200 = impact_frame * RESAMPLE_RATIO

    keep = []
    for i in range(len(windows)):
        w_start = i * step_size
        w_end = w_start + window_size
        # Window overlaps with [onset, impact+margin]
        margin = window_size // 2  # add 1s margin after impact
        if w_start <= impact_200 + margin and w_end >= onset_200:
            keep.append(True)
        else:
            keep.append(False)
    keep = np.array(keep)
    return windows[keep], keep


# ─── Load entire KFall dataset ───────────────────────────────────────────────
def load_kfall_dataset(kfall_dir: str,
                       subjects: list[str] | None = None,
                       apply_label_refinement: bool = True,
                       refinement_method: str = 'label',
                       fall_avm_threshold: float = 1.8,
                       verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load entire KFall dataset, return:
        X    : (N, 400, 6) resampled + filtered windows
        y    : (N,) label 0/1
        meta : list dict {subject, activity_code, ...}

    refinement_method:
        'label' = use Fall_onset_frame/Fall_impact_frame from Excel (most accurate)
        'avm'   = use AVM peak >= threshold (like SisFall)
        'none'  = no refinement
    """
    sensor_dir = os.path.join(kfall_dir, 'sensor_data_new')
    label_dir = os.path.join(kfall_dir, 'label_data_new')

    if not os.path.isdir(sensor_dir):
        raise FileNotFoundError(f"KFall sensor_data_new not found: {sensor_dir}")

    all_subjects = sorted([d for d in os.listdir(sensor_dir)
                           if os.path.isdir(os.path.join(sensor_dir, d))])
    if subjects is not None:
        all_subjects = [s for s in all_subjects if s in subjects]

    if verbose:
        print(f"[KFall] Reading {len(all_subjects)} subjects from {kfall_dir}")
        print(f"[KFall] Refinement: {refinement_method}")

    all_X, all_y, all_meta = [], [], []
    n_fall_total = 0
    n_fall_kept = 0
    n_adl_total = 0

    iterator = tqdm(all_subjects, desc="KFall subjects") if verbose else all_subjects
    for subj in iterator:
        subj_dir = os.path.join(sensor_dir, subj)

        # Load labels for this subject
        fall_labels = {}
        if apply_label_refinement and refinement_method == 'label':
            fall_labels = load_labels(label_dir, subj)

        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.csv')])
        for fname in files:
            fpath = os.path.join(subj_dir, fname)
            try:
                info = _parse_filename(fname)
                task_num = info['task']
                trial_num = info['trial']
                is_fall = _is_fall_task(task_num)

                windows = file_to_windows_kfall(fpath)
                if len(windows) == 0:
                    continue

                if is_fall:
                    n_fall_total += len(windows)

                    if apply_label_refinement and refinement_method != 'none':
                        if refinement_method == 'label' and (task_num, trial_num) in fall_labels:
                            onset, impact = fall_labels[(task_num, trial_num)]
                            if onset is not None and impact is not None:
                                windows, keep = _refine_fall_windows_label(
                                    windows, onset, impact)
                            else:
                                windows, keep = _refine_fall_windows_avm(
                                    windows, fall_avm_threshold)
                        else:
                            windows, keep = _refine_fall_windows_avm(
                                windows, fall_avm_threshold)

                        if len(windows) == 0:
                            continue
                    n_fall_kept += len(windows)
                    label = 1
                else:
                    n_adl_total += len(windows)
                    label = 0

                all_X.append(windows)
                all_y.append(np.full(len(windows), label, dtype=np.int8))

                task_code = f"T{task_num:02d}"
                for _ in range(len(windows)):
                    all_meta.append({
                        'subject': subj,
                        'activity_code': task_code,
                        'activity_type': 'Fall' if is_fall else 'ADL',
                        'trial': trial_num,
                        'source_file': fname,
                    })
                del windows
            except Exception as e:
                if verbose:
                    tqdm.write(f"  [skip] {fname}: {e}")
        gc.collect()

    if not all_X:
        raise RuntimeError("Could not read any files from KFall.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    if verbose:
        print(f"\n[KFall] Total windows: {len(X):,}")
        print(f"  Fall windows: {(y==1).sum():,} (kept from {n_fall_total:,} raw)")
        print(f"  ADL windows:  {(y==0).sum():,}")
        print(f"  Shape: {X.shape}")
        if apply_label_refinement and refinement_method != 'none':
            dropped = n_fall_total - n_fall_kept
            print(f"  Label refinement ({refinement_method}): kept {n_fall_kept:,}, "
                  f"dropped {dropped:,} ({dropped/max(n_fall_total,1)*100:.1f}%)")

    return X, y, all_meta


# ─── CLI test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Load KFall dataset.")
    parser.add_argument("--kfall_dir", default="K-Fall dataset",
                        help="Path to KFall root folder")
    parser.add_argument("--refinement", default="label",
                        choices=["label", "avm", "none"])
    parser.add_argument("--first_subject_only", action="store_true")
    args = parser.parse_args()

    subjects = None
    if args.first_subject_only:
        sensor_dir = os.path.join(args.kfall_dir, 'sensor_data_new')
        sub_dirs = sorted(os.listdir(sensor_dir))
        if sub_dirs:
            subjects = [sub_dirs[0]]
            print(f"Smoke test: {subjects}")

    X, y, meta = load_kfall_dataset(
        args.kfall_dir,
        subjects=subjects,
        apply_label_refinement=(args.refinement != 'none'),
        refinement_method=args.refinement,
        verbose=True,
    )
    print(f"\nDone! X.shape={X.shape}, y.shape={y.shape}")
    print(f"Sample meta: {meta[0]}")

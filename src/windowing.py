"""
windowing.py — Cut time series into sliding windows
               and extract labels for each window.
"""
import os
import gc
import numpy as np
from tqdm import tqdm
from config import (DATA_DIR, SENSOR_COLS, WINDOW_SIZE, STEP_SIZE,
                    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SA, TEST_SE,
                    USE_LABEL_REFINEMENT, FALL_AVM_THRESHOLD, FALL_LABEL_MODE)
from data_loader import load_raw_file, butterworth_filter


# ─── AVM-based label refinement ──────────────────────────────────────────────
def _compute_avm_peak(window_3d_first3: np.ndarray) -> float:
    """AVM peak (max value of AVM) from the first 3 axes (always main accelerometer)."""
    avm = np.sqrt(np.sum(window_3d_first3**2, axis=1))
    return float(np.max(avm))


def _refine_fall_labels(windows: np.ndarray,
                        avm_threshold: float,
                        mode: str = "drop") -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate AVM peak for each window (using first 3 axes, assuming it's accelerometer).
    - mode='drop': keep windows with peak >= threshold, drop the rest.
    - mode='keep_as_adl': keep all, but assign label=0 for windows below threshold.

    Returns:
      keep_mask : ndarray (N,) bool
      labels    : ndarray (N,) int8 (1 = still Fall, 0 = downgraded to ADL)
    """
    n = len(windows)
    peaks = np.empty(n, dtype=np.float32)
    for i in range(n):
        peaks[i] = _compute_avm_peak(windows[i, :, :3])

    is_fall = peaks >= avm_threshold
    if mode == "drop":
        return is_fall, np.ones(n, dtype=np.int8)
    elif mode == "keep_as_adl":
        return np.ones(n, dtype=bool), is_fall.astype(np.int8)
    else:
        raise ValueError(f"mode must be 'drop' or 'keep_as_adl', got {mode}")


# ─── Create windows from a series ────────────────────────────────────────────────
def extract_windows_from_series(data: np.ndarray,
                                 window_size: int = WINDOW_SIZE,
                                 step_size: int = STEP_SIZE) -> np.ndarray:
    """
    Cut array (T, C) into windows (N, window_size, C).
    Returns 3D array.
    """
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[start:start + window_size])
    return np.array(windows) if windows else np.empty((0, window_size, data.shape[1]))


# ─── Create the full window dataset ─────────────────────────────────────────────
def build_window_dataset(subjects: list[str] | None = None,
                          sensor_cols: list[str] | None = None,
                          window_size: int = WINDOW_SIZE,
                          step_size: int = STEP_SIZE,
                          apply_filter: bool = True,
                          verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Read all .txt files, cut into windows, return:
        X      : (N, window_size, n_channels)
        y      : (N,) labels 0/1
        meta   : list of dicts containing subject, activity, trial info

    sensor_cols: None = use all SENSOR_COLS (9 channels)
                 or pass specific list of channels to use
    """
    if sensor_cols is None:
        sensor_cols = SENSOR_COLS

    all_subjects = [d for d in sorted(os.listdir(DATA_DIR))
                    if os.path.isdir(os.path.join(DATA_DIR, d))
                    and (d.startswith('SA') or d.startswith('SE'))]
    if subjects is not None:
        all_subjects = [s for s in all_subjects if s in subjects]

    # PASS 1: count total windows first -> pre-allocate (saves 50% peak RAM)
    file_info = []  # (fpath, subj_id, act_code, trial, label, n_windows)
    total_windows = 0
    iter1 = tqdm(all_subjects, desc="Scanning files") if verbose else all_subjects
    for subj in iter1:
        subj_path = os.path.join(DATA_DIR, subj)
        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith('.txt'):
                continue
            parts = fname.replace('.txt', '').split('_')
            if len(parts) != 3:
                continue
            act_code, subj_id, trial = parts
            label = 1 if act_code.startswith('F') else 0
            fpath = os.path.join(subj_path, fname)
            try:
                n_lines = sum(1 for _ in open(fpath))
                n_win = max(0, (n_lines - window_size) // step_size + 1)
                if n_win == 0:
                    continue
                file_info.append((fpath, subj_id, act_code, trial, label, n_win))
                total_windows += n_win
            except Exception:
                continue

    # Pre-allocate
    n_ch = len(sensor_cols)
    X = np.empty((total_windows, window_size, n_ch), dtype=np.float32)
    y = np.empty(total_windows, dtype=np.int8)
    meta_list = []

    # PASS 2: read and fill X, y
    # If label refinement is enabled: for Fall files, calculate AVM peak per window,
    # keep windows exceeding threshold (mode 'drop') or downgrade label (mode 'keep_as_adl').
    cur = 0
    n_fall_dropped = 0
    iter2 = tqdm(file_info, desc="Reading files") if verbose else file_info
    for fpath, subj_id, act_code, trial, label, n_win in iter2:
        try:
            df = load_raw_file(fpath)
            data = df[sensor_cols].values.astype(np.float32)
            if apply_filter:
                data = butterworth_filter(data).astype(np.float32)
            windows = extract_windows_from_series(data, window_size, step_size)
            n_actual = len(windows)
            if n_actual == 0:
                continue

            # === Label refinement only applies to Fall files ===
            # Only refine when the first sensor is an accelerometer (ADXL or MMA).
            # For variants with only ITG (gyro) -> skip refinement.
            has_accel = (sensor_cols[0].startswith('ADXL') or
                         sensor_cols[0].startswith('MMA'))
            if (USE_LABEL_REFINEMENT and label == 1 and n_actual > 0
                    and has_accel):
                if windows.shape[2] >= 3:
                    keep_mask, fall_labels = _refine_fall_labels(
                        windows, FALL_AVM_THRESHOLD, FALL_LABEL_MODE)
                    if FALL_LABEL_MODE == "drop":
                        if not keep_mask.any():
                            # entire file has no window exceeding threshold -> drop file
                            n_fall_dropped += n_actual
                            del df, data, windows
                            continue
                        n_dropped = (~keep_mask).sum()
                        n_fall_dropped += int(n_dropped)
                        windows = windows[keep_mask]
                        win_labels = fall_labels[keep_mask]
                    else:  # keep_as_adl
                        win_labels = fall_labels
                    n_actual = len(windows)
                else:
                    # not enough 3 axes -> cannot refine, use original label
                    win_labels = np.full(n_actual, label, dtype=np.int8)
            else:
                win_labels = np.full(n_actual, label, dtype=np.int8)

            X[cur:cur + n_actual] = windows
            y[cur:cur + n_actual] = win_labels
            age_group = 'SA' if subj_id.startswith('SA') else 'SE'
            for i in range(n_actual):
                act_type = 'Fall' if win_labels[i] == 1 else 'ADL'
                meta_list.append({
                    'subject':       subj_id,
                    'age_group':     age_group,
                    'activity_code': act_code,
                    'activity_type': act_type,
                    'trial':         trial,
                })
            cur += n_actual
            del df, data, windows
        except Exception as e:
            if verbose:
                print(f"  Error {os.path.basename(fpath)}: {type(e).__name__} - {str(e)[:50]}")

    # Trim excess capacity (if n_win estimate didn't perfectly match)
    if cur < total_windows:
        X = X[:cur]
        y = y[:cur]
    gc.collect()

    if verbose:
        print(f"\nTotal windows: {len(X):,}")
        print(f"  Fall: {(y==1).sum():,}  |  ADL: {(y==0).sum():,}")
        print(f"  Shape X: {X.shape}")
        if USE_LABEL_REFINEMENT and n_fall_dropped:
            print(f"  [Label refinement] dropped {n_fall_dropped:,} 'fake-fall' windows "
                  f"(AVM peak < {FALL_AVM_THRESHOLD}g, mode={FALL_LABEL_MODE})")
    return X, y, meta_list


# ─── Subject-wise train / val / test split ────────────────────────────────────
def subject_split(X: np.ndarray,
                  y: np.ndarray,
                  meta: list,
                  train_subjects: list[str] = TRAIN_SUBJECTS,
                  val_subjects:   list[str] = VAL_SUBJECTS,
                  test_subjects:  list[str] | None = None,
                  verbose: bool = True) -> dict:
    """
    Split dataset by subject (prevents data leakage).
    test_subjects: None = TEST_SA + TEST_SE (default)
    """
    if test_subjects is None:
        test_subjects = TEST_SA + TEST_SE

    subj_arr = np.array([m['subject'] for m in meta])

    def mask(subjects):
        return np.isin(subj_arr, subjects)

    splits = {}
    for name, subjs in [('train', train_subjects),
                        ('val',   val_subjects),
                        ('test',  test_subjects)]:
        idx = np.where(mask(subjs))[0]
        splits[name] = {
            'X': X[idx], 'y': y[idx],
            'meta': [meta[i] for i in idx],
            'subjects': subjs,
        }
        if verbose:
            print(f"{name:5s}: {len(idx):6,} windows | "
                  f"Fall={( y[idx]==1).sum():,} ADL={(y[idx]==0).sum():,} | "
                  f"subjects={len(subjs)}")

    # Add dedicated cross-age split
    for name, age in [('test_SA', 'SA'), ('test_SE', 'SE')]:
        age_subjects = [s for s in test_subjects if s.startswith(age)]
        if age_subjects:
            idx = np.where(mask(age_subjects))[0]
            splits[name] = {
                'X': X[idx], 'y': y[idx],
                'meta': [meta[i] for i in idx],
                'subjects': age_subjects,
            }
    return splits


# ─── Cross-age scenario (Train SA, Test SE) ───────────────────────
def cross_age_split(sensor_cols: list[str] | None = None,
                    verbose: bool = True) -> dict:
    """
    Cross-age scenario:
    Train: SA01-SA23 (all young adults)
    Test:  SE01-SE15 (all elderly)

    Note: SE has very few Falls (only SE06), so testing focuses on ADL.
    """
    all_SA = [f"SA{i:02d}" for i in range(1, 24)]
    all_SE = [f"SE{i:02d}" for i in range(1, 16)]

    if verbose:
        print("=== Cross-Age Split (Train SA, Test SE) ===")

    X_sa, y_sa, meta_sa = build_window_dataset(
        subjects=all_SA, sensor_cols=sensor_cols, verbose=verbose)
    X_se, y_se, meta_se = build_window_dataset(
        subjects=all_SE, sensor_cols=sensor_cols, verbose=verbose)

    return {
        'train': {'X': X_sa, 'y': y_sa, 'meta': meta_sa},
        'test':  {'X': X_se, 'y': y_se, 'meta': meta_se},
    }

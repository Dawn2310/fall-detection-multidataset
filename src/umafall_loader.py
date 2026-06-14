"""
umafall_loader.py — Load UMAFall dataset for cross-dataset evaluation.

UMAFall (University of Malaga, Casilari et al.):
  - 19 subjects (ages 14-68, mix young+elderly)
  - WAIST SensorTag: accel + gyro at ~20 Hz (MPU-9250)
  - Smartphone POCKET: accel at ~200 Hz (BOSCH)
  - 208 Fall files (forward, backward, lateral)
  - 538 ADL files (walking, sitting, bending, etc.)
  - File format: semicolon-separated, header lines start with %

Sensor mapping to SisFall:
  UMAFall WAIST accel X/Y/Z -> MMA_x/y/z (already in g)
  UMAFall WAIST gyro  X/Y/Z -> ITG_x/y/z (already in deg/s)

Usage:
    from umafall_loader import load_umafall_dataset
    X, y, meta = load_umafall_dataset('UMAFall_Dataset_corrected_version/')
"""
import os
import re
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

from config import WINDOW_SIZE, STEP_SIZE, CUTOFF_HZ, FS, FILTER_ORDER
from data_loader import butterworth_filter

UMAFALL_FS = 20
SISFALL_FS = FS  # 200 Hz
RESAMPLE_UP = SISFALL_FS // UMAFALL_FS  # 10

WAIST_ID = 2
ACCEL_TYPE = 0
GYRO_TYPE = 1


def _parse_filename(fname: str) -> dict:
    """Parse UMAFall_Subject_02_Fall_forwardFall_1_2016-... -> dict."""
    m = re.match(
        r'UMAFall_(Subject_\d+)_(ADL|Fall)_(.+?)_(\d+)_(\d{4}-\d{2}-\d{2})',
        fname
    )
    if not m:
        return None
    return {
        'subject': m.group(1),
        'activity_type': m.group(2),
        'activity_name': m.group(3),
        'trial': int(m.group(4)),
        'date': m.group(5),
    }


def _read_umafall_csv(filepath: str) -> tuple:
    """Read a single UMAFall CSV, extract WAIST accel+gyro.

    Returns:
        accel: (N, 3) array in g
        gyro:  (N, 3) array in deg/s
        header_info: dict with age, gender, etc.
    """
    header_info = {}
    data_rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%'):
                if 'Age:' in line:
                    m = re.search(r'Age:\s*(\d+)', line)
                    if m:
                        header_info['age'] = int(m.group(1))
                if 'Gender:' in line:
                    header_info['gender'] = line.split(':')[-1].strip()
                continue
            if not line:
                continue
            parts = line.rstrip(';').split(';')
            if len(parts) >= 7:
                try:
                    data_rows.append([float(x) for x in parts[:7]])
                except ValueError:
                    continue

    if not data_rows:
        return None, None, header_info

    data = np.array(data_rows)
    # columns: ts, sample, x, y, z, sensor_type, sensor_id

    # Extract WAIST accel (type=0, id=2)
    mask_acc = (data[:, 5] == ACCEL_TYPE) & (data[:, 6] == WAIST_ID)
    acc = data[mask_acc][:, 2:5] if mask_acc.any() else None

    # Extract WAIST gyro (type=1, id=2)
    mask_gyro = (data[:, 5] == GYRO_TYPE) & (data[:, 6] == WAIST_ID)
    gyro = data[mask_gyro][:, 2:5] if mask_gyro.any() else None

    return acc, gyro, header_info


def _align_and_merge(acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """Align accel and gyro arrays to same length, return (N, 6)."""
    min_len = min(len(acc), len(gyro))
    return np.hstack([acc[:min_len], gyro[:min_len]]).astype(np.float32)


def file_to_windows_umafall(filepath: str,
                             window_size: int = WINDOW_SIZE,
                             step_size: int = STEP_SIZE) -> np.ndarray:
    """Read 1 UMAFall CSV -> sliding windows (N, 400, 6)."""
    acc, gyro, _ = _read_umafall_csv(filepath)

    if acc is None or gyro is None or len(acc) < 3 or len(gyro) < 3:
        return np.empty((0, window_size, 6), dtype=np.float32)

    data = _align_and_merge(acc, gyro)

    # Upsample 20Hz -> 200Hz
    data = resample_poly(data, up=RESAMPLE_UP, down=1, axis=0).astype(np.float32)

    # Butterworth filter
    if len(data) > window_size:
        data = butterworth_filter(data, cutoff=CUTOFF_HZ,
                                  fs=SISFALL_FS, order=FILTER_ORDER).astype(np.float32)

    # Sliding windows
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[start:start + window_size])

    return np.array(windows) if windows else np.empty((0, window_size, 6), dtype=np.float32)


def _refine_fall_windows_avm(windows: np.ndarray,
                              threshold: float = 1.8) -> tuple:
    """Keep Fall windows with AVM peak >= threshold (g)."""
    avm = np.sqrt(np.sum(windows[:, :, :3]**2, axis=2))
    peaks = avm.max(axis=1)
    keep = peaks >= threshold
    return windows[keep], keep


def load_umafall_dataset(umafall_dir: str,
                          apply_label_refinement: bool = True,
                          fall_avm_threshold: float = 1.8,
                          verbose: bool = True) -> tuple:
    """Load UMAFall dataset.

    Returns:
        X:    (N, 400, 6) windows [accX, accY, accZ, gyrX, gyrY, gyrZ]
        y:    (N,) labels 0=ADL, 1=Fall
        meta: list of dicts
    """
    files = sorted([f for f in os.listdir(umafall_dir) if f.endswith('.csv')])

    if verbose:
        print(f"[UMAFall] {len(files)} files in {umafall_dir}")

    all_X, all_y, all_meta = [], [], []
    n_fall_raw, n_fall_kept, n_adl = 0, 0, 0
    skipped = 0

    iterator = tqdm(files, desc="UMAFall files") if verbose else files
    for fname in iterator:
        info = _parse_filename(fname)
        if info is None:
            skipped += 1
            continue

        fpath = os.path.join(umafall_dir, fname)
        try:
            windows = file_to_windows_umafall(fpath)
        except Exception as e:
            if verbose:
                tqdm.write(f"  [skip] {fname}: {e}")
            skipped += 1
            continue

        if len(windows) == 0:
            skipped += 1
            continue

        is_fall = info['activity_type'] == 'Fall'

        if is_fall:
            n_fall_raw += len(windows)
            if apply_label_refinement:
                windows, _ = _refine_fall_windows_avm(windows, fall_avm_threshold)
                if len(windows) == 0:
                    continue
            n_fall_kept += len(windows)
            label = 1
        else:
            n_adl += len(windows)
            label = 0

        all_X.append(windows)
        all_y.append(np.full(len(windows), label, dtype=np.int8))
        for _ in range(len(windows)):
            all_meta.append({
                'subject': info['subject'],
                'activity_code': info['activity_name'],
                'activity_type': info['activity_type'],
                'trial': str(info['trial']),
                'source_file': fname,
            })
        del windows

    if not all_X:
        raise RuntimeError("No valid files loaded from UMAFall.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    if verbose:
        print(f"\n[UMAFall] Total windows: {len(X):,}")
        print(f"  Fall: {(y==1).sum():,} (kept from {n_fall_raw:,} raw)")
        print(f"  ADL:  {(y==0).sum():,}")
        print(f"  Skipped files: {skipped}")
        print(f"  Shape: {X.shape}")
        if apply_label_refinement:
            dropped = n_fall_raw - n_fall_kept
            pct = dropped / max(n_fall_raw, 1) * 100
            print(f"  AVM refinement: dropped {dropped:,} ({pct:.1f}%)")

    return X, y, all_meta


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="UMAFall_Dataset_corrected_version")
    parser.add_argument("--no_refine", action="store_true")
    args = parser.parse_args()

    X, y, meta = load_umafall_dataset(
        args.dir,
        apply_label_refinement=not args.no_refine,
    )
    print(f"\nDone! X.shape={X.shape}, y.shape={y.shape}")
    subjects = set(m['subject'] for m in meta)
    print(f"Subjects: {sorted(subjects)}")

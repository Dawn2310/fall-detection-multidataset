"""
features.py — Feature extraction from time windows.
Each window (window_size, n_channels) -> 1D feature vector.

v2: added jerk (acceleration derivative), SMA (Signal Magnitude Area), tilt angle (from ITG).
Jerk is the strongest fall feature - impact creates a massive derivative spike.
"""
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq
from config import FS


# ─── Per-channel features (time + freq + jerk) ───────────────────────────────
def features_per_channel(window: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Calculate 15 features for a single channel (1D vector of length T).
    Returns array (15,) = 12 classic + 3 jerk.
    """
    # Time domain
    mean   = np.mean(window)
    std    = np.std(window)
    rms    = np.sqrt(np.mean(window**2))
    vmin   = np.min(window)
    vmax   = np.max(window)
    vrange = vmax - vmin
    kurt   = kurtosis(window)
    skewn  = skew(window)

    # Frequency domain (FFT)
    fft_mag = np.abs(rfft(window))
    freqs   = rfftfreq(len(window), d=1.0/fs)
    spec_energy = np.sum(fft_mag**2) / len(fft_mag)
    if fft_mag.sum() > 0:
        spectral_centroid = np.sum(freqs * fft_mag) / fft_mag.sum()
    else:
        spectral_centroid = 0.0
    peak_freq = freqs[np.argmax(fft_mag[1:])+1] if len(fft_mag) > 1 else 0.0
    norm_fft = fft_mag / (fft_mag.sum() + 1e-12)
    spec_entropy = -np.sum(norm_fft * np.log(norm_fft + 1e-12))

    # === NEW: Jerk (Signal derivative) ===
    # Jerk = d/dt(window). A fall creates a very strong impact spike in jerk.
    jerk = np.diff(window) * fs            # multiply by fs to get g/s
    jerk_max = np.max(np.abs(jerk)) if len(jerk) > 0 else 0.0
    jerk_rms = np.sqrt(np.mean(jerk**2))   if len(jerk) > 0 else 0.0
    jerk_std = np.std(jerk)                 if len(jerk) > 0 else 0.0

    return np.array([mean, std, rms, vmin, vmax, vrange,
                     kurt, skewn, spec_energy, spectral_centroid,
                     peak_freq, spec_entropy,
                     jerk_max, jerk_rms, jerk_std], dtype=np.float32)


# ─── AVM + SMA Features ─────────────────────────────────────────────────────
def features_avm(window_3d: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Calculate 9 features from AVM of a 3-axis group.
    window_3d: (T, 3)
    Classic: 6 AVM features. New: + SMA + jerk_AVM_max + tilt_change.
    """
    avm = np.sqrt(np.sum(window_3d**2, axis=1))

    # Classic (6)
    f_max    = np.max(avm)
    f_mean   = np.mean(avm)
    f_std    = np.std(avm)
    f_min    = np.min(avm)
    f_range  = f_max - f_min
    f_peakpos = np.argmax(avm) / len(avm)

    # New: SMA (Signal Magnitude Area) - gold standard in HAR
    # SMA = (1/T) * sum(|x| + |y| + |z|)
    sma = np.mean(np.sum(np.abs(window_3d), axis=1))

    # New: AVM jerk (peak rate-of-change)
    avm_jerk = np.diff(avm) * fs
    avm_jerk_max = np.max(np.abs(avm_jerk)) if len(avm_jerk) > 0 else 0.0

    # New: orientation change (vector angle change from start -> end)
    # = angle between first and last vector
    v_start = window_3d[0]
    v_end   = window_3d[-1]
    norm_start = np.linalg.norm(v_start) + 1e-12
    norm_end   = np.linalg.norm(v_end)   + 1e-12
    cos_ang = np.dot(v_start, v_end) / (norm_start * norm_end)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    tilt_change = np.arccos(cos_ang)  # radian, [0, pi]

    return np.array([f_max, f_mean, f_std, f_min, f_range, f_peakpos,
                     sma, avm_jerk_max, tilt_change], dtype=np.float32)


# ─── Extract features for a single window ───────────────────────────────────────
def extract_features(window: np.ndarray,
                     channel_names: list[str]) -> np.ndarray:
    """
    Extract all features for 1 window (T, C).
    Returns 1D feature vector.

    Structure:
      - 15 features x C channels (time + freq + jerk)
      - 9 AVM features for each 3-axis group (max, mean, std, min, range,
        peakpos, SMA, jerk_max, tilt_change)
    """
    feats = []

    # Per-channel features
    for i in range(window.shape[1]):
        feats.append(features_per_channel(window[:, i]))

    # AVM features per sensor group
    groups = {}
    for i, name in enumerate(channel_names):
        prefix = name.rsplit('_', 1)[0]  # ADXL, MMA, ITG
        groups.setdefault(prefix, []).append(i)

    for prefix, idxs in groups.items():
        if len(idxs) == 3:
            feats.append(features_avm(window[:, idxs]))

    return np.concatenate(feats)


# ─── Extract features for entire dataset ────────────────────────────────
def extract_features_batch(X: np.ndarray,
                            channel_names: list[str],
                            verbose: bool = True) -> np.ndarray:
    """
    X: (N, T, C) -> X_feat: (N, n_features)
    """
    from tqdm import tqdm
    n_samples = X.shape[0]
    sample_feat = extract_features(X[0], channel_names)
    n_feats = len(sample_feat)

    X_feat = np.zeros((n_samples, n_feats), dtype=np.float32)
    iterator = tqdm(range(n_samples), desc="Extract features") if verbose else range(n_samples)
    for i in iterator:
        X_feat[i] = extract_features(X[i], channel_names)

    if verbose:
        print(f"Feature shape: {X_feat.shape} ({n_feats} features/window)")
    return X_feat


# ─── Feature names (for results printing) ─────────────────────────────────────────
def get_feature_names(channel_names: list[str]) -> list[str]:
    """Generate corresponding feature names list."""
    time_freq_names = ['mean','std','rms','min','max','range',
                       'kurtosis','skewness','spec_energy',
                       'spec_centroid','peak_freq','spec_entropy',
                       'jerk_max','jerk_rms','jerk_std']
    names = []
    for ch in channel_names:
        for feat in time_freq_names:
            names.append(f"{ch}_{feat}")

    # AVM groups
    groups = {}
    for name in channel_names:
        prefix = name.rsplit('_', 1)[0]
        groups.setdefault(prefix, []).append(name)

    avm_names = ['avm_max','avm_mean','avm_std','avm_min','avm_range',
                 'avm_peakpos','sma','avm_jerk_max','tilt_change']
    for prefix, cols in groups.items():
        if len(cols) == 3:
            for feat in avm_names:
                names.append(f"{prefix}_{feat}")
    return names

"""
preprocess.py — Run once, cache all processed data to disk.

Output: data/cache/windows_<variant>.npz containing X, y, subjects, activity_codes
        data/cache/features_<variant>.npz containing X_feat, y, feature_names

Usage:
    python src/preprocess.py            # run all 7 variants
    python src/preprocess.py --variant ADXL  # only 1 variant
"""
import os
import sys
import gc
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SENSOR_VARIANTS, BASE_DIR
from windowing import build_window_dataset
from features import extract_features_batch, get_feature_names

CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def preprocess_variant(variant_name: str, force: bool = False) -> dict:
    """Cut windows + extract features for 1 sensor variant, save .npz."""
    sensor_cols = SENSOR_VARIANTS[variant_name]
    win_path  = os.path.join(CACHE_DIR, f"windows_{variant_name}.npz")
    feat_path = os.path.join(CACHE_DIR, f"features_{variant_name}.npz")

    # Skip if already cached
    if not force and os.path.exists(win_path) and os.path.exists(feat_path):
        print(f"[SKIP] {variant_name}: cache already exists")
        return {"win_path": win_path, "feat_path": feat_path, "skipped": True}

    print(f"\n{'='*55}")
    print(f"  Preprocess: {variant_name}  ({len(sensor_cols)} channels)")
    print(f"{'='*55}")

    # 1) Cut windows from entire dataset
    X, y, meta = build_window_dataset(
        subjects=None,           # None = read all SA + SE
        sensor_cols=sensor_cols,
        verbose=True,
    )

    # Separate metadata into arrays to save via numpy
    subjects       = np.array([m["subject"]       for m in meta])
    activity_codes = np.array([m["activity_code"] for m in meta])
    age_groups     = np.array([m["age_group"]     for m in meta])

    print(f"\nSaving windows -> {win_path}")
    np.savez_compressed(
        win_path,
        X=X.astype(np.float32),
        y=y.astype(np.int8),
        subjects=subjects,
        activity_codes=activity_codes,
        age_groups=age_groups,
        sensor_cols=np.array(sensor_cols),
    )

    # 2) Extract features
    print("Extract features...")
    X_feat = extract_features_batch(X, sensor_cols, verbose=True)
    feat_names = get_feature_names(sensor_cols)

    print(f"Saving features -> {feat_path}")
    np.savez_compressed(
        feat_path,
        X_feat=X_feat.astype(np.float32),
        y=y.astype(np.int8),
        subjects=subjects,
        activity_codes=activity_codes,
        age_groups=age_groups,
        feature_names=np.array(feat_names),
    )

    sz_win  = os.path.getsize(win_path)  / 1e6
    sz_feat = os.path.getsize(feat_path) / 1e6
    print(f"Completed {variant_name}: windows {sz_win:.1f} MB | features {sz_feat:.1f} MB")
    return {"win_path": win_path, "feat_path": feat_path, "skipped": False}


def load_cache(variant_name: str, kind: str = "features") -> dict:
    """Load cache. kind = 'features' or 'windows'."""
    path = os.path.join(CACHE_DIR, f"{kind}_{variant_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}\nRun 'python src/preprocess.py' first.")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    parser = argparse.ArgumentParser(description="Preprocess SisFall data and cache to disk.")
    parser.add_argument("--variant", default="all",
                        help="Sensor variant name or 'all' (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing cache, rerun from scratch")
    args = parser.parse_args()

    if args.variant == "all":
        variants = list(SENSOR_VARIANTS.keys())
    else:
        if args.variant not in SENSOR_VARIANTS:
            raise ValueError(f"Variant must be one of: {list(SENSOR_VARIANTS.keys())}")
        variants = [args.variant]

    print(f"\nWill preprocess {len(variants)} variants: {variants}")
    print(f"Cache dir: {CACHE_DIR}\n")

    results = {}
    for v in variants:
        results[v] = preprocess_variant(v, force=args.force)
        gc.collect()  # clear RAM between variants

    # Final summary
    print(f"\n{'='*55}")
    print("  PREPROCESSING COMPLETED")
    print(f"{'='*55}")
    for v, info in results.items():
        status = "SKIP" if info["skipped"] else "DONE"
        print(f"  [{status}] {v:10s} -> {os.path.basename(info['feat_path'])}")


if __name__ == "__main__":
    main()

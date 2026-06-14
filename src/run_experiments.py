"""
run_experiments.py — Main script to run all experiments (RAM + time optimized).

Pipeline:
  1. preprocess_variant() for each sensor variant -> cache .npz
  2. Train RF/SVM/KNN for each variant, evaluate on Test
  3. Cross-age: train SA -> test SE
  4. Plot charts + save CSV

Usage:
    python src/run_experiments.py                # run all
    python src/run_experiments.py --use_dl       # add DL models
    python src/run_experiments.py --variants ADXL ALL9   # only 2 variants
    python src/run_experiments.py --skip_intra   # skip intra-age
    python src/run_experiments.py --skip_cross   # skip cross-age
"""
import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SENSOR_VARIANTS, TRAIN_SUBJECTS, VAL_SUBJECTS,
                    TEST_SA, TEST_SE, METRIC_DIR, FIG_DIR, MODEL_DIR)
from preprocess import preprocess_variant, load_cache
from ml_models import run_all_models, run_cross_age
from evaluate import (plot_sensor_variant_comparison,
                       plot_cross_age_gap,
                       make_results_table)
import progress as pg

# Disable matplotlib interactive mode when running in background (overnight)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # no popup when running headless

os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _split_features_by_subject(d: dict) -> dict:
    """From cache .npz, split into train/val/test_SA/test_SE by subject."""
    X_feat   = d['X_feat']
    y        = d['y']
    subjects = d['subjects']

    masks = {
        'train':   np.isin(subjects, TRAIN_SUBJECTS),
        'val':     np.isin(subjects, VAL_SUBJECTS),
        'test_SA': np.isin(subjects, TEST_SA),
        'test_SE': np.isin(subjects, TEST_SE),
    }
    splits = {}
    for name, mask in masks.items():
        splits[name] = {'X': X_feat[mask], 'y': y[mask]}

    # 'test' = SA22-23 + all SE (according to original config)
    test_mask = masks['test_SA'] | masks['test_SE']
    splits['test'] = {'X': X_feat[test_mask], 'y': y[test_mask]}
    return splits


# ─── Intra-age experiments ───────────────────────────────────────────────────
def run_intra_age(variants: list[str], use_dl: bool = False) -> pd.DataFrame:
    all_results = []
    interim_path = os.path.join(METRIC_DIR, "intra_age_results_interim.csv")

    pg.set_stage("INTRA", var_total=len(variants), mdl_total=3)
    for i, variant_name in enumerate(variants, 1):
        pg.set_variant(variant_name, i)
        pg.banner(f"INTRA-AGE | {variant_name} ({i}/{len(variants)})")

        # 1) Ensure cache exists (preprocess if missing)
        preprocess_variant(variant_name, force=False)

        # 2) Load cache + split
        pg.log("Loading feature cache...")
        d = load_cache(variant_name, kind='features')
        sp = _split_features_by_subject(d)
        del d
        gc.collect()

        pg.log(f"train: {sp['train']['X'].shape} | "
               f"val: {sp['val']['X'].shape} | "
               f"test: {sp['test']['X'].shape}")

        # 3) Train ML models
        ml_res = run_all_models(
            sp['train']['X'], sp['train']['y'],
            sp['val']['X'],   sp['val']['y'],
            sp['test']['X'],  sp['test']['y'],
            experiment_name=f"variant_{variant_name}",
            save_models=True,
        )
        all_results.append(ml_res)

        # 4) DL (optional, needs windows instead of features)
        if use_dl:
            try:
                from preprocess import load_cache as _lc
                from dl_models import run_dl_models
                pg.log("Loading window cache for DL...")
                dw = _lc(variant_name, kind='windows')
                ws_masks = {
                    'train': np.isin(dw['subjects'], TRAIN_SUBJECTS),
                    'val':   np.isin(dw['subjects'], VAL_SUBJECTS),
                    'test':  np.isin(dw['subjects'], TEST_SA + TEST_SE),
                }
                dl_res = run_dl_models(
                    dw['X'][ws_masks['train']], dw['y'][ws_masks['train']],
                    dw['X'][ws_masks['val']],   dw['y'][ws_masks['val']],
                    dw['X'][ws_masks['test']],  dw['y'][ws_masks['test']],
                    experiment_name=f"variant_{variant_name}",
                )
                all_results.append(dl_res)
                del dw, ws_masks
            except Exception as e:
                pg.log(f"[DL skip {variant_name}] {e}")

        # 5) Save interim, clear RAM
        pd.concat(all_results, ignore_index=True).to_csv(interim_path, index=False)
        pg.log(f"[interim] saved {interim_path}")
        del sp, ml_res
        gc.collect()

    return pd.concat(all_results, ignore_index=True)


# ─── Cross-age experiments ───────────────────────────────────────────────────
def run_cross_age_experiments(variants: list[str]) -> pd.DataFrame:
    all_SA = set(TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SA)
    all_SE = set(TEST_SE)
    cross_results = []
    interim_path = os.path.join(METRIC_DIR, "cross_age_results_interim.csv")

    pg.set_stage("CROSS", var_total=len(variants), mdl_total=3)
    for i, variant_name in enumerate(variants, 1):
        pg.set_variant(variant_name, i)
        pg.banner(f"CROSS-AGE | {variant_name} ({i}/{len(variants)})", char="=")

        preprocess_variant(variant_name, force=False)
        pg.log("Loading feature cache...")
        d = load_cache(variant_name, kind='features')
        subjects = d['subjects']

        mask_sa = np.array([s in all_SA for s in subjects])
        mask_se = np.array([s in all_SE for s in subjects])
        X_sa, y_sa = d['X_feat'][mask_sa], d['y'][mask_sa]
        X_se, y_se = d['X_feat'][mask_se], d['y'][mask_se]
        del d
        gc.collect()

        # Split SA further into train+val to tune threshold (90/10 random by subject possible,
        # but for simplicity: use stratified 90/10 random split)
        from sklearn.model_selection import train_test_split
        try:
            X_sa_tr, X_sa_va, y_sa_tr, y_sa_va = train_test_split(
                X_sa, y_sa, test_size=0.1, stratify=y_sa, random_state=42)
        except ValueError:
            X_sa_tr, X_sa_va, y_sa_tr, y_sa_va = X_sa, X_sa[:0], y_sa, y_sa[:0]
        res = run_cross_age(X_sa_tr, y_sa_tr, X_se, y_se,
                             sensor_variant=variant_name,
                             X_val=X_sa_va, y_val=y_sa_va)
        cross_results.append(res)

        pd.concat(cross_results, ignore_index=True).to_csv(interim_path, index=False)
        pg.log(f"[interim] saved {interim_path}")
        del X_sa, y_sa, X_se, y_se, res
        gc.collect()

    return pd.concat(cross_results, ignore_index=True)


# ─── Main ────────────────────────────────────────────────────────────────────
def main(args):
    pg.start()
    variants = args.variants or list(SENSOR_VARIANTS.keys())
    pg.log(f"Running {len(variants)} variants: {variants}")
    pg.log(f"use_dl={args.use_dl} | skip_intra={args.skip_intra} | skip_cross={args.skip_cross}")

    intra_results = None
    cross_results = None

    # ── Experiment 1: Intra-age ───────────────────────────────────────────
    if not args.skip_intra:
        pg.banner("[1/2] INTRA-AGE BASELINE")
        intra_results = run_intra_age(variants, use_dl=args.use_dl)
        intra_path = os.path.join(METRIC_DIR, "intra_age_results.csv")
        intra_results.to_csv(intra_path, index=False)
        pg.log(f"Saved: {intra_path}")

    # ── Experiment 2: Cross-age ───────────────────────────────────────────
    if not args.skip_cross:
        pg.banner("[2/2] CROSS-AGE EVALUATION")
        cross_results = run_cross_age_experiments(variants)
        cross_path = os.path.join(METRIC_DIR, "cross_age_results.csv")
        cross_results.to_csv(cross_path, index=False)
        pg.log(f"Saved: {cross_path}")

    # ── Plotting and Summary ──────────────────────────────────────────────────
    pg.banner("PLOT & SUMMARY")
    try:
        if intra_results is not None:
            plot_sensor_variant_comparison(
                intra_results[intra_results['split'] == 'Test'],
                metric='f1', split='Test',
                save_name="fig_sensor_comparison_F1.png",
            )
            make_results_table(intra_results, "table_intra_age.csv")

        if cross_results is not None:
            make_results_table(cross_results, "table_cross_age.csv")

        if intra_results is not None and cross_results is not None:
            plot_cross_age_gap(
                intra_results[intra_results['split'] == 'Test'],
                cross_results,
                save_name="fig_cross_age_gap.png",
            )
    except Exception as e:
        pg.log(f"[plot skip] {e}")

    pg.banner("COMPLETED! Results saved in: results/")
    return intra_results, cross_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dl', action='store_true',
                        help='Add DL models (CNN-LSTM)')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Only run specific variants, e.g.: --variants ADXL ALL9')
    parser.add_argument('--skip_intra', action='store_true')
    parser.add_argument('--skip_cross', action='store_true')
    args = parser.parse_args()
    main(args)

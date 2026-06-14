"""
run_ablation.py — Ablation study: measure impact of each optimization step.

Uses importlib.reload to force config changes to propagate through the module chain.
"""
import sys
import os
import gc
import time
import importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import config
from features import extract_features_batch, features_per_channel, features_avm
from evaluate import event_level_metrics
from ml_models import compute_metrics

os.makedirs(config.METRIC_DIR, exist_ok=True)

VARIANT = 'ADXL'
CH_NAMES = ['ADXL_x', 'ADXL_y', 'ADXL_z']


def load_sisfall_splits(cutoff_hz=15, apply_avm=True):
    """Load SisFall with specific config by reloading modules."""
    # Patch config BEFORE reloading dependent modules
    config.CUTOFF_HZ = cutoff_hz
    config.USE_LABEL_REFINEMENT = apply_avm

    # Reload data_loader (picks up new CUTOFF_HZ default)
    import data_loader
    importlib.reload(data_loader)

    # Reload windowing (picks up new USE_LABEL_REFINEMENT)
    import windowing
    importlib.reload(windowing)

    channels = config.SENSOR_VARIANTS[VARIANT]
    splits = {}
    for name, subjects in [('train', config.TRAIN_SUBJECTS),
                           ('val', config.VAL_SUBJECTS),
                           ('test', config.TEST_SA + config.TEST_SE)]:
        X, y, meta = windowing.build_window_dataset(
            subjects=subjects, sensor_cols=channels, verbose=False)
        splits[name] = (X, y, meta)
    return splits


def extract_feats(X):
    parts = []
    BS = 3000
    for i in range(0, len(X), BS):
        parts.append(extract_features_batch(X[i:i+BS], CH_NAMES, verbose=False))
    return np.concatenate(parts)


def extract_basic_feats(X):
    """12 classic per-channel + 6 classic AVM (no jerk/SMA/tilt)."""
    n = len(X)
    n_feats = 12 * 3 + 6 * 1
    out = np.zeros((n, n_feats), dtype=np.float32)
    for i in range(n):
        f = []
        for c in range(3):
            full = features_per_channel(X[i, :, c])
            f.append(full[:12])
        avm_full = features_avm(X[i, :, :3])
        f.append(avm_full[:6])
        out[i] = np.concatenate(f)
    return out


def get_meta_arrays(meta):
    subj = np.array([m['subject'] for m in meta])
    act = np.array([m['activity_code'] for m in meta])
    # NOTE: trials collapsed to 0 to match the event-grouping used in Table 5
    # (compute_event_metrics.py) — 1 event = (subject, activity_code).
    tri = np.zeros(len(meta), dtype=int)
    return subj, act, tri


def train_eval(model_cls, model_kwargs, X_tr, y_tr, X_val, y_val,
               X_te, y_te, meta_te, fixed_threshold=None):
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('clf', model_cls(**model_kwargs))])
    pipe.fit(X_tr, y_tr)

    if fixed_threshold is not None:
        best_thr = fixed_threshold
    else:
        probs_val = pipe.predict_proba(X_val)[:, 1]
        best_thr, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.96, 0.05):
            yp = (probs_val >= t).astype(int)
            m = compute_metrics(y_val, yp, probs_val)
            if m['f1'] > best_f1:
                best_f1 = m['f1']
                best_thr = round(t, 2)

    probs_te = pipe.predict_proba(X_te)[:, 1]
    y_pred = (probs_te >= best_thr).astype(int)
    m_win = compute_metrics(y_te, y_pred, probs_te)
    subj, act, tri = get_meta_arrays(meta_te)
    m_ev = event_level_metrics(y_te, probs_te, subj, act, tri,
                               threshold=best_thr, agg='mean')
    return m_win, m_ev, best_thr


RF_KWARGS = dict(n_estimators=200, max_depth=20, min_samples_split=5,
                 min_samples_leaf=2, class_weight='balanced',
                 random_state=42, n_jobs=-1)
KNN_KWARGS = dict(n_neighbors=5, weights='distance', n_jobs=-1)

MODELS = [
    ('RF', RandomForestClassifier, RF_KWARGS),
    ('KNN', KNeighborsClassifier, KNN_KWARGS),
]


def run_condition(label, desc, splits, feat_fn, fixed_thr=None, rows=None):
    X_tr = feat_fn(splits['train'][0])
    X_val = feat_fn(splits['val'][0])
    X_te = feat_fn(splits['test'][0])
    y_tr, y_val, y_te = splits['train'][1], splits['val'][1], splits['test'][1]
    meta_te = splits['test'][2]

    n_fall_tr = int((y_tr == 1).sum())
    n_adl_tr = int((y_tr == 0).sum())
    print(f"  Data: train={len(y_tr)} ({n_fall_tr}F+{n_adl_tr}A) "
          f"val={len(y_val)} test={len(y_te)} feats={X_tr.shape[1]}", flush=True)

    if rows is None:
        rows = []
    for name, cls, kwargs in MODELS:
        mw, me, thr = train_eval(cls, kwargs, X_tr, y_tr, X_val, y_val,
                                 X_te, y_te, meta_te, fixed_threshold=fixed_thr)
        print(f"  {name:<4s} w_F1={mw['f1']*100:6.2f}  e_F1={me['f1']*100:6.2f}  "
              f"e_Sens={me['sensitivity']*100:5.1f}%  e_Spec={me['specificity']*100:5.1f}%  "
              f"thr={thr:.2f}", flush=True)
        rows.append(dict(condition=label, desc=desc, model=name, threshold=thr,
                         w_f1=mw['f1'], w_acc=mw['accuracy'],
                         w_sens=mw['sensitivity'], w_spec=mw['specificity'],
                         e_f1=me['f1'], e_acc=me['accuracy'],
                         e_sens=me['sensitivity'], e_spec=me['specificity'],
                         e_tp=me['tp'], e_fn=me['fn'],
                         e_fp=me['fp'], e_tn=me['tn']))


if __name__ == '__main__':
    print("=" * 70)
    print("  ABLATION STUDY — ADXL + RF/KNN")
    print("=" * 70)

    rows = []

    # A: Full pipeline
    print("\n[A] Full pipeline (baseline)...", flush=True)
    t0 = time.time()
    splits_full = load_sisfall_splits(cutoff_hz=15, apply_avm=True)
    run_condition('A_full', 'Full pipeline', splits_full, extract_feats, rows=rows)
    print(f"    ({time.time()-t0:.0f}s)")

    # B: Cutoff 5Hz
    print("\n[B] Cutoff=5Hz...", flush=True)
    t0 = time.time()
    splits_b = load_sisfall_splits(cutoff_hz=5, apply_avm=True)
    run_condition('B_cutoff5', 'Cutoff 5Hz (not 15Hz)', splits_b, extract_feats, rows=rows)
    print(f"    ({time.time()-t0:.0f}s)")
    del splits_b; gc.collect()

    # C: No AVM refinement
    print("\n[C] No AVM refinement...", flush=True)
    t0 = time.time()
    splits_c = load_sisfall_splits(cutoff_hz=15, apply_avm=False)
    run_condition('C_no_avm', 'No AVM label refinement', splits_c, extract_feats, rows=rows)
    print(f"    ({time.time()-t0:.0f}s)")
    del splits_c; gc.collect()

    # D: Basic features (no jerk/SMA/tilt)
    print("\n[D] Basic features only (no jerk/SMA/tilt)...", flush=True)
    t0 = time.time()
    run_condition('D_basic_feats', 'No jerk/SMA/tilt features', splits_full, extract_basic_feats, rows=rows)
    print(f"    ({time.time()-t0:.0f}s)")

    # E: Default threshold 0.5
    print("\n[E] Default threshold=0.5...", flush=True)
    t0 = time.time()
    run_condition('E_thr05', 'Default threshold 0.5', splits_full, extract_feats, fixed_thr=0.5, rows=rows)
    print(f"    ({time.time()-t0:.0f}s)")

    del splits_full; gc.collect()

    # Restore config
    config.CUTOFF_HZ = 15
    config.USE_LABEL_REFINEMENT = True

    # Save
    df = pd.DataFrame(rows)
    out = os.path.join(config.METRIC_DIR, 'ablation_study.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary
    print("\n" + "=" * 90)
    print("  ABLATION SUMMARY — ADXL variant")
    print("=" * 90)
    header = (f"{'Condition':<35s} {'Model':<5s} {'w_F1':>7s} {'e_F1':>7s} "
              f"{'e_Sens':>7s} {'e_Spec':>7s} {'de_F1':>8s}")
    print(header)
    print("-" * 90)

    base = {}
    for r in rows:
        if r['condition'] == 'A_full':
            base[r['model']] = r['e_f1']

    for r in rows:
        delta = (r['e_f1'] - base[r['model']]) * 100
        sign = '+' if delta >= 0 else ''
        print(f"{r['desc']:<35s} {r['model']:<5s} {r['w_f1']*100:7.2f} "
              f"{r['e_f1']*100:7.2f} {r['e_sens']*100:7.1f} {r['e_spec']*100:7.1f} "
              f"{sign}{delta:7.2f}")

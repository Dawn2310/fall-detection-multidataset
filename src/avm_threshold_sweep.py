"""
avm_threshold_sweep.py — Sensitivity analysis of the AVM label-refinement
threshold (reviewer request). For the ADXL variant on SisFall, sweep the
threshold and measure, at each value:
  - RF event-level F1 / sensitivity / specificity on the held-out test set
  - how many genuine fall recordings are *entirely* discarded (i.e. every
    window falls below the threshold) — the "missed low-impact falls" risk.

Run:  python src/avm_threshold_sweep.py
"""
import sys, os, gc, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import config
from features import extract_features_batch
from evaluate import event_level_metrics
from ml_models import compute_metrics

VARIANT = 'ADXL'
CH_NAMES = ['ADXL_x', 'ADXL_y', 'ADXL_z']
RF_KWARGS = dict(n_estimators=300, class_weight='balanced',
                 random_state=42, n_jobs=-1)


def count_fall_files_dropped(threshold):
    """Count how many Fall recordings in the TEST set lose ALL windows."""
    from data_loader import butterworth_filter, load_raw_file, DATA_DIR
    from windowing import extract_windows_from_series
    chans = config.SENSOR_VARIANTS[VARIANT]
    test_subs = set(config.TEST_SA + config.TEST_SE)
    n_fall_files = 0
    n_fall_dropped = 0
    for subj in sorted(os.listdir(DATA_DIR)):
        if subj not in test_subs:
            continue
        subj_path = os.path.join(DATA_DIR, subj)
        if not os.path.isdir(subj_path):
            continue
        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith('.txt') or not fname.startswith('F'):
                continue
            n_fall_files += 1
            df = load_raw_file(os.path.join(subj_path, fname))
            data = butterworth_filter(df[chans].values.astype(np.float32)).astype(np.float32)
            w = extract_windows_from_series(data, config.WINDOW_SIZE, config.STEP_SIZE)
            if len(w) == 0:
                n_fall_dropped += 1
                continue
            avm = np.sqrt(np.sum(w[:, :, :3] ** 2, axis=2)).max(axis=1)
            if (avm >= threshold).sum() == 0:
                n_fall_dropped += 1
    return n_fall_files, n_fall_dropped


def load_splits(threshold):
    """Rebuild ADXL windows with a given AVM threshold (None = no refinement)."""
    config.USE_LABEL_REFINEMENT = threshold is not None
    config.FALL_AVM_THRESHOLD = threshold if threshold is not None else 0.0
    import data_loader, windowing
    importlib.reload(data_loader)
    importlib.reload(windowing)
    chans = config.SENSOR_VARIANTS[VARIANT]
    out = {}
    for name, subs in [('train', config.TRAIN_SUBJECTS),
                       ('val', config.VAL_SUBJECTS),
                       ('test', config.TEST_SA + config.TEST_SE)]:
        X, y, meta = windowing.build_window_dataset(
            subjects=subs, sensor_cols=chans, verbose=False)
        out[name] = (X, y, meta)
    return out


def feats(X):
    parts = []
    for i in range(0, len(X), 3000):
        parts.append(extract_features_batch(X[i:i+3000], CH_NAMES, verbose=False))
    return np.concatenate(parts)


def evaluate(threshold):
    sp = load_splits(threshold)
    Xtr, ytr = feats(sp['train'][0]), sp['train'][1]
    Xva, yva = feats(sp['val'][0]),   sp['val'][1]
    Xte, yte, mte = feats(sp['test'][0]), sp['test'][1], sp['test'][2]
    n_fall_tr = int((ytr == 1).sum())

    pipe = Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(**RF_KWARGS))])
    pipe.fit(Xtr, ytr)

    pv = pipe.predict_proba(Xva)[:, 1]
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.05):
        m = compute_metrics(yva, (pv >= t).astype(int), pv)
        if m['f1'] > best_f1:
            best_f1, best_t = m['f1'], round(t, 2)

    pt = pipe.predict_proba(Xte)[:, 1]
    subj = np.array([m['subject'] for m in mte])
    act = np.array([m['activity_code'] for m in mte])
    tri = np.zeros(len(mte), dtype=int)   # match Table-5 event grouping
    me = event_level_metrics(yte, pt, subj, act, tri, threshold=best_t, agg='mean')
    return n_fall_tr, best_t, me


if __name__ == '__main__':
    thresholds = [None, 1.5, 1.8, 2.0, 2.2, 2.5]
    rows = []
    print("=" * 78)
    print("  AVM THRESHOLD SENSITIVITY — ADXL + RF (SisFall, event-level test)")
    print("=" * 78)
    for thr in thresholds:
        label = "no-refine" if thr is None else f"{thr:.1f}g"
        print(f"\n[threshold = {label}]", flush=True)
        n_fall_tr, best_t, me = evaluate(thr)
        # count fully-dropped fall recordings in the TEST set
        if thr is None:
            n_files, n_dropped = 0, 0
        else:
            n_files, n_dropped = count_fall_files_dropped(thr)
        rows.append(dict(
            threshold=label, train_fall_windows=n_fall_tr, dec_thr=best_t,
            e_f1=round(me['f1']*100, 2), e_sens=round(me['sensitivity']*100, 2),
            e_spec=round(me['specificity']*100, 2),
            e_tp=me['tp'], e_fp=me['fp'], e_fn=me['fn'],
            test_fall_files=n_files, test_fall_files_lost=n_dropped))
        print(f"  train fall windows={n_fall_tr} | dec_thr={best_t} | "
              f"e_F1={me['f1']*100:.2f} Sens={me['sensitivity']*100:.1f} "
              f"Spec={me['specificity']*100:.2f} | "
              f"fall recordings lost={n_dropped}/{n_files}", flush=True)
        gc.collect()

    df = pd.DataFrame(rows)
    out = os.path.join(config.METRIC_DIR, 'avm_threshold_sweep.csv')
    df.to_csv(out, index=False)
    print("\n" + "=" * 78)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")

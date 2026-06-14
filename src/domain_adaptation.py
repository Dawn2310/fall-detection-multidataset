"""
domain_adaptation.py — Few-shot elderly personalization for the cross-age gap.

Motivation
----------
The cross-age experiment showed a large performance drop when a model trained on
young adults (SA) is applied to elderly subjects (SE). In SisFall, *only one*
elderly subject (SE06) recorded falls (15 fall types x 5 trials), so a
leave-elderly-out fall evaluation is impossible. Instead we study the realistic
**personalization / few-shot domain-adaptation** scenario: a deployed device
collects a small amount of calibration data from the elderly user, and we ask
whether injecting that data closes the gap.

Design
------
Source domain (young) : SA01-SA23 (all young adults)
Target few-shot       : SE06 trials {R01, R02}  (calibration data, 2 of 5 trials)
Test sets             :
   (a) SE06 trials {R03, R04, R05}  -> held-out personalized falls + ADL
   (b) all other SE (SE01-05, SE07-15) ADL  -> cross-subject elderly specificity

Conditions (per sensor variant)
   A. Zero-shot     : RF trained on SA only            (baseline = current paper)
   B. Few-shot DA   : RF trained on SA + SE06{R01,R02}  (data-augmentation adapt)
   C. Target-only   : RF trained on SE06{R01,R02} only  (shows few-shot alone fails)

Operating point: for every condition the decision threshold is calibrated on the
SE06 adaptation set (the only elderly calibration data realistically available),
maximizing Youden's J. We also report threshold-free window-level ROC-AUC.

Usage
-----
    python src/domain_adaptation.py
    python src/domain_adaptation.py --variants ADXL MMA_ITG
"""
import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SENSOR_VARIANTS, METRIC_DIR, RANDOM_STATE)
from windowing import build_window_dataset
from features import extract_features_batch, get_feature_names
from evaluate import event_level_metrics

ALL_SA   = [f"SA{i:02d}" for i in range(1, 24)]
FALLER   = "SE06"
OTHER_SE = [f"SE{i:02d}" for i in range(1, 16) if f"SE{i:02d}" != FALLER]
ADAPT_TRIALS = {"R01", "R02"}
TEST_TRIALS  = {"R03", "R04", "R05"}


def _build_features(subjects, sensor_cols, verbose=False):
    """Build windows for given subjects and return (feat, y, meta)."""
    X, y, meta = build_window_dataset(
        subjects=subjects, sensor_cols=sensor_cols, verbose=verbose)
    names = get_feature_names(sensor_cols)
    # Extract in batches to bound memory.
    batch = 5000
    feats = []
    for i in range(0, len(X), batch):
        feats.append(extract_features_batch(X[i:i+batch], sensor_cols, verbose=False))
    feat = np.concatenate(feats, axis=0) if feats else np.empty((0, len(names)))
    del X, feats
    gc.collect()
    return feat, y, meta, names


def _meta_arrays(meta):
    subj = np.array([m['subject'] for m in meta])
    act  = np.array([m['activity_code'] for m in meta])
    trl  = np.array([m['trial'] for m in meta])
    return subj, act, trl


def _train_rf(Xf, y):
    """Train a standardized RF pipeline (balanced)."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2,
            class_weight='balanced', n_jobs=2, random_state=RANDOM_STATE)),
    ])
    pipe.fit(Xf, y)
    return pipe


def _youden_threshold(y_true, y_prob):
    """Pick threshold maximizing sensitivity+specificity-1 (Youden's J)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def _evaluate(pipe, Xf, y, meta, threshold):
    """Return event-level metrics + window-AUC for a feature set."""
    if len(Xf) == 0:
        return None
    prob = pipe.predict_proba(Xf)[:, 1]
    subj, act, trl = _meta_arrays(meta)
    ev = event_level_metrics(y, prob, subj, act, trl,
                             threshold=threshold, agg='max')
    try:
        ev['w_auc'] = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float('nan')
    except ValueError:
        ev['w_auc'] = float('nan')
    return ev


def run_variant(variant, sensor_cols, verbose=True):
    if verbose:
        print(f"\n{'='*70}\n  VARIANT: {variant}  ({len(sensor_cols)} channels)\n{'='*70}")

    # --- Build feature sets ---
    Xf_sa,  y_sa,  meta_sa,  names = _build_features(ALL_SA, sensor_cols)
    Xf_se6, y_se6, meta_se6, _     = _build_features([FALLER], sensor_cols)
    Xf_oth, y_oth, meta_oth, _     = _build_features(OTHER_SE, sensor_cols)

    # --- Split SE06 by trial ---
    _, _, trl6 = _meta_arrays(meta_se6)
    adapt_mask = np.isin(trl6, list(ADAPT_TRIALS))
    test_mask  = np.isin(trl6, list(TEST_TRIALS))

    Xf_adapt, y_adapt = Xf_se6[adapt_mask], y_se6[adapt_mask]
    [meta_se6[i] for i in np.where(adapt_mask)[0]]
    Xf_t6, y_t6 = Xf_se6[test_mask], y_se6[test_mask]
    meta_t6 = [meta_se6[i] for i in np.where(test_mask)[0]]

    if verbose:
        print(f"  SA(source)       : {len(y_sa):6,} win  ({(y_sa==1).sum()} F / {(y_sa==0).sum()} A)")
        print(f"  SE06 adapt {{R01,R02}}: {len(y_adapt):5,} win  ({(y_adapt==1).sum()} F / {(y_adapt==0).sum()} A)")
        print(f"  SE06 test  {{R03-05}}: {len(y_t6):5,} win  ({(y_t6==1).sum()} F / {(y_t6==0).sum()} A)")
        print(f"  other-SE ADL     : {len(y_oth):6,} win  ({(y_oth==1).sum()} F / {(y_oth==0).sum()} A)")

    conditions = {
        'A_zero_shot':   (Xf_sa, y_sa),
        'B_fewshot_DA':  (np.vstack([Xf_sa, Xf_adapt]), np.concatenate([y_sa, y_adapt])),
        'C_target_only': (Xf_adapt, y_adapt),
    }

    rows = []
    for cond, (Xtr, ytr) in conditions.items():
        if len(np.unique(ytr)) < 2:
            if verbose:
                print(f"  [skip {cond}] training set single-class")
            continue
        pipe = _train_rf(Xtr, ytr)

        # Calibrate threshold on the SE06 adaptation set (available elderly calib data)
        prob_adapt = pipe.predict_proba(Xf_adapt)[:, 1]
        thr = _youden_threshold(y_adapt, prob_adapt)

        ev6  = _evaluate(pipe, Xf_t6,  y_t6,  meta_t6,  thr)   # personalized held-out
        evoth = _evaluate(pipe, Xf_oth, y_oth, meta_oth, thr)  # cross-subject ADL

        # Combined elderly test (held-out SE06 + other-SE ADL)
        Xf_all = np.vstack([Xf_t6, Xf_oth])
        y_all  = np.concatenate([y_t6, y_oth])
        meta_all = meta_t6 + meta_oth
        evall = _evaluate(pipe, Xf_all, y_all, meta_all, thr)

        if verbose:
            print(f"\n  [{cond}] thr={thr:.3f}")
            for tag, ev in [('SE06_heldout', ev6), ('otherSE_ADL', evoth),
                            ('ALL_elderly', evall)]:
                if ev is None:
                    continue
                print(f"    {tag:13s}: e_F1={ev['f1']*100:6.2f} Sens={ev['sensitivity']*100:6.2f} "
                      f"Spec={ev['specificity']*100:6.2f} AUC={ev['w_auc']:.3f} "
                      f"(TP{ev['tp']} FN{ev['fn']} FP{ev['fp']} TN{ev['tn']})")

        for tag, ev in [('SE06_heldout', ev6), ('otherSE_ADL', evoth),
                        ('ALL_elderly', evall)]:
            if ev is None:
                continue
            rows.append({
                'variant': variant, 'condition': cond, 'test_set': tag,
                'threshold': thr,
                'e_f1': ev['f1'], 'e_sensitivity': ev['sensitivity'],
                'e_specificity': ev['specificity'], 'e_precision': ev['precision'],
                'e_accuracy': ev['accuracy'], 'w_auc': ev['w_auc'],
                'tp': ev['tp'], 'fn': ev['fn'], 'fp': ev['fp'], 'tn': ev['tn'],
                'n_events': ev['n_events'],
            })
        del pipe
        gc.collect()

    del Xf_sa, Xf_se6, Xf_oth
    gc.collect()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=['ADXL', 'MMA_ITG'],
                    help="Sensor variants to evaluate")
    args = ap.parse_args()

    print(f"\n{'#'*70}")
    print("#  DOMAIN ADAPTATION — Few-Shot Elderly Personalization (cross-age)")
    print(f"#  Source: SA01-23 | Few-shot: {FALLER}{{R01,R02}} | Test: {FALLER}{{R03-05}} + other-SE")
    print(f"{'#'*70}")

    all_rows = []
    for v in args.variants:
        if v not in SENSOR_VARIANTS:
            print(f"[skip] unknown variant {v}")
            continue
        all_rows += run_variant(v, SENSOR_VARIANTS[v], verbose=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(METRIC_DIR, exist_ok=True)
    out = os.path.join(METRIC_DIR, 'domain_adaptation.csv')
    df.to_csv(out, index=False)
    print(f"\n>>> Saved: {out} ({len(df)} rows)")

    # Summary: ALL_elderly test, the headline comparison
    print(f"\n{'='*78}")
    print("  SUMMARY — ALL_elderly test set (SE06 held-out falls + all other-SE ADL)")
    print(f"{'='*78}")
    sub = df[df['test_set'] == 'ALL_elderly'].copy()
    if len(sub):
        disp = sub[['variant', 'condition', 'e_f1', 'e_sensitivity',
                    'e_specificity', 'w_auc']].copy()
        for c in ['e_f1', 'e_sensitivity', 'e_specificity']:
            disp[c] = (disp[c] * 100).round(2)
        disp['w_auc'] = disp['w_auc'].round(4)
        print(disp.to_string(index=False))


if __name__ == '__main__':
    main()

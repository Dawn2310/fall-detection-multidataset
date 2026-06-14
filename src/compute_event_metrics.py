"""
compute_event_metrics.py — Compute event-level metrics for ALL 70 rows
of intra-age results (5 models x 7 variants x 2 splits = Val + Test).

Usage:
    python src/compute_event_metrics.py

Output:
    results/metrics/intra_age_results_event.csv   (~70 rows, both window-level and event-level)
    results/metrics/table_intra_age_event.csv     (paper-ready Test only, x100)
"""
import os
import sys
import gc
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SENSOR_VARIANTS, VAL_SUBJECTS,
                    TEST_SA, TEST_SE, METRIC_DIR, MODEL_DIR)
from preprocess import load_cache
from evaluate import event_level_metrics
from dl_models import CNNLSTM, CNNBiLSTMAttention, _get_device

ML_MODELS = ['RF', 'SVM', 'KNN']
DL_MODELS = {'cnn_lstm': CNNLSTM, 'cnn_bilstm_attention': CNNBiLSTMAttention}


def _load_test_data(variant: str, kind: str = 'features'):
    """Load val + test data (windows or features) + meta for 1 variant."""
    d = load_cache(variant, kind=kind)
    X_key = 'X_feat' if kind == 'features' else 'X'
    X = d[X_key]
    y = d['y']
    subjects = d['subjects'].astype(str)
    activity_codes = d['activity_codes'].astype(str)

    val_mask  = np.isin(subjects, VAL_SUBJECTS)
    test_mask = np.isin(subjects, list(TEST_SA) + list(TEST_SE))
    return {
        'val':  (X[val_mask],  y[val_mask],  subjects[val_mask],  activity_codes[val_mask]),
        'test': (X[test_mask], y[test_mask], subjects[test_mask], activity_codes[test_mask]),
    }


def _predict_ml(pkl_path: str, X: np.ndarray) -> np.ndarray:
    pipe = joblib.load(pkl_path)
    return pipe.predict_proba(X)[:, 1]


def _predict_dl(ckpt_path: str, X: np.ndarray, arch_class) -> tuple:
    """Return (probs, threshold saved in checkpoint)."""
    device = _get_device()
    n_ch = X.shape[-1]
    model = arch_class(n_channels=n_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    threshold = float(ckpt.get('best_threshold', 0.5))

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i+256].astype(np.float32)).to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs), threshold


def _compute_all_metrics(probs, y, subjects, act_codes, threshold):
    """Compute window-level + event-level (mean agg)."""
    from ml_models import compute_metrics
    y_pred = (probs >= threshold).astype(int)
    m_win = compute_metrics(y, y_pred, probs)
    trials = np.zeros(len(subjects), dtype=int)
    m_ev = event_level_metrics(y, probs, subjects, act_codes, trials,
                                threshold=threshold, agg='mean')
    return m_win, m_ev


def _row(experiment, model, split, threshold, m_win, m_ev):
    return {
        'experiment': experiment, 'model': model, 'split': split.capitalize(),
        'threshold': threshold,
        'w_accuracy': m_win['accuracy'],   'w_sensitivity': m_win['sensitivity'],
        'w_specificity': m_win['specificity'], 'w_precision': m_win['precision'],
        'w_f1': m_win['f1'], 'w_auc': m_win.get('auc', float('nan')),
        'e_accuracy': m_ev['accuracy'],   'e_sensitivity': m_ev['sensitivity'],
        'e_specificity': m_ev['specificity'], 'e_precision': m_ev['precision'],
        'e_f1': m_ev['f1'], 'e_n_events': m_ev['n_events'],
        'e_tp': m_ev['tp'], 'e_tn': m_ev['tn'],
        'e_fp': m_ev['fp'], 'e_fn': m_ev['fn'],
    }


def main():
    rows = []
    variants = list(SENSOR_VARIANTS.keys())
    print(f"Computing event-level for {len(variants)} variants x 5 models...\n")

    for vi, variant in enumerate(variants, 1):
        print(f"[{vi}/{len(variants)}] {variant}")
        try:
            ft = _load_test_data(variant, kind='features')
            wn = _load_test_data(variant, kind='windows')
        except FileNotFoundError as e:
            print(f"  Skip {variant}: {e}")
            continue

        # ML models - use pre-saved threshold from current run (default 0.5)
        # Note: original run tuned threshold on Val before saving. Because pkl
        # does not save threshold, we re-tune on Val and apply to Test (same as original logic)
        for model_name in ML_MODELS:
            pkl_path = os.path.join(MODEL_DIR, f"{model_name}_variant_{variant}.pkl")
            if not os.path.exists(pkl_path):
                print(f"  Skip {model_name}: missing {os.path.basename(pkl_path)}")
                continue

            # Tune threshold on Val
            X_val, y_val, subj_val, act_val = ft['val']
            probs_val = _predict_ml(pkl_path, X_val)
            from ml_models import compute_metrics
            best_thr, best_f1 = 0.5, -1.0
            for t in np.linspace(0.05, 0.95, 19):
                yp = (probs_val >= t).astype(int)
                m = compute_metrics(y_val, yp, probs_val)
                if m['f1'] > best_f1:
                    best_f1, best_thr = m['f1'], float(t)

            # Val metrics
            m_win, m_ev = _compute_all_metrics(probs_val, y_val, subj_val, act_val, best_thr)
            rows.append(_row(f'variant_{variant}', model_name, 'Val', best_thr, m_win, m_ev))

            # Test metrics
            X_te, y_te, subj_te, act_te = ft['test']
            probs_te = _predict_ml(pkl_path, X_te)
            m_win, m_ev = _compute_all_metrics(probs_te, y_te, subj_te, act_te, best_thr)
            rows.append(_row(f'variant_{variant}', model_name, 'Test', best_thr, m_win, m_ev))
            print(f"  {model_name}: thr={best_thr:.2f} | Test w_f1={m_win['f1']*100:.2f} e_f1={m_ev['f1']*100:.2f} e_acc={m_ev['accuracy']*100:.2f} e_sens={m_ev['sensitivity']*100:.2f}")

        # DL models
        for arch_name, arch_class in DL_MODELS.items():
            ckpt_path = os.path.join(MODEL_DIR, f"{arch_name}_variant_{variant}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  Skip {arch_name}: missing")
                continue

            # Val
            X_val, y_val, subj_val, act_val = wn['val']
            probs_val, thr = _predict_dl(ckpt_path, X_val, arch_class)
            m_win, m_ev = _compute_all_metrics(probs_val, y_val, subj_val, act_val, thr)
            rows.append(_row(f'variant_{variant}', arch_name, 'Val', thr, m_win, m_ev))

            # Test
            X_te, y_te, subj_te, act_te = wn['test']
            probs_te, _ = _predict_dl(ckpt_path, X_te, arch_class)
            m_win, m_ev = _compute_all_metrics(probs_te, y_te, subj_te, act_te, thr)
            rows.append(_row(f'variant_{variant}', arch_name, 'Test', thr, m_win, m_ev))
            print(f"  {arch_name}: thr={thr:.2f} | Test w_f1={m_win['f1']*100:.2f} e_f1={m_ev['f1']*100:.2f} e_acc={m_ev['accuracy']*100:.2f} e_sens={m_ev['sensitivity']*100:.2f}")

        del ft, wn
        gc.collect()

    df = pd.DataFrame(rows)
    os.makedirs(METRIC_DIR, exist_ok=True)

    raw_path = os.path.join(METRIC_DIR, 'intra_age_results_event.csv')
    df.to_csv(raw_path, index=False)
    print(f"\n>>> Saved raw: {raw_path} ({len(df)} rows)")

    # Paper-ready: x100, round 2, Test only
    paper = df[df['split'] == 'Test'].copy()
    for c in paper.columns:
        if (c.startswith('w_') or c.startswith('e_')) and c not in ('e_n_events','e_tp','e_tn','e_fp','e_fn'):
            paper[c] = (paper[c] * 100).round(2)
    table_path = os.path.join(METRIC_DIR, 'table_intra_age_event.csv')
    paper.to_csv(table_path, index=False)
    print(f">>> Saved paper-ready: {table_path}")

    # Top 10 by event-level F1
    print("\n=== TOP 10 by Event-level F1 (Test) ===")
    top = paper.sort_values('e_f1', ascending=False).head(10)
    cols = ['experiment','model','w_f1','e_accuracy','e_sensitivity','e_specificity','e_f1','e_n_events']
    print(top[cols].to_string(index=False))


if __name__ == '__main__':
    main()

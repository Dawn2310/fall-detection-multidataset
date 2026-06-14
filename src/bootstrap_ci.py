"""
bootstrap_ci.py — Event-level bootstrap 95% confidence intervals for the
headline configurations (reviewer request: small fall-event counts demand CIs).

Resampling is done at the EVENT level (not the window level) so that the
overlapping-window dependence does not inflate precision. For each config we
aggregate window probabilities to events, then resample events with
replacement B times and recompute F1 / sensitivity / specificity.

Run:  python src/bootstrap_ci.py
"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import config
from features import extract_features_batch
from cross_dataset import (predict_ml, predict_dl, _map_kfall_to_variant,
                           VARIANT_CHANNELS)

RNG = np.random.default_rng(42)
B = 5000


def events_from_windows(y, probs, subj, act, trials, threshold):
    """Aggregate windows -> events (mean prob). Returns (y_true, y_pred) arrays."""
    df = pd.DataFrame({'s': subj, 'a': act, 't': trials, 'y': y, 'p': probs})
    ev = df.groupby(['s', 'a', 't']).agg(yt=('y', 'max'), pp=('p', 'mean'))
    yt = ev['yt'].to_numpy().astype(int)
    yp = (ev['pp'].to_numpy() >= threshold).astype(int)
    return yt, yp


def _metrics(yt, yp):
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) else 0.0
    sens = tp / (tp + fn) if (tp+fn) else 0.0
    spec = tn / (tn + fp) if (tn+fp) else 0.0
    return f1*100, sens*100, spec*100


def bootstrap(yt, yp, b=B):
    n = len(yt)
    idx = np.arange(n)
    f1s, ses, sps = [], [], []
    point = _metrics(yt, yp)
    for _ in range(b):
        s = RNG.choice(idx, size=n, replace=True)
        f, se, sp = _metrics(yt[s], yp[s])
        f1s.append(f); ses.append(se); sps.append(sp)
    def ci(a):
        return np.percentile(a, 2.5), np.percentile(a, 97.5)
    return point, ci(f1s), ci(ses), ci(sps)


def sisfall_adxl_rf():
    """Intra-dataset headline: ADXL + RF on the SisFall test set."""
    cache = os.path.join(config.BASE_DIR, 'data', 'cache', 'features_ADXL.npz')
    d = np.load(cache, allow_pickle=True)
    X, y, subj, act = d['X_feat'], d['y'], d['subjects'], d['activity_codes']
    test = np.isin(subj, config.TEST_SA + config.TEST_SE)
    Xte, yte, ste, ate = X[test], y[test], subj[test], act[test]
    probs, _ = predict_ml('RF', 'ADXL', Xte)
    trials = np.zeros(len(yte), dtype=int)            # match Table-5 grouping
    yt, yp = events_from_windows(yte, probs, ste, ate, trials, 0.50)
    return ('SisFall', 'ADXL', 'RF', len(yt), int(yt.sum()), yt, yp)


def cross(target, loader_kwargs, variant, model):
    """Cross-dataset headline configs."""
    from cross_dataset import load_target
    X_raw, y, meta = load_target(target, **loader_kwargs)
    Xv = _map_kfall_to_variant(X_raw, variant)
    is_dl = model in ('cnn_lstm', 'cnn_bilstm_attention')
    if is_dl:
        probs, thr = predict_dl(model, variant, Xv)
    else:
        chn = VARIANT_CHANNELS.get(variant, [f'ch{i}' for i in range(Xv.shape[-1])])
        feats = []
        for i in range(0, len(Xv), 5000):
            feats.append(extract_features_batch(Xv[i:i+5000], chn, verbose=False))
        Xf = np.concatenate(feats)
        probs, thr = predict_ml(model, variant, Xf)
    subj = np.array([m['subject'] for m in meta])
    actc = np.array([m['activity_code'] for m in meta])
    tri = np.array([m.get('trial', 0) for m in meta], dtype=int)
    yt, yp = events_from_windows(y, probs, subj, actc, tri, thr)
    del X_raw; gc.collect()
    return (target, variant, model, len(yt), int(yt.sum()), yt, yp)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    configs = []
    print("Computing event predictions...", flush=True)
    configs.append(sisfall_adxl_rf()); print("  SisFall done", flush=True)
    configs.append(cross('kfall', {'kfall_dir': 'K-Fall dataset'}, 'ADXL_ITG', 'KNN')); print("  KFall done", flush=True)
    configs.append(cross('umafall', {'umafall_dir': 'UMAFall_Dataset_corrected_version'}, 'MMA_ITG', 'RF')); print("  UMAFall done", flush=True)
    configs.append(cross('upfall', {'upfall_dir': 'UP-Fall dataset'}, 'MMA_ITG', 'RF')); print("  UP-Fall done", flush=True)

    rows = []
    print("\n" + "="*92)
    print(f"{'Dataset':9s} {'Variant':9s} {'Model':5s} {'Nev':>5s} {'Fall':>5s} "
          f"{'F1 [95% CI]':>22s} {'Sens [95% CI]':>22s} {'Spec [95% CI]':>22s}")
    print("="*92)
    for tgt, var, mdl, nev, nfall, yt, yp in configs:
        (f1, se, sp), cf, cs, csp = bootstrap(yt, yp)
        rows.append(dict(dataset=tgt, variant=var, model=mdl, n_events=nev,
                         n_fall=nfall,
                         f1=round(f1,2), f1_lo=round(cf[0],2), f1_hi=round(cf[1],2),
                         sens=round(se,2), sens_lo=round(cs[0],2), sens_hi=round(cs[1],2),
                         spec=round(sp,2), spec_lo=round(csp[0],2), spec_hi=round(csp[1],2)))
        print(f"{tgt:9s} {var:9s} {mdl:5s} {nev:5d} {nfall:5d} "
              f"{f1:6.2f} [{cf[0]:5.2f},{cf[1]:6.2f}] "
              f"{se:6.2f} [{cs[0]:5.2f},{cs[1]:6.2f}] "
              f"{sp:6.2f} [{csp[0]:5.2f},{csp[1]:6.2f}]")

    out = os.path.join(config.METRIC_DIR, 'bootstrap_ci.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

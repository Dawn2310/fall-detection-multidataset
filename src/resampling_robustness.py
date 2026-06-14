"""
resampling_robustness.py — Reviewer 3.6: does the 20 Hz -> 200 Hz upsampling
used for UMAFall/UP-Fall create artifacts that distort results / unfairly
favour ML over DL?

We isolate the resampling variable on SisFall, where ground truth is known.
For each ADXL test window (native 200 Hz) we apply the SAME low-rate pipeline
the external datasets undergo: decimate 200 -> 20 Hz (anti-aliased), then
polyphase upsample 20 -> 200 Hz. We then re-extract the 54 features and
evaluate the unchanged, SisFall-trained models, comparing native vs round-trip.

Run:  python src/resampling_robustness.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

import config
from features import extract_features_batch
from evaluate import event_level_metrics
from cross_dataset import predict_ml, predict_dl

CH = ['ADXL_x', 'ADXL_y', 'ADXL_z']


def roundtrip(windows):
    """200 Hz -> 20 Hz (decimate) -> 200 Hz (upsample), per window, per channel."""
    n, T, c = windows.shape
    down = resample_poly(windows, up=1, down=10, axis=1)   # ->20 Hz (T/10)
    up = resample_poly(down, up=10, down=1, axis=1)         # ->200 Hz
    # restore exact length T
    if up.shape[1] != T:
        if up.shape[1] > T:
            up = up[:, :T, :]
        else:
            up = np.pad(up, ((0, 0), (0, T - up.shape[1]), (0, 0)), mode='edge')
    return up.astype(np.float32)


def feats(X):
    out = []
    for i in range(0, len(X), 3000):
        out.append(extract_features_batch(X[i:i+3000], CH, verbose=False))
    return np.concatenate(out)


def eval_set(Xwin, y, subj, act, model, is_dl, thr_override=None):
    if is_dl:
        probs, thr = predict_dl(model, 'ADXL', Xwin)
    else:
        probs, thr = predict_ml(model, 'ADXL', feats(Xwin))
    if thr_override is not None:
        thr = thr_override
    trials = np.zeros(len(y), dtype=int)
    me = event_level_metrics(y, probs, subj, act, trials, threshold=thr, agg='mean')
    return me, thr


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    cache = os.path.join(config.BASE_DIR, 'data', 'cache', 'windows_ADXL.npz')
    d = np.load(cache, allow_pickle=True)
    Xw, y, subj, act = d['X'], d['y'], d['subjects'], d['activity_codes']
    test = np.isin(subj, config.TEST_SA + config.TEST_SE)
    Xt, yt, st, at = Xw[test], y[test], subj[test], act[test]
    print(f"Test windows: {len(yt)} | building round-tripped copy...", flush=True)
    Xt_rt = roundtrip(Xt)

    models = [('RF', False), ('KNN', False),
              ('cnn_lstm', True), ('cnn_bilstm_attention', True)]
    rows = []
    print("\n" + "=" * 80)
    print(f"{'Model':22s} {'native e_F1':>12s} {'roundtrip e_F1':>15s} {'Δ':>8s}")
    print("=" * 80)
    for model, is_dl in models:
        me_n, thr = eval_set(Xt, yt, st, at, model, is_dl)
        me_r, _ = eval_set(Xt_rt, yt, st, at, model, is_dl, thr_override=thr)
        d_f1 = (me_r['f1'] - me_n['f1']) * 100
        rows.append(dict(model=model,
                         native_e_f1=round(me_n['f1']*100, 2),
                         native_sens=round(me_n['sensitivity']*100, 2),
                         roundtrip_e_f1=round(me_r['f1']*100, 2),
                         roundtrip_sens=round(me_r['sensitivity']*100, 2),
                         delta_f1=round(d_f1, 2)))
        print(f"{model:22s} {me_n['f1']*100:12.2f} {me_r['f1']*100:15.2f} {d_f1:+8.2f}")

    out = os.path.join(config.METRIC_DIR, 'resampling_robustness.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

"""
cross_dataset.py — Zero-shot cross-dataset evaluation.

Pipeline:
  1. Load best SisFall model (trained on SisFall Train set)
  2. Load target dataset (KFall) via respective loader
  3. Extract features (for ML) or keep windows (for DL)
  4. Predict zero-shot (NO fine-tuning)
  5. Compute window-level + event-level metrics
  6. Save results to CSV

Usage:
    python src/cross_dataset.py --target kfall --kfall_dir "K-Fall dataset"
    python src/cross_dataset.py --target kfall --model RF --variant MMA_ITG
    python src/cross_dataset.py --target kfall --model all
"""
import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import METRIC_DIR, MODEL_DIR, SENSOR_VARIANTS
from features import extract_features_batch
from evaluate import event_level_metrics
from ml_models import compute_metrics
from dl_models import CNNLSTM, CNNBiLSTMAttention, _get_device

ML_MODELS = ['RF', 'SVM', 'KNN']
DL_MODELS = {'cnn_lstm': CNNLSTM, 'cnn_bilstm_attention': CNNBiLSTMAttention}
ALL_MODELS = ML_MODELS + list(DL_MODELS.keys())

VARIANT_CHANNELS = SENSOR_VARIANTS


# ─── Load target dataset ─────────────────────────────────────────────────────
def load_target(target_name: str, **kwargs):
    if target_name == 'kfall':
        from kfall_loader import load_kfall_dataset
        return load_kfall_dataset(**kwargs)
    elif target_name == 'umafall':
        from umafall_loader import load_umafall_dataset
        return load_umafall_dataset(**kwargs)
    elif target_name == 'upfall':
        from upfall_loader import load_upfall_dataset
        return load_upfall_dataset(**kwargs)
    else:
        raise ValueError(f"Target not supported: {target_name}")


def _map_kfall_to_variant(X_6ch: np.ndarray, variant: str) -> np.ndarray:
    """Map KFall 6 channels [AccX,AccY,AccZ,GyrX,GyrY,GyrZ] to SisFall variant.

    KFall has 6 channels: 3 accel (~ MMA) + 3 gyro (~ ITG)
    -> Usable variants: MMA_ITG (6ch), MMA (3ch), ITG (3ch)
    -> Accel-only variants: ADXL (3ch) ~ use KFall accel as proxy
    -> 9ch variants (ALL9): not enough channels -> skip or duplicate
    """
    if variant in ('MMA_ITG', 'ADXL_ITG', 'ADXL_MMA'):
        return X_6ch  # 6 channels as-is
    elif variant in ('MMA', 'ADXL'):
        return X_6ch[:, :, :3]  # accel only (3ch)
    elif variant == 'ITG':
        return X_6ch[:, :, 3:]  # gyro only (3ch)
    elif variant == 'ALL9':
        # Duplicate accel to fill 9ch: [acc, acc, gyro]
        acc = X_6ch[:, :, :3]
        gyro = X_6ch[:, :, 3:]
        return np.concatenate([acc, acc, gyro], axis=2)
    else:
        raise ValueError(f"Variant not supported: {variant}")


# ─── Predict ─────────────────────────────────────────────────────────────────
def predict_ml(model_name: str, variant: str,
                X_feat: np.ndarray) -> tuple[np.ndarray, float]:
    pkl_path = os.path.join(MODEL_DIR, f"{model_name}_variant_{variant}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"No model {pkl_path}")
    pipe = joblib.load(pkl_path)

    # Try to get tuned threshold from companion file
    thr_path = os.path.join(MODEL_DIR, f"{model_name}_variant_{variant}_threshold.txt")
    threshold = 0.5
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            threshold = float(f.read().strip())

    # Predict in batches to avoid memory issues
    batch_size = 5000
    probs = []
    for i in range(0, len(X_feat), batch_size):
        probs.append(pipe.predict_proba(X_feat[i:i+batch_size])[:, 1])
    return np.concatenate(probs), threshold


def predict_dl(arch_name: str, variant: str,
                X_win: np.ndarray) -> tuple[np.ndarray, float]:
    ckpt_path = os.path.join(MODEL_DIR, f"{arch_name}_variant_{variant}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint {ckpt_path}")

    device = _get_device()
    n_ch = X_win.shape[-1]
    arch_class = DL_MODELS[arch_name]
    model = arch_class(n_channels=n_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    thr = float(ckpt.get('best_threshold', 0.5))

    batch_size = 128
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_win), batch_size):
            xb = torch.from_numpy(X_win[i:i+batch_size].astype(np.float32)).to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
            del xb
    torch.cuda.empty_cache()
    del model, ckpt
    torch.cuda.empty_cache()
    return np.concatenate(probs), thr


# ─── Evaluate 1 model+variant ────────────────────────────────────────────────
def run_zero_shot(target_name: str,
                   variant: str,
                   model_choice: str,
                   X_raw: np.ndarray,
                   y: np.ndarray,
                   meta: list,
                   verbose: bool = True) -> dict | None:
    """
    Zero-shot predict on pre-loaded target data.
    X_raw: (N, 400, 6) — KFall raw 6 channels
    """
    # Map channels (same logic for both KFall and UMAFall — both 6ch accel+gyro)
    try:
        X = _map_kfall_to_variant(X_raw, variant)
    except ValueError as e:
        if verbose:
            print(f"  Skip {variant}: {e}")
        return None

    # Check model exists
    is_dl = model_choice in DL_MODELS
    if is_dl:
        ckpt_path = os.path.join(MODEL_DIR, f"{model_choice}_variant_{variant}.pt")
        if not os.path.exists(ckpt_path):
            if verbose:
                print(f"  Skip {model_choice}+{variant}: no checkpoint")
            return None
    else:
        pkl_path = os.path.join(MODEL_DIR, f"{model_choice}_variant_{variant}.pkl")
        if not os.path.exists(pkl_path):
            if verbose:
                print(f"  Skip {model_choice}+{variant}: no model")
            return None

    # Predict
    try:
        if is_dl:
            probs, threshold = predict_dl(model_choice, variant, X)
        else:
            ch_names = VARIANT_CHANNELS.get(variant, [f"ch{i}" for i in range(X.shape[-1])])
            # Extract features in batches to save memory
            batch = 5000
            feats = []
            for i in range(0, len(X), batch):
                feats.append(extract_features_batch(X[i:i+batch], ch_names, verbose=False))
            X_feat = np.concatenate(feats, axis=0)
            del feats
            probs, threshold = predict_ml(model_choice, variant, X_feat)
            del X_feat
    except Exception as e:
        if verbose:
            print(f"  Error {model_choice}+{variant}: {e}")
        return None

    # Metrics
    subj = np.array([m['subject'] for m in meta])
    act = np.array([m['activity_code'] for m in meta])
    trials = np.array([m.get('trial', 0) for m in meta], dtype=int)

    y_pred = (probs >= threshold).astype(int)
    m_win = compute_metrics(y, y_pred, probs)

    # Event-level: group by (subject, activity, trial)
    m_ev = event_level_metrics(y, probs, subj, act, trials,
                                threshold=threshold, agg='mean')

    if verbose:
        print(f"  {model_choice:25s} + {variant:10s} | thr={threshold:.2f} | "
              f"w_F1={m_win['f1']*100:6.2f} | e_F1={m_ev['f1']*100:6.2f} "
              f"e_Sens={m_ev['sensitivity']*100:6.2f} "
              f"({m_ev['tp']}TP {m_ev['fn']}FN {m_ev['fp']}FP)")

    return {
        'target': target_name,
        'variant': variant,
        'model': model_choice,
        'threshold': threshold,
        'window': m_win,
        'event': m_ev,
    }


# ─── Run all combinations ────────────────────────────────────────────────────
def run_all_zero_shot(target_name: str,
                       X_raw: np.ndarray,
                       y: np.ndarray,
                       meta: list,
                       models: list[str] | None = None,
                       variants: list[str] | None = None) -> pd.DataFrame:
    """Run zero-shot for multiple model+variant combinations."""
    if models is None:
        models = ALL_MODELS
    if variants is None:
        variants = list(SENSOR_VARIANTS.keys())

    print(f"\n{'='*70}")
    print(f"  ZERO-SHOT: SisFall -> {target_name.upper()}")
    print(f"  {len(models)} models x {len(variants)} variants = {len(models)*len(variants)} combinations")
    print(f"  Target: {len(y):,} windows ({(y==1).sum():,} Fall + {(y==0).sum():,} ADL)")
    print(f"{'='*70}\n")

    rows = []
    for variant in variants:
        print(f"[{variant}]")
        for model_choice in models:
            result = run_zero_shot(
                target_name, variant, model_choice,
                X_raw, y, meta, verbose=True
            )
            gc.collect()
            if result is None:
                continue
            rows.append({
                'target': result['target'],
                'variant': result['variant'],
                'model': result['model'],
                'threshold': result['threshold'],
                'w_accuracy': result['window']['accuracy'],
                'w_sensitivity': result['window']['sensitivity'],
                'w_specificity': result['window']['specificity'],
                'w_precision': result['window']['precision'],
                'w_f1': result['window']['f1'],
                'w_auc': result['window'].get('auc', float('nan')),
                'e_accuracy': result['event']['accuracy'],
                'e_sensitivity': result['event']['sensitivity'],
                'e_specificity': result['event']['specificity'],
                'e_precision': result['event']['precision'],
                'e_f1': result['event']['f1'],
                'e_n_events': result['event']['n_events'],
                'e_tp': result['event']['tp'],
                'e_tn': result['event']['tn'],
                'e_fp': result['event']['fp'],
                'e_fn': result['event']['fn'],
            })
        print()

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Zero-shot cross-dataset evaluation.")
    parser.add_argument('--target', default='kfall', choices=['kfall', 'umafall', 'upfall'])
    parser.add_argument('--variant', default='all',
                        help="Sensor variant or 'all'")
    parser.add_argument('--model', default='all',
                        help="Model name or 'all'")
    parser.add_argument('--kfall_dir', default='K-Fall dataset')
    parser.add_argument('--umafall_dir', default='UMAFall_Dataset_corrected_version')
    parser.add_argument('--upfall_dir', default='UP-Fall dataset')
    parser.add_argument('--refinement', default='label',
                        choices=['label', 'avm', 'none'])
    args = parser.parse_args()

    # Load target dataset once
    print(f"Loading {args.target.upper()} dataset...")
    if args.target == 'kfall':
        target_kwargs = {
            'kfall_dir': args.kfall_dir,
            'apply_label_refinement': (args.refinement != 'none'),
            'refinement_method': args.refinement,
        }
    elif args.target == 'umafall':
        target_kwargs = {
            'umafall_dir': args.umafall_dir,
            'apply_label_refinement': (args.refinement != 'none'),
        }
    elif args.target == 'upfall':
        target_kwargs = {
            'upfall_dir': args.upfall_dir,
            'apply_label_refinement': (args.refinement != 'none'),
        }
    X_raw, y, meta = load_target(args.target, **target_kwargs)

    # Parse model/variant
    models = ALL_MODELS if args.model == 'all' else [args.model]
    variants = list(SENSOR_VARIANTS.keys()) if args.variant == 'all' else [args.variant]

    # Run
    df = run_all_zero_shot(args.target, X_raw, y, meta,
                            models=models, variants=variants)

    # Save
    os.makedirs(METRIC_DIR, exist_ok=True)
    out_path = os.path.join(METRIC_DIR, f'cross_dataset_{args.target}.csv')
    df.to_csv(out_path, index=False)
    print(f"\n>>> Saved: {out_path} ({len(df)} rows)")

    # Print top results
    if len(df) > 0:
        print(f"\n{'='*70}")
        print(f"  TOP 10 by Event-Level F1 ({args.target.upper()})")
        print(f"{'='*70}")
        top = df.sort_values('e_f1', ascending=False).head(10)
        cols = ['variant', 'model', 'threshold', 'w_f1', 'e_accuracy',
                'e_sensitivity', 'e_specificity', 'e_f1', 'e_n_events']
        # Format percentages
        display = top[cols].copy()
        for c in ['w_f1', 'e_accuracy', 'e_sensitivity', 'e_specificity', 'e_f1']:
            display[c] = (display[c] * 100).round(2)
        print(display.to_string(index=False))


if __name__ == '__main__':
    main()

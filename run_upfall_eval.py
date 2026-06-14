"""Quick script to run UP-Fall cross-dataset evaluation."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from upfall_loader import load_upfall_dataset
from cross_dataset import run_all_zero_shot, ALL_MODELS
from config import METRIC_DIR, SENSOR_VARIANTS
import pandas as pd
import numpy as np

print("=" * 60, flush=True)
print("Loading UP-Fall dataset...", flush=True)

X_raw, y, meta = load_upfall_dataset(
    'UP-Fall dataset',
    use_tag_labels=False,
    apply_label_refinement=True,
)
print(f"Loaded: {X_raw.shape}, Falls={int((y==1).sum())}, ADL={int((y==0).sum())}", flush=True)

# Run all models and variants
models_to_run = ['RF', 'SVM', 'KNN', 'cnn_lstm', 'cnn_bilstm_attention']
variants_to_run = ['ADXL', 'MMA', 'ITG', 'ADXL_ITG', 'MMA_ITG', 'ADXL_MMA', 'ALL9']

print(f"\nRunning models: {models_to_run}", flush=True)
print(f"Variants: {variants_to_run}", flush=True)

df = run_all_zero_shot(
    'upfall', X_raw, y, meta,
    models=models_to_run,
    variants=variants_to_run,
)

# Save results
os.makedirs(METRIC_DIR, exist_ok=True)
out_path = os.path.join(METRIC_DIR, 'cross_dataset_upfall.csv')
df.to_csv(out_path, index=False)
print(f"\n>>> Saved: {out_path} ({len(df)} rows)", flush=True)

if len(df) > 0:
    print("\n" + "=" * 60, flush=True)
    print("  RESULTS: SisFall -> UP-Fall (Zero-Shot)", flush=True)
    print("=" * 60, flush=True)
    cols = ['variant', 'model', 'threshold', 'w_f1', 'e_accuracy',
            'e_sensitivity', 'e_specificity', 'e_f1', 'e_n_events']
    display = df[cols].copy()
    for c in ['w_f1', 'e_accuracy', 'e_sensitivity', 'e_specificity', 'e_f1']:
        display[c] = (display[c] * 100).round(2)
    print(display.to_string(index=False), flush=True)

print("\nDone!", flush=True)

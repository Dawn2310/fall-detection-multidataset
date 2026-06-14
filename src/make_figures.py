"""
make_figures.py — Generate publication-quality figures for the paper.

Outputs (results/figures/, both .pdf for LaTeX and .png for preview):
  1. fig_feature_importance   — Top-20 RF feature importances (ADXL)
  2. fig_sensor_ablation      — Event-level F1 by sensor variant x model (intra Test)
  3. fig_cross_dataset        — Best event-F1 across SisFall / KFall / UMAFall
  4. fig_domain_adaptation    — Cross-age few-shot DA
  5. fig_pipeline             — Methodology pipeline flowchart
  6. fig_signal_example       — Raw Fall vs ADL accelerometer waveform
  7. fig_avm_distribution     — AVM peak histogram with 1.8g threshold
  8. fig_confusion_matrix     — Confusion matrix for best model (ADXL+RF)
  9. fig_dl_training          — DL training curves (loss + accuracy)
 10. fig_tsne                 — t-SNE feature space visualization
 11. fig_roc_curves           — ROC curves for all 5 models on ADXL

Usage:
    python src/make_figures.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (METRIC_DIR, FIG_DIR, BASE_DIR, ADXL_FACTOR, MMA_FACTOR, ITG_FACTOR,
                    FS, CUTOFF_HZ, FALL_AVM_THRESHOLD, FILTER_ORDER)

os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.bbox': 'tight',
})
C_ML  = '#2c7fb8'   # ML blue
C_DL  = '#de2d26'   # DL red
PAL   = ['#2c7fb8', '#7fcdbb', '#fdae61', '#d7191c', '#756bb1']


def _save(fig, name):
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


# ── 1. Feature importance ─────────────────────────────────────────────────────
def fig_feature_importance():
    f = os.path.join(METRIC_DIR, 'feature_importance_RF_ADXL.csv')
    if not os.path.exists(f):
        print("  [skip] feature_importance_RF_ADXL.csv not found")
        return
    df = pd.read_csv(f).sort_values('importance', ascending=False).head(20)
    df = df.iloc[::-1]  # ascending for barh

    def _cat_color(name):
        if 'jerk' in name:  return '#fdae61'   # jerk
        if any(k in name for k in ('AVM', 'SMA', 'tilt')): return '#2c7fb8'  # proposed
        if 'spec' in name or 'freq' in name: return '#7fcdbb'  # frequency
        return '#bdbdbd'  # classic time-domain

    colors = [_cat_color(n) for n in df['feature']]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.barh(range(len(df)), df['importance'] * 100, color=colors, edgecolor='black', lw=0.4)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'], fontsize=8.5)
    ax.set_xlabel('Gini Importance (%)')
    ax.set_title('Top-20 Feature Importances (Random Forest, ADXL)')
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#2c7fb8', label='Proposed (AVM/SMA/tilt)'),
              Patch(facecolor='#fdae61', label='Jerk'),
              Patch(facecolor='#7fcdbb', label='Frequency-domain'),
              Patch(facecolor='#bdbdbd', label='Classic time-domain')]
    ax.legend(handles=legend, loc='lower right', framealpha=0.95)
    _save(fig, 'fig_feature_importance')


# ── 2. Sensor ablation (event-F1 by variant x model, Test split) ──────────────
def fig_sensor_ablation():
    f = os.path.join(METRIC_DIR, 'intra_age_results_event.csv')
    if not os.path.exists(f):
        print("  [skip] intra_age_results_event.csv not found")
        return
    df = pd.read_csv(f)
    df = df[df['split'] == 'Test'].copy()
    df['variant'] = df['experiment'].str.replace('variant_', '', regex=False)

    model_order = ['RF', 'SVM', 'KNN', 'cnn_lstm', 'cnn_bilstm_attention']
    model_label = {'RF': 'RF', 'SVM': 'SVM', 'KNN': 'KNN',
                   'cnn_lstm': 'CNN-LSTM', 'cnn_bilstm_attention': 'BiLSTM-Att'}
    var_order = ['ADXL', 'MMA', 'ITG', 'ADXL_ITG', 'MMA_ITG', 'ADXL_MMA', 'ALL9']
    var_order = [v for v in var_order if v in df['variant'].unique()]

    pivot = df.pivot_table(index='variant', columns='model', values='e_f1')
    pivot = pivot.reindex(var_order)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(var_order))
    w = 0.16
    for i, m in enumerate(model_order):
        if m not in pivot.columns:
            continue
        vals = pivot[m].values * 100
        ax.bar(x + (i - 2) * w, vals, w, label=model_label[m], color=PAL[i],
               edgecolor='black', lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '+') for v in var_order], rotation=15)
    ax.set_ylabel('Event-level F1 (%)')
    ax.set_ylim(60, 101)
    ax.set_xlabel('Sensor Variant (channels)')
    ax.set_title('Sensor Ablation: Event-level F1 on SisFall Test Set')
    ax.axhline(98.90, ls='--', color='gray', lw=1, alpha=0.7)
    ax.text(len(var_order) - 0.5, 99.1, 'ADXL+RF = 98.90', fontsize=8,
            ha='right', color='gray')
    ax.legend(ncol=5, loc='lower center', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)
    _save(fig, 'fig_sensor_ablation')


# ── 3. Cross-dataset comparison ───────────────────────────────────────────────
def fig_cross_dataset():
    rows = []
    # SisFall intra (best per model from event Test)
    fi = os.path.join(METRIC_DIR, 'intra_age_results_event.csv')
    if os.path.exists(fi):
        d = pd.read_csv(fi)
        d = d[d['split'] == 'Test']
        for m in ['RF', 'SVM', 'KNN']:
            best = d[d['model'] == m]['e_f1'].max()
            rows.append(('SisFall\n(intra)', m, best * 100))
    # KFall + UMAFall + UP-Fall (best per model)
    for ds, fn in [('KFall\n(cross)', 'cross_dataset_kfall.csv'),
                   ('UMAFall\n(cross)', 'cross_dataset_umafall.csv'),
                   ('UP-Fall\n(cross)', 'cross_dataset_upfall.csv')]:
        fp = os.path.join(METRIC_DIR, fn)
        if not os.path.exists(fp):
            continue
        d = pd.read_csv(fp)
        for m in ['RF', 'SVM', 'KNN']:
            sub = d[d['model'] == m]
            if len(sub):
                rows.append((ds, m, sub['e_f1'].max() * 100))

    if not rows:
        print("  [skip] no cross-dataset data")
        return
    df = pd.DataFrame(rows, columns=['dataset', 'model', 'f1'])
    datasets = ['SisFall\n(intra)', 'KFall\n(cross)', 'UMAFall\n(cross)',
                'UP-Fall\n(cross)']
    datasets = [d for d in datasets if d in df['dataset'].unique()]
    models = ['RF', 'SVM', 'KNN']

    fig, ax = plt.subplots(figsize=(8.8, 5))
    x = np.arange(len(datasets))
    w = 0.25
    for i, m in enumerate(models):
        vals = [df[(df['dataset'] == d) & (df['model'] == m)]['f1'].max()
                for d in datasets]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m, color=PAL[i],
                      edgecolor='black', lw=0.3)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Best Event-level F1 (%)')
    ax.set_ylim(60, 103)
    ax.set_title('Cross-Dataset Generalization (best ML per dataset)')
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)
    _save(fig, 'fig_cross_dataset')


# ── 4. Domain adaptation (cross-age few-shot) ─────────────────────────────────
def fig_domain_adaptation():
    f = os.path.join(METRIC_DIR, 'domain_adaptation.csv')
    if not os.path.exists(f):
        print("  [skip] domain_adaptation.csv not found (run domain_adaptation.py)")
        return
    df = pd.read_csv(f)
    df = df[df['test_set'] == 'ALL_elderly'].copy()
    if df.empty:
        print("  [skip] no ALL_elderly rows in domain_adaptation.csv")
        return

    cond_order = ['A_zero_shot', 'B_fewshot_DA', 'C_target_only']
    cond_label = {'A_zero_shot': 'Zero-shot\n(baseline)',
                  'B_fewshot_DA': 'Few-shot DA\n(SA+SE06)',
                  'C_target_only': 'Target-only\n(SE06)'}
    variants = sorted(df['variant'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in [(axes[0], 'e_f1', 'Event-level F1'),
                              (axes[1], 'w_auc', 'Window-level ROC-AUC')]:
        x = np.arange(len(cond_order))
        w = 0.35
        for i, v in enumerate(variants):
            vals = []
            for c in cond_order:
                sub = df[(df['variant'] == v) & (df['condition'] == c)]
                vals.append(sub[metric].values[0] * (100 if metric == 'e_f1' else 1)
                            if len(sub) else np.nan)
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=v.replace('_', '+'),
                          color=PAL[i], edgecolor='black', lw=0.3)
            ax.bar_label(bars, fmt='%.1f' if metric == 'e_f1' else '%.3f',
                         fontsize=8, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels([cond_label[c] for c in cond_order])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.legend(title='Variant')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Few-Shot Domain Adaptation on Elderly (cross-age gap)',
                 fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig_domain_adaptation')


# ── 5. Methodology pipeline flowchart ────────────────────────────────────────
def fig_pipeline():
    """Premium Q1-style system architecture: headered cards, soft shadows,
    phase brackets, tensor-shape annotations, and cross-dataset branch."""
    fig, ax = plt.subplots(figsize=(13.6, 7.2))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.6)
    ax.axis('off')

    # (fill, header/edge) per phase — modern muted palette
    PAL = {
        'in':   ('#eceff1', '#607d8b'),
        'pre':  ('#dbe9f6', '#3f78b0'),
        'feat': ('#dcefd6', '#4f9145'),
        'model':('#e6ddf2', '#7a5ba6'),
        'eval': ('#fce7d2', '#d2873a'),
        'out':  ('#f9dcd9', '#c0504d'),
        'cd':   ('#fff5d6', '#caa520'),
    }
    arrow = dict(arrowstyle='-|>', color='#455a64', lw=1.8, mutation_scale=18)

    def card(cx, cy, title, detail, key, w=2.05, h=1.2):
        fill, edge = PAL[key]
        # soft drop shadow
        sh = FancyBboxPatch((cx - w/2 + 0.05, cy - h/2 - 0.07), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor='#000000', edgecolor='none',
                            alpha=0.12, zorder=1)
        ax.add_patch(sh)
        # body
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.02,rounding_size=0.10",
                              facecolor=fill, edgecolor=edge, lw=1.5, zorder=2)
        ax.add_patch(rect)
        # header colour strip
        hs = FancyBboxPatch((cx - w/2, cy + h/2 - 0.34), w, 0.34,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor=edge, edgecolor=edge, lw=0, zorder=3)
        ax.add_patch(hs)
        ax.text(cx, cy + h/2 - 0.17, title, ha='center', va='center',
                fontsize=10.0, fontweight='bold', color='white', zorder=4)
        ax.text(cx, cy - 0.10, detail, ha='center', va='center',
                fontsize=8.6, color='#222', linespacing=1.3, zorder=4)
        return (cx, cy, w, h)

    def harrow(b1, b2, label=None):
        ax.annotate('', xy=(b2[0]-b2[2]/2, b2[1]), xytext=(b1[0]+b1[2]/2, b1[1]),
                    arrowprops=arrow)
        if label:
            xm = (b1[0]+b1[2]/2 + b2[0]-b2[2]/2) / 2
            ax.text(xm, b1[1] + b1[3]/2 + 0.18, label, ha='center', va='bottom',
                    fontsize=8.0, fontstyle='italic', color='#607d8b')

    def bracket(x0, x1, y, label, color, fs=11.0):
        ax.plot([x0, x1], [y, y], color=color, lw=1.8, solid_capstyle='round', zorder=2)
        ax.plot([x0, x0], [y, y+0.13], color=color, lw=1.8, zorder=2)
        ax.plot([x1, x1], [y, y+0.13], color=color, lw=1.8, zorder=2)
        ax.text((x0+x1)/2, y-0.30, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=color)

    # ===== ROW 1 : input -> preprocessing -> features ===================
    y1 = 5.5
    cx = [1.55, 3.85, 6.05, 8.25, 11.0]
    b_in   = card(cx[0], y1, 'INPUT', 'Raw IMU\nSisFall · 9 ch\n200 Hz', 'in')
    b_filt = card(cx[1], y1, 'FILTER', 'Butterworth\nlow-pass 15 Hz', 'pre')
    b_win  = card(cx[2], y1, 'WINDOW', '2 s window\n50% overlap', 'pre')
    b_avm  = card(cx[3], y1, 'LABEL CLEAN', 'AVM refinement\nthreshold 1.8 g', 'pre')
    b_feat = card(cx[4], y1, 'FEATURES', '54 descriptors\njerk · AVM · SMA\ntilt · spectral', 'feat', w=2.45, h=1.35)
    harrow(b_in, b_filt)
    harrow(b_filt, b_win)
    harrow(b_win, b_avm, 'N × 400 × 9')
    harrow(b_avm, b_feat)

    yb1 = y1 - 1.35/2 - 0.40
    bracket(cx[1]-1.05, cx[3]+1.05, yb1, 'Signal Preprocessing', PAL['pre'][1])
    bracket(cx[4]-1.30, cx[4]+1.30, yb1, 'Feature Engineering', PAL['feat'][1])

    # wrap arrow down to row 2 (with feature-vector annotation)
    y2 = 2.7
    ax.annotate('', xy=(cx[4], y2 + 1.35/2), xytext=(cx[4], y1 - 1.35/2),
                arrowprops=arrow)
    ax.text(cx[4]+0.20, y1 - 1.35/2 - 0.25, 'N × 54', ha='left', va='center',
            fontsize=8.0, fontstyle='italic', color='#607d8b')

    # ===== ROW 2 : modeling -> evaluation (right -> left) ===============
    b_split = card(cx[4], y2, 'SPLIT', 'Subject-wise\nTrain / Val / Test', 'model', w=2.45, h=1.35)
    b_clf   = card(cx[3], y2, 'TRAIN MODELS', 'RF · SVM · KNN\nCNN-LSTM\nBiLSTM-Att', 'model', w=2.35, h=1.35)
    b_thr   = card(cx[2], y2, 'CALIBRATE', 'Threshold\ntuning (Val)', 'eval')
    b_agg   = card(cx[1], y2, 'AGGREGATE', 'Event-level\nvoting', 'eval')
    b_pred  = card(cx[0], y2, 'OUTPUT', 'Prediction\nFall / ADL', 'out')
    for a, b in [(b_split, b_clf), (b_clf, b_thr), (b_thr, b_agg), (b_agg, b_pred)]:
        ax.annotate('', xy=(b[0]+b[2]/2, y2), xytext=(a[0]-a[2]/2, y2), arrowprops=arrow)

    yb2 = y2 - 1.35/2 - 0.40
    bracket(cx[3]-1.25, cx[4]+1.30, yb2, 'Modeling', PAL['model'][1])
    bracket(cx[0]-1.05, cx[2]+1.05, yb2, 'Intra-dataset Evaluation', PAL['eval'][1])

    # ===== Cross-dataset branch =========================================
    y3 = 0.55
    card(cx[2], y3, 'ZERO-SHOT CROSS-DATASET',
                'KFall  F1 98.64        UMAFall  F1 88.30',
                'cd', w=6.0, h=1.0)
    ax.annotate('', xy=(cx[2]+1.6, y3 + 0.5), xytext=(b_clf[0], y2 - 1.35/2),
                arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=1.8,
                                mutation_scale=18, connectionstyle="arc3,rad=0.18"))
    ax.text(b_clf[0]-0.10, (y2 + y3)/2 - 0.05, 'trained\nmodel', ha='right',
            va='center', fontsize=8.0, fontstyle='italic', color='#c0392b')

    fig.tight_layout()
    _save(fig, 'fig_pipeline')


# ── 6. Raw signal example (Fall vs ADL) ──────────────────────────────────────
def fig_signal_example():
    data_dir = None
    for candidate in [os.path.join(BASE_DIR, "SisFall Dataset", "SisFall_dataset"),
                      os.path.join(BASE_DIR, "data", "SisFall_dataset")]:
        if os.path.isdir(candidate):
            data_dir = candidate
            break
    if data_dir is None:
        print("  [skip] SisFall raw data not found")
        return

    from scipy.signal import butter, filtfilt
    b, c = butter(FILTER_ORDER, CUTOFF_HZ / (FS / 2), btype='low')

    def load_file(path):
        lines = open(path, 'r').readlines()
        rows = []
        for line in lines:
            vals = line.strip().rstrip(';').split(',')
            if len(vals) >= 9:
                rows.append([float(v) for v in vals[:9]])
        arr = np.array(rows)
        arr[:, 0:3] *= ADXL_FACTOR
        arr[:, 3:6] *= ITG_FACTOR
        arr[:, 6:9] *= MMA_FACTOR
        for i in range(9):
            arr[:, i] = filtfilt(b, c, arr[:, i])
        return arr

    fall_file = os.path.join(data_dir, "SA01", "F01_SA01_R01.txt")
    adl_file = os.path.join(data_dir, "SA01", "D01_SA01_R01.txt")
    if not os.path.exists(fall_file) or not os.path.exists(adl_file):
        print("  [skip] SA01 data files not found")
        return

    fall_data = load_file(fall_file)
    adl_data = load_file(adl_file)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    t_fall = np.arange(len(fall_data)) / FS
    t_adl = np.arange(len(adl_data)) / FS

    for i, (label, color) in enumerate([('ADXL_x', '#e41a1c'),
                                         ('ADXL_y', '#377eb8'),
                                         ('ADXL_z', '#4daf4a')]):
        axes[0].plot(t_fall, fall_data[:, i], color=color, lw=0.8,
                     label=label.split('_')[1] + '-axis')
    avm_fall = np.sqrt(np.sum(fall_data[:, 0:3]**2, axis=1))
    peak_idx = np.argmax(avm_fall)
    axes[0].axvline(t_fall[peak_idx], color='red', ls='--', lw=1, alpha=0.7)
    axes[0].text(t_fall[peak_idx] + 0.1, axes[0].get_ylim()[1] * 0.8,
                 f'AVM peak\n= {avm_fall[peak_idx]:.1f} g', fontsize=8, color='red')
    axes[0].set_title('(a) Fall Event (F01 — Forward Fall)', fontweight='bold')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    for i, (label, color) in enumerate([('ADXL_x', '#e41a1c'),
                                         ('ADXL_y', '#377eb8'),
                                         ('ADXL_z', '#4daf4a')]):
        axes[1].plot(t_adl, adl_data[:, i], color=color, lw=0.8,
                     label=label.split('_')[1] + '-axis')
    axes[1].set_title('(b) ADL Activity (D01 — Standing Up)', fontweight='bold')
    axes[1].set_ylabel('Acceleration (g)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, 'fig_signal_example')


# ── 7. AVM distribution and label refinement ────────────────────────────────
def fig_avm_distribution():
    cache_dir = os.path.join(BASE_DIR, "data", "cache")
    npz_path = os.path.join(cache_dir, "windows_ADXL.npz")
    if not os.path.exists(npz_path):
        print("  [skip] windows_ADXL.npz not found in cache")
        return

    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    data['subjects'] if 'subjects' in data else None
    data['meta'] if 'meta' in data else None

    avm_peaks = np.sqrt(np.sum(X**2, axis=2)).max(axis=1) if X.ndim == 3 else None
    if avm_peaks is None:
        avm = np.sqrt(np.sum(X[:, :3]**2, axis=1)) if X.ndim == 2 else None
        if avm is None:
            print("  [skip] cannot compute AVM from cached data shape", X.shape)
            return
        avm_peaks = avm

    fall_avm = avm_peaks[y == 1]
    adl_avm = avm_peaks[y == 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(adl_avm, bins=80, alpha=0.7, color='#2c7fb8', label='ADL', density=True)
    axes[0].hist(fall_avm, bins=80, alpha=0.7, color='#de2d26', label='Fall', density=True)
    axes[0].axvline(FALL_AVM_THRESHOLD, color='black', ls='--', lw=2, label=f'Threshold = {FALL_AVM_THRESHOLD} g')
    axes[0].set_xlabel('Peak AVM (g)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(a) AVM Peak Distribution by Class')
    axes[0].legend()
    axes[0].set_xlim(0, min(avm_peaks.max(), 12))
    axes[0].grid(alpha=0.3)

    # Real counts from the canonical training partition (no_avm vs refined).
    # Computed once from data_loader/windowing; hardcoded here to keep figure
    # generation independent of the dataset loader and matches the numbers
    # quoted in the manuscript and ablation Table 11 (condition C).
    n_fall_before = 18816    # train fall-file windows BEFORE AVM refinement
    n_fall_after  = 4024     # train fall windows AFTER  AVM refinement (canonical)
    n_fall_removed = n_fall_before - n_fall_after
    categories = ['Before\nRefinement', 'Removed\n(AVM < 1.8g)', 'After\nRefinement']
    values = [n_fall_before, n_fall_removed, n_fall_after]
    colors_bar = ['#fdae61', '#de2d26', '#2c7fb8']
    bars = axes[1].bar(categories, values, color=colors_bar, edgecolor='black', lw=0.5)
    axes[1].bar_label(bars, fmt='%d', fontsize=10, fontweight='bold', padding=3)
    axes[1].set_ylabel('Number of Fall Windows')
    pct = n_fall_removed / n_fall_before * 100 if n_fall_before > 0 else 0
    axes[1].set_title(f'(b) Label Refinement Effect (Training Set) — '
                       f'{pct:.0f}% of fall-file windows removed')
    axes[1].grid(axis='y', alpha=0.3)
    # Headroom so bar_labels do not collide with the title
    axes[1].set_ylim(0, max(values) * 1.18)

    fig.tight_layout()
    _save(fig, 'fig_avm_distribution')


# ── 8. Confusion matrices: intra-dataset + zero-shot cross-dataset ──────────
def fig_confusion_matrix():
    def _draw(ax, cm, title, total):
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        vmax = cm.max()
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > vmax * 0.4 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=22, fontweight='bold', color=color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['ADL', 'Fall'])
        ax.set_yticklabels(['ADL', 'Fall'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return im

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) SisFall intra-dataset: ADXL + RF, 308 events
    cm_sisfall = np.array([[262, 1],
                           [0,  45]])
    im_a = _draw(axes[0], cm_sisfall,
                 '(a) SisFall — ADXL + RF (intra-dataset, 308 events)',
                 308)
    plt.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04).set_label('Count')

    # (b) KFall zero-shot: ADXL+ITG + KNN, 5,075 events (highest accuracy 98.76%)
    # ADL: TN=2729, FP=0 (2729 total); Fall: FN=63, TP=2283 (2346 total)
    cm_kfall = np.array([[2729, 0],
                         [63, 2283]])
    im_b = _draw(axes[1], cm_kfall,
                 '(b) KFall — ADXL+ITG + KNN (zero-shot, 5,075 events)',
                 5075)
    plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04).set_label('Count')

    fig.tight_layout()
    _save(fig, 'fig_confusion_matrix')


# ── 9. DL training curves ───────────────────────────────────────────────────
def fig_dl_training():
    model_dir = os.path.join(BASE_DIR, "results", "models")
    pt_file = os.path.join(model_dir, "cnn_bilstm_attention_variant_ADXL.pt")
    if not os.path.exists(pt_file):
        print("  [skip] cnn_bilstm_attention_variant_ADXL.pt not found")
        return
    try:
        import torch
        ckpt = torch.load(pt_file, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  [skip] could not load DL checkpoint: {e}")
        return

    history = ckpt.get('history', None)
    if history is None:
        print("  [skip] no training history in checkpoint")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if 'train_loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 'b-', lw=1.5, label='Train Loss')
        axes[0].plot(epochs, history['val_loss'], 'r--', lw=1.5, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('(a) Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    if 'train_acc' in history and 'val_acc' in history:
        epochs = range(1, len(history['train_acc']) + 1)
        axes[1].plot(epochs, history['train_acc'], 'b-', lw=1.5, label='Train Acc')
        axes[1].plot(epochs, history['val_acc'], 'r--', lw=1.5, label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('(b) Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    elif 'train_f1' in history and 'val_f1' in history:
        epochs = range(1, len(history['train_f1']) + 1)
        axes[1].plot(epochs, history['train_f1'], 'b-', lw=1.5, label='Train F1')
        axes[1].plot(epochs, history['val_f1'], 'r--', lw=1.5, label='Val F1')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1-score')
        axes[1].set_title('(b) Training & Validation F1')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.suptitle('CNN-BiLSTM-Attention Training Curves (ADXL Variant)', fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig_dl_training')


# ── 10. t-SNE feature space visualization ────────────────────────────────────
def fig_tsne():
    cache_dir = os.path.join(BASE_DIR, "data", "cache")
    feat_path = os.path.join(cache_dir, "features_ADXL.npz")
    if not os.path.exists(feat_path):
        print("  [skip] features_ADXL.npz not found")
        return

    data = np.load(feat_path, allow_pickle=True)
    X = data['X_feat']
    y = data['y']

    n = min(3000, len(X))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), n, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  [skip] sklearn not available for t-SNE")
        return

    X_scaled = StandardScaler().fit_transform(X_sub)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 6))
    adl_mask = y_sub == 0
    fall_mask = y_sub == 1
    ax.scatter(X_2d[adl_mask, 0], X_2d[adl_mask, 1], c='#2c7fb8', s=8,
               alpha=0.4, label=f'ADL (n={adl_mask.sum()})')
    ax.scatter(X_2d[fall_mask, 0], X_2d[fall_mask, 1], c='#de2d26', s=12,
               alpha=0.6, label=f'Fall (n={fall_mask.sum()})')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('t-SNE Visualization of Extracted Features (ADXL, 54 features)')
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, 'fig_tsne')


# ── 11. ROC curves for all models on ADXL ────────────────────────────────────
def fig_roc_curves():
    cache_dir = os.path.join(BASE_DIR, "data", "cache")
    feat_path = os.path.join(cache_dir, "features_ADXL.npz")
    if not os.path.exists(feat_path):
        print("  [skip] features_ADXL.npz not found for ROC")
        return

    data = np.load(feat_path, allow_pickle=True)
    X = data['X_feat']
    y = data['y']
    subjects = data['subjects']

    from config import TEST_SA, TEST_SE
    test_subjects = TEST_SA + TEST_SE
    test_mask = np.isin(subjects, test_subjects)
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_test) == 0:
        print("  [skip] no test data found for ROC")
        return

    try:
        from sklearn.metrics import roc_curve, auc
        import joblib
    except ImportError:
        print("  [skip] sklearn not available for ROC")
        return

    model_dir = os.path.join(BASE_DIR, "results", "models")
    ml_models = {
        'RF': os.path.join(model_dir, 'RF_variant_ADXL.pkl'),
        'SVM': os.path.join(model_dir, 'SVM_variant_ADXL.pkl'),
        'KNN': os.path.join(model_dir, 'KNN_variant_ADXL.pkl'),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {'RF': '#2c7fb8', 'SVM': '#fdae61', 'KNN': '#7fcdbb'}

    for name, path in ml_models.items():
        if not os.path.exists(path):
            continue
        pipeline = joblib.load(path)
        if hasattr(pipeline, 'predict_proba'):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, 'decision_function'):
            y_score = pipeline.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[name], lw=2,
                label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — ML Models on ADXL Test Set (Window-level)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig_roc_curves')


def main():
    print("Generating figures ->", FIG_DIR)
    fig_feature_importance()
    fig_sensor_ablation()
    fig_cross_dataset()
    fig_domain_adaptation()
    fig_pipeline()
    fig_signal_example()
    fig_avm_distribution()
    fig_confusion_matrix()
    fig_dl_training()
    fig_tsne()
    fig_roc_curves()
    print("Done.")


if __name__ == '__main__':
    main()

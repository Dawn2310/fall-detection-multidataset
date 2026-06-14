"""
evaluate.py — Plot charts, print results, save comparison tables + event-level metrics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config import FIG_DIR, METRIC_DIR


# ─── Event-level aggregation ─────────────────────────────────────────────────
def event_level_metrics(y_true_win: np.ndarray,
                         y_prob_win: np.ndarray,
                         subjects: np.ndarray,
                         activity_codes: np.ndarray,
                         trials: np.ndarray,
                         threshold: float = 0.5,
                         agg: str = 'max') -> dict:
    """
    Aggregate window predictions by event (subject + activity_code + trial).
    Event level Fall = F* file has at least 1 window exceeding threshold.
    Event level ADL  = D* file has all windows below threshold.

    agg = 'max' (max prob in event, most correct) | 'mean' (average probability)
    Returns: dict accuracy/sensitivity/specificity/precision/f1/tp/tn/fp/fn + n_events
    """
    df = pd.DataFrame({
        'subject':       subjects,
        'activity_code': activity_codes,
        'trial':         trials,
        'y_true':        y_true_win,
        'y_prob':        y_prob_win,
    })

    if agg == 'max':
        ev = df.groupby(['subject', 'activity_code', 'trial']).agg(
            event_true=('y_true', 'max'),
            event_prob=('y_prob', 'max'),
        )
    else:  # mean
        ev = df.groupby(['subject', 'activity_code', 'trial']).agg(
            event_true=('y_true', 'max'),
            event_prob=('y_prob', 'mean'),
        )
    ev['event_pred'] = (ev['event_prob'] >= threshold).astype(int)
    y_t = ev['event_true'].to_numpy()
    y_p = ev['event_pred'].to_numpy()

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = len(ev)
    return {
        'accuracy':    (tp + tn) / n if n > 0 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'precision':   tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'f1':          2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'n_events': int(n),
        'threshold': float(threshold),
        'agg': agg,
    }

os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)


# ─── Confusion Matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           title: str = "Confusion Matrix",
                           save_name: str | None = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['ADL (pred)', 'Fall (pred)'],
                yticklabels=['ADL (true)', 'Fall (true)'])
    ax.set_title(title)
    plt.tight_layout()
    if save_name:
        path = os.path.join(FIG_DIR, save_name)
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# ─── ROC Curve ───────────────────────────────────────────────────────────────
def plot_roc_curve(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    label: str = "Model",
                    save_name: str | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True)
    plt.tight_layout()
    if save_name:
        path = os.path.join(FIG_DIR, save_name)
        plt.savefig(path, dpi=150)
    plt.show()


# ─── Compare sensor variants ─────────────────────────────────────────────────
def plot_sensor_variant_comparison(results_df: pd.DataFrame,
                                    metric: str = 'f1',
                                    split: str = 'Test',
                                    save_name: str = "sensor_comparison.png"):
    """
    Plot bar chart comparing 7 sensor combinations.
    results_df: DataFrame with columns experiment, model, split, + metrics
    """
    df = results_df[results_df['split'] == split].copy()
    df['variant'] = df['experiment'].str.replace('variant_', '')

    metrics_labels = {
        'accuracy':    'Accuracy',
        'sensitivity': 'Sensitivity (Recall)',
        'specificity': 'Specificity',
        'f1':          'F1-score',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, (met, label) in enumerate(metrics_labels.items()):
        ax = axes[idx]
        pivot = df.pivot(index='variant', columns='model', values=met)
        pivot.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel('Sensor Variant')
        ax.set_ylabel(label)
        ax.set_ylim(max(0, df[met].min() - 0.05), 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(title='Model', fontsize=9)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.5)

        # Add values on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=7, padding=2)

    plt.suptitle(f'Sensor Variant Comparison ({split} Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.show()


# ─── Compare cross-age ───────────────────────────────────────────────────────
def plot_cross_age_gap(intra_df: pd.DataFrame,
                        cross_df: pd.DataFrame,
                        save_name: str = "cross_age_gap.png"):
    """
    Plot chart showing performance drop when trained on SA, tested on SE.
    """
    intra = intra_df[intra_df['split']=='Test'].copy()
    cross = cross_df[cross_df['split']=='CrossAge_SE'].copy()
    intra['scenario'] = 'Intra-Age (SA->SA)'
    cross['scenario'] = 'Cross-Age (SA->SE)'

    combined = pd.concat([intra, cross])

    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1']
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, met in zip(axes, metrics):
        sns.barplot(data=combined, x='model', y=met, hue='scenario',
                    ax=ax, palette=['steelblue', 'tomato'])
        ax.set_title(met.capitalize())
        ax.set_ylim(max(0, combined[met].min() - 0.1), 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.set_xlabel('')
        ax.legend(title='', fontsize=8)

    plt.suptitle('Intra-Age vs Cross-Age Performance Gap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.show()


# ─── Summary results table (paper-ready) ─────────────────────────────────────
def make_results_table(results_df: pd.DataFrame,
                        save_name: str = "results_summary.csv") -> pd.DataFrame:
    """Create formatted results table for paper usage."""
    cols = ['experiment', 'model', 'split',
            'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc']
    available = [c for c in cols if c in results_df.columns]
    table = results_df[available].copy()

    for met in ['accuracy','sensitivity','specificity','precision','f1','auc']:
        if met in table.columns:
            table[met] = (table[met] * 100).round(2)

    path = os.path.join(METRIC_DIR, save_name)
    table.to_csv(path, index=False)
    print(f"Saved results table: {path}")
    return table


# ─── Feature Importance (for RF) ─────────────────────────────────────────────
def plot_feature_importance(rf_model,
                             feature_names: list[str],
                             top_n: int = 20,
                             save_name: str = "feature_importance.png"):
    """Plot top N most important features of Random Forest."""
    importances = rf_model.named_steps['clf'].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[idx][::-1], color='steelblue', edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.show()

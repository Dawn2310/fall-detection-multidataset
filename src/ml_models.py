"""
ml_models.py — ML models with hyperparameter optimization using GridSearchCV.
Models: Random Forest, SVM, KNN
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score,
                              precision_score, f1_score,
                              roc_auc_score, confusion_matrix)
from config import (RF_PARAM_GRID, SVM_PARAM_GRID, KNN_PARAM_GRID,
                    RANDOM_STATE, CV_FOLDS, N_JOBS, MODEL_DIR)
from config import ML_MAX_SAMPLES

try:
    from . import progress as pg
except ImportError:
    try:
        import progress as pg
    except ImportError:
        class _NoOp:
            def __getattr__(self, _):
                return lambda *a, **k: None
        pg = _NoOp()


# ─── Compute metrics function ─────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray | None = None) -> dict:
    """
    Compute full evaluation metrics:
      Accuracy, Sensitivity (Recall), Specificity, Precision, F1, AUC
    """
    # labels=[0,1] ensures matrix is always 2x2 even if 1 class is missing
    # (e.g., cross-age test on SE is almost entirely ADL)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy':    accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # recall
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'precision':   precision_score(y_true, y_pred, zero_division=0),
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['auc'] = float('nan')
    return metrics


def print_metrics(metrics: dict, prefix: str = "") -> None:
    p = f"[{prefix}] " if prefix else ""
    print(f"{p}Accuracy:    {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"{p}Sensitivity: {metrics['sensitivity']:.4f}  (detect Fall)")
    print(f"{p}Specificity: {metrics['specificity']:.4f}  (recognize ADL)")
    print(f"{p}Precision:   {metrics['precision']:.4f}")
    print(f"{p}F1-score:    {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"{p}AUC-ROC:     {metrics['auc']:.4f}")
    print(f"{p}CM: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")


# ─── Class MLTrainer ─────────────────────────────────────────────────────────
class MLTrainer:
    """
    Wraps Pipeline (Scaler + Classifier) with GridSearchCV.

    Usage:
        trainer = MLTrainer(model_name='RF')
        trainer.fit(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        trainer.save('results/models/RF_ALL9.pkl')
    """

    MODELS = {
        'RF':  (RandomForestClassifier(random_state=RANDOM_STATE),
                {f'clf__{k}': v for k, v in RF_PARAM_GRID.items()}),
        'SVM': (SVC(probability=True, random_state=RANDOM_STATE),
                {f'clf__{k}': v for k, v in SVM_PARAM_GRID.items()}),
        'KNN': (KNeighborsClassifier(),
                {f'clf__{k}': v for k, v in KNN_PARAM_GRID.items()}),
    }

    def __init__(self, model_name: str = 'RF', cv_folds: int = CV_FOLDS):
        if model_name not in self.MODELS:
            raise ValueError(f"model_name must be: {list(self.MODELS.keys())}")
        self.model_name = model_name
        self.cv_folds   = cv_folds
        self.best_params_ = None
        self.pipeline_    = None
        self.cv_results_  = None
        self.train_time_  = None

        clf, param_grid = self.MODELS[model_name]
        self.pipeline_ = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf),
        ])
        self.param_grid = param_grid

    # Default subsample: SVM scales O(n^2-n^3) -> must cap; RF/KNN unchanged
    DEFAULT_MAX_SAMPLES = ML_MAX_SAMPLES

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            verbose: int = 1,
            max_samples: int | None = "auto") -> "MLTrainer":
        """Run GridSearchCV to find optimal parameters, then train on full train set.

        max_samples: 'auto' = use DEFAULT_MAX_SAMPLES by model_name
                     int    = cap number of train samples (stratified)
                     None   = use all data
        """
        if max_samples == "auto":
            max_samples = self.DEFAULT_MAX_SAMPLES.get(self.model_name)

        if max_samples is not None and len(X_train) > max_samples:
            from sklearn.model_selection import train_test_split
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train,
                train_size=max_samples,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )
            print(f"  [Subsample] {self.model_name}: capped at {max_samples:,} samples")

        print(f"\n{'='*55}")
        print(f"  GridSearchCV: {self.model_name}  |  {self.cv_folds}-fold CV")
        print(f"  Train: {X_train.shape}  |  Fall:{(y_train==1).sum()}  ADL:{(y_train==0).sum()}")
        print(f"{'='*55}")

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=self.pipeline_,
            param_grid=self.param_grid,
            cv=cv,
            scoring='f1',       # optimize F1 (balance Sensitivity vs Specificity)
            n_jobs=N_JOBS,
            verbose=verbose,
            return_train_score=True,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        self.train_time_ = time.time() - t0

        self.pipeline_    = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.cv_results_  = pd.DataFrame(grid.cv_results_)

        print(f"\nBest params: {self.best_params_}")
        print(f"Best CV F1:  {grid.best_score_:.4f}")
        print(f"Train time:  {self.train_time_:.1f}s")
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 split_name: str = "Test",
                 threshold: float = 0.5) -> dict:
        """Evaluate model on any dataset with optional threshold."""
        y_prob = self.pipeline_.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y, y_pred, y_prob)
        metrics['threshold'] = threshold
        print(f"\n--- {self.model_name} | {split_name} | thr={threshold:.2f} ---")
        print_metrics(metrics, self.model_name)
        return metrics

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                       metric: str = 'f1') -> float:
        """Scan threshold 0.05-0.95, select optimal thr by F1 (default) on Val."""
        y_prob = self.pipeline_.predict_proba(X_val)[:, 1]
        thrs = np.linspace(0.05, 0.95, 19)
        best_thr, best_score = 0.5, -1.0
        for t in thrs:
            y_pred = (y_prob >= t).astype(int)
            m = compute_metrics(y_val, y_pred, y_prob)
            if m[metric] > best_score:
                best_score, best_thr = m[metric], float(t)
        print(f"  [tune_threshold] best thr={best_thr:.2f} | best {metric}={best_score:.4f}")
        return best_thr

    def save(self, filepath: str | None = None) -> str:
        """Save trained model."""
        if filepath is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            filepath = os.path.join(MODEL_DIR, f"{self.model_name}_best.pkl")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline_, filepath)
        print(f"Model saved: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str, model_name: str = 'RF') -> "MLTrainer":
        trainer = cls.__new__(cls)
        trainer.model_name = model_name
        trainer.pipeline_ = joblib.load(filepath)
        return trainer


# ─── Run all models on 1 dataset ───────────────────────────────────
MODEL_LIST = ['RF', 'SVM', 'KNN']


def run_all_models(X_train: np.ndarray, y_train: np.ndarray,
                   X_val:   np.ndarray, y_val:   np.ndarray,
                   X_test:  np.ndarray, y_test:  np.ndarray,
                   experiment_name: str = "default",
                   save_models: bool = True,
                   resume: bool = True) -> pd.DataFrame:
    """
    Train and evaluate RF, SVM, KNN.
    Return summary results DataFrame.

    resume=True: if .pkl file exists, reload instead of retraining.
    """
    results = []
    for mi, model_name in enumerate(MODEL_LIST, 1):
        pg.set_model(model_name, mi)
        fname = f"{model_name}_{experiment_name}.pkl"
        fpath = os.path.join(MODEL_DIR, fname)

        if resume and os.path.exists(fpath):
            pg.log(f"[RESUME] Load {fname} (skip GridSearchCV)")
            trainer = MLTrainer.load(fpath, model_name=model_name)
            trainer.train_time_ = None
            trainer.best_params_ = None
        else:
            pg.log(f"Train {model_name}...")
            trainer = MLTrainer(model_name)
            trainer.fit(X_train, y_train)
            if save_models:
                trainer.save(fpath)

        # === Threshold tuning on Val ===
        pg.log(f"Tune threshold on Val ({model_name})...")
        best_thr = trainer.tune_threshold(X_val, y_val, metric='f1')

        for split_name, Xs, ys in [('Val',  X_val,  y_val),
                                   ('Test', X_test, y_test)]:
            pg.log(f"Evaluate {model_name} on {split_name}...")
            metrics = trainer.evaluate(Xs, ys, split_name, threshold=best_thr)
            results.append({
                'experiment': experiment_name,
                'model':      model_name,
                'split':      split_name,
                **metrics,
                'train_time': trainer.train_time_,
                'best_params': str(trainer.best_params_),
            })

    df = pd.DataFrame(results)
    return df


# ─── Train on SA, evaluate on SE (cross-age) ─────────────────────────────
def run_cross_age(X_train_SA: np.ndarray, y_train_SA: np.ndarray,
                  X_test_SE:  np.ndarray, y_test_SE:  np.ndarray,
                  sensor_variant: str = "ALL9",
                  resume: bool = True,
                  X_val: np.ndarray | None = None,
                  y_val: np.ndarray | None = None) -> pd.DataFrame:
    """
    Train on young adults (SA), test on elderly (SE).
    This is Cross-Age Generalization experiment.

    resume=True: if crossage .pkl exists, reload.
    """
    pg.banner(f"CROSS-AGE EVALUATION: {sensor_variant}", char="=")
    pg.log(f"Train SA: {X_train_SA.shape} | Test SE: {X_test_SE.shape}")

    results = []
    for mi, model_name in enumerate(MODEL_LIST, 1):
        pg.set_model(model_name, mi)
        fname = f"{model_name}_CrossAge_{sensor_variant}.pkl"
        fpath = os.path.join(MODEL_DIR, fname)

        if resume and os.path.exists(fpath):
            pg.log(f"[RESUME] Load {fname}")
            trainer = MLTrainer.load(fpath, model_name=model_name)
            trainer.train_time_ = None
        else:
            pg.log(f"Train {model_name} (cross-age)...")
            trainer = MLTrainer(model_name)
            trainer.fit(X_train_SA, y_train_SA)
            trainer.save(fpath)

        # Tune threshold on SA val (if available), otherwise use 0.5
        if X_val is not None and y_val is not None and len(np.unique(y_val)) > 1:
            pg.log(f"Tune threshold on Val SA ({model_name})...")
            best_thr = trainer.tune_threshold(X_val, y_val, metric='f1')
        else:
            best_thr = 0.5

        pg.log(f"Evaluate {model_name} on SE (thr={best_thr:.2f})...")
        metrics = trainer.evaluate(X_test_SE, y_test_SE, "CrossAge_SE", threshold=best_thr)
        results.append({
            'experiment':     f"CrossAge_{sensor_variant}",
            'model':          model_name,
            'split':          'CrossAge_SE',
            **metrics,
            'train_time':     trainer.train_time_,
        })
    return pd.DataFrame(results)

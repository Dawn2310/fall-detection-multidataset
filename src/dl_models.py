"""
dl_models.py — CNN-LSTM and CNN-BiLSTM-Attention models using PyTorch + CUDA.
Input: (batch, window_size, n_channels)
Output: Fall probability (sigmoid)

Rationale: TensorFlow >= 2.11 on native Windows does not support NVIDIA GPU.
PyTorch uses CUDA directly -> RTX 4050 runs fast immediately.
"""
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import (BATCH_SIZE, EPOCHS, PATIENCE, LR,
                    MODEL_DIR, RANDOM_STATE, DL_DEVICE, DL_ARCHS)
from ml_models import compute_metrics, print_metrics

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

torch.manual_seed(RANDOM_STATE)


def _get_device() -> torch.device:
    if DL_DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── CNN-LSTM Architecture ──────────────────────────────────────────────────────
class CNNLSTM(nn.Module):
    """Conv1D x2 -> LSTM -> Dense -> Sigmoid."""

    def __init__(self, n_channels: int,
                 conv_filters=(64, 128),
                 lstm_units: int = 64,
                 dense_units: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, conv_filters[0], kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(conv_filters[0])
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(conv_filters[1])
        self.pool2 = nn.MaxPool1d(2)
        self.lstm  = nn.LSTM(conv_filters[1], lstm_units, batch_first=True)
        self.fc1   = nn.Linear(lstm_units, dense_units)
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(dense_units, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h).squeeze(-1)


# ─── CNN-BiLSTM-Attention Architecture ──────────────────────────────────────────
class CNNBiLSTMAttention(nn.Module):
    """Conv1D x2 -> BiLSTM -> MultiHead Self-Attention -> GAP -> Dense -> Sigmoid."""

    def __init__(self, n_channels: int,
                 conv_filters=(64, 128),
                 lstm_units: int = 64,
                 dense_units: int = 32,
                 dropout: float = 0.3,
                 num_heads: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, conv_filters[0], kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(conv_filters[0])
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(conv_filters[1])
        self.pool2 = nn.MaxPool1d(2)
        self.bilstm = nn.LSTM(conv_filters[1], lstm_units,
                               batch_first=True, bidirectional=True)
        self.attn   = nn.MultiheadAttention(embed_dim=2 * lstm_units,
                                             num_heads=num_heads,
                                             batch_first=True)
        self.fc1  = nn.Linear(2 * lstm_units, dense_units)
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(dense_units, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        out, _ = self.bilstm(x)
        attn_out, _ = self.attn(out, out, out)
        h = attn_out.mean(dim=1)
        h = F.relu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h).squeeze(-1)


ARCH_MAP = {
    'cnn_lstm':             CNNLSTM,
    'cnn_bilstm_attention': CNNBiLSTMAttention,
}


# ─── DLTrainer ───────────────────────────────────────────────────────────────
class DLTrainer:
    def __init__(self, arch: str = 'cnn_lstm', n_channels: int = 9,
                 device: torch.device | None = None):
        if arch not in ARCH_MAP:
            raise ValueError(f"arch must be one of {list(ARCH_MAP.keys())}")
        self.arch = arch
        self.device = device or _get_device()
        self.model = ARCH_MAP[arch](n_channels).to(self.device)
        self.history_ = {'train_loss': [], 'val_loss': [],
                          'train_acc':  [], 'val_acc':  []}
        self.train_time_ = None
        self.best_threshold_ = 0.5
        pg.log(f"[DL] arch={arch} | device={self.device} | n_channels={n_channels}")

    def _make_loader(self, X, y, batch_size, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(self.device.type == 'cuda'))

    def fit(self, X_train, y_train, X_val, y_val,
            batch_size: int = BATCH_SIZE,
            epochs:     int = EPOCHS,
            patience:   int = PATIENCE,
            lr:       float = LR) -> "DLTrainer":

        train_loader = self._make_loader(X_train, y_train, batch_size, True)
        val_loader   = self._make_loader(X_val,   y_val,   batch_size, False)

        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_w = max(n_neg / max(n_pos, 1.0), 1.0)
        pg.log(f"[DL] pos_weight={pos_w:.2f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_w], device=self.device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

        os.makedirs(MODEL_DIR, exist_ok=True)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            # === Train ===
            self.model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * xb.size(0)
                pred = (torch.sigmoid(logits) >= 0.5).float()
                tr_correct += (pred == yb).sum().item()
                tr_total   += yb.size(0)
            tr_loss /= max(tr_total, 1)
            tr_acc  =  tr_correct / max(tr_total, 1)

            # === Val ===
            self.model.eval()
            va_loss, va_correct, va_total = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    va_loss += loss.item() * xb.size(0)
                    pred = (torch.sigmoid(logits) >= 0.5).float()
                    va_correct += (pred == yb).sum().item()
                    va_total   += yb.size(0)
            va_loss /= max(va_total, 1)
            va_acc  =  va_correct / max(va_total, 1)

            self.history_['train_loss'].append(tr_loss)
            self.history_['val_loss'].append(va_loss)
            self.history_['train_acc'].append(tr_acc)
            self.history_['val_acc'].append(va_acc)

            scheduler.step(va_loss)
            cur_lr = optimizer.param_groups[0]['lr']
            pg.log(f"[DL] ep {ep:02d}/{epochs} | loss tr={tr_loss:.4f} va={va_loss:.4f} | "
                   f"acc tr={tr_acc:.4f} va={va_acc:.4f} | lr={cur_lr:.1e}")

            if va_loss < best_val_loss - 1e-4:
                best_val_loss = va_loss
                epochs_no_improve = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    pg.log(f"[DL] early stopping at epoch {ep} (no improve {patience})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.train_time_ = time.time() - t0
        pg.log(f"[DL] train_time = {self.train_time_:.1f}s")
        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        loader = self._make_loader(X, np.zeros(len(X), dtype=np.float32),
                                    batch_size, False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)

    def tune_threshold(self, X_val, y_val) -> float:
        y_prob = self.predict_proba(X_val)
        thrs = np.linspace(0.05, 0.95, 19)
        best_thr, best_f1 = 0.5, -1.0
        for t in thrs:
            y_pred = (y_prob >= t).astype(int)
            m = compute_metrics(y_val, y_pred, y_prob)
            if m['f1'] > best_f1:
                best_f1, best_thr = m['f1'], float(t)
        self.best_threshold_ = best_thr
        pg.log(f"  [DL tune_threshold] best thr={best_thr:.2f} | best F1={best_f1:.4f}")
        return best_thr

    def evaluate(self, X, y, split_name: str = "Test",
                 threshold: float | None = None) -> dict:
        thr = threshold if threshold is not None else self.best_threshold_
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= thr).astype(int)
        metrics = compute_metrics(y, y_pred, y_prob)
        metrics['threshold'] = thr
        print(f"\n--- {self.arch.upper()} | {split_name} | thr={thr:.2f} ---", flush=True)
        print_metrics(metrics, self.arch)
        return metrics

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'state_dict': self.model.state_dict(),
                    'arch':       self.arch,
                    'best_threshold': self.best_threshold_,
                    'history':    self.history_}, path)
        pg.log(f"DL model saved: {path}")
        return path

    def plot_history(self, save_path: str | None = None):
        import matplotlib.pyplot as plt
        from config import FIG_DIR
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history_['train_loss'], label='Train')
        axes[0].plot(self.history_['val_loss'],   label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(self.history_['train_acc'], label='Train')
        axes[1].plot(self.history_['val_acc'],   label='Val')
        axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True)
        plt.suptitle(f'{self.arch.upper()} Training History')
        plt.tight_layout()
        if save_path is None:
            os.makedirs(FIG_DIR, exist_ok=True)
            save_path = os.path.join(FIG_DIR, f"{self.arch}_history.png")
        plt.savefig(save_path, dpi=130)
        plt.close()
        pg.log(f"Saved: {save_path}")


# ─── Run all DL architectures ────────────────────────────────────────────────
def run_dl_models(X_train, y_train, X_val, y_val, X_test, y_test,
                  experiment_name: str = "default",
                  resume: bool = True) -> pd.DataFrame:
    """Train and evaluate DL architectures in DL_ARCHS."""
    results = []
    n_channels = X_train.shape[-1]

    for arch in DL_ARCHS:
        pg.set_model(arch, 1)
        ckpt = os.path.join(MODEL_DIR, f"{arch}_{experiment_name}.pt")
        trainer = DLTrainer(arch, n_channels=n_channels)

        if resume and os.path.exists(ckpt):
            pg.log(f"[RESUME] Load {ckpt}")
            chk = torch.load(ckpt, map_location=trainer.device)
            trainer.model.load_state_dict(chk['state_dict'])
            trainer.best_threshold_ = chk.get('best_threshold', 0.5)
            trainer.history_ = chk.get('history', trainer.history_)
        else:
            trainer.fit(X_train, y_train, X_val, y_val)
            trainer.tune_threshold(X_val, y_val)
            trainer.save(ckpt)
            try:
                trainer.plot_history()
            except Exception as e:
                pg.log(f"[plot history skip] {e}")

        for split_name, Xs, ys in [('Val',  X_val,  y_val),
                                   ('Test', X_test, y_test)]:
            m = trainer.evaluate(Xs, ys, split_name)
            results.append({
                'experiment': experiment_name,
                'model':      arch,
                'split':      split_name,
                **m,
                'train_time': trainer.train_time_,
            })

    return pd.DataFrame(results)

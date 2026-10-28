#!/usr/bin/env python3
"""Plots and metrics for a RawMod training run.

    rawmod-visualize RUN_DIR [--out RUN_DIR/plots]

RUN_DIR is a run as scripts/train/run_matched_loco.sh writes it. Every history.npz
below it (train_loss, val_auprc, best_epoch per epoch) becomes a training-curve
figure; every scores/*.npz (y_true, y_score per test site) becomes a precision-recall
curve with its threshold sweep, a confusion matrix at the F1-optimal threshold, and a
row in metrics.tsv. The functions can also be imported and used on any (y_true, y_score).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

try:
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ModuleNotFoundError:
    def confusion_matrix(y_true, y_pred, labels=None):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        labels = [0, 1] if labels is None else list(labels)
        out = np.zeros((len(labels), len(labels)), dtype=np.int64)
        label_to_i = {label: i for i, label in enumerate(labels)}
        for true, pred in zip(y_true, y_pred):
            if true in label_to_i and pred in label_to_i:
                out[label_to_i[true], label_to_i[pred]] += 1
        return out

    def precision_score(y_true, y_pred, pos_label=1, zero_division=0):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
        fp = int(((y_true != pos_label) & (y_pred == pos_label)).sum())
        denom = tp + fp
        return float(zero_division) if denom == 0 else float(tp / denom)

    def recall_score(y_true, y_pred, pos_label=1, zero_division=0):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
        fn = int(((y_true == pos_label) & (y_pred != pos_label)).sum())
        denom = tp + fn
        return float(zero_division) if denom == 0 else float(tp / denom)

    def f1_score(y_true, y_pred, pos_label=1, zero_division=0):
        precision = precision_score(
            y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
        recall = recall_score(
            y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
        denom = precision + recall
        return float(zero_division) if denom == 0 else float(2 * precision * recall / denom)

    def precision_recall_curve(y_true, y_score):
        y_true = (np.asarray(y_true).astype(int) == 1).astype(np.int64)
        y_score = np.asarray(y_score).astype(float)
        if y_true.size == 0:
            return np.array([1.0]), np.array([0.0]), np.array([])

        order = np.argsort(y_score, kind='mergesort')[::-1]
        y_true = y_true[order]
        y_score = y_score[order]

        distinct = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        thresholds = y_score[threshold_idxs]

        precision = tps / np.maximum(tps + fps, 1)
        if tps[-1] == 0:
            recall = np.ones_like(tps, dtype=float)
        else:
            recall = tps / tps[-1]

        sl = slice(None, None, -1)
        return (
            np.r_[precision[sl], 1.0],
            np.r_[recall[sl], 0.0],
            thresholds[sl],
        )

    def average_precision_score(y_true, y_score):
        y_true = (np.asarray(y_true).astype(int) == 1).astype(np.int64)
        if int(y_true.sum()) == 0:
            return 0.0
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(-np.sum(np.diff(recall) * precision[:-1]))

    def roc_auc_score(y_true, y_score):
        y_true = (np.asarray(y_true).astype(int) == 1)
        y_score = np.asarray(y_score).astype(float)
        n_pos = int(y_true.sum())
        n_neg = int((~y_true).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError('ROC AUC is undefined with one class.')

        order = np.argsort(y_score, kind='mergesort')
        sorted_scores = y_score[order]
        ranks = np.empty_like(sorted_scores, dtype=float)
        i = 0
        while i < len(sorted_scores):
            j = i + 1
            while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            ranks[i:j] = (i + 1 + j) / 2.0
            i = j
        original_ranks = np.empty_like(ranks)
        original_ranks[order] = ranks
        pos_rank_sum = float(original_ranks[y_true].sum())
        return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def classification_report(y_true, y_pred, labels=None, target_names=None,
                              zero_division=0):
        labels = [0, 1] if labels is None else list(labels)
        target_names = target_names or [str(label) for label in labels]
        rows = ['              precision    recall  f1-score   support']
        for label, name in zip(labels, target_names):
            support = int((np.asarray(y_true).astype(int) == label).sum())
            p = precision_score(y_true, y_pred, pos_label=label,
                                zero_division=zero_division)
            r = recall_score(y_true, y_pred, pos_label=label,
                             zero_division=zero_division)
            f = f1_score(y_true, y_pred, pos_label=label,
                         zero_division=zero_division)
            rows.append(f'{name:>12s} {p:>10.2f} {r:>9.2f} {f:>9.2f} {support:>9d}')
        return '\n'.join(rows)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    plt = None
    _MPL_IMPORT_ERROR = exc


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to generate figures. Activate the same "
            "Python environment used for training, or install matplotlib in "
            "the current environment."
        ) from _MPL_IMPORT_ERROR


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the threshold that maximizes positive-class F1."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or int(y_true.sum()) == 0:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1s = (2 * precisions[:-1] * recalls[:-1]
           / (precisions[:-1] + recalls[:-1] + 1e-8))
    return float(thresholds[int(np.argmax(f1s))])


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Compute precision, recall, F1, AUPRC, and AUROC.

    If there are no modified positives, metrics are reported for correctly
    predicting the unmodified class, matching loo.py's historical behavior.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return {
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'auprc': float('nan'),
            'auroc': float('nan'),
            'threshold': 0.5,
            'all_unmod': False,
        }

    if int(y_true.sum()) == 0:
        thresh = 0.5 if threshold is None else float(threshold)
        y_pred = (y_prob >= thresh).astype(int)
        return {
            'precision': float(precision_score(
                y_true, y_pred, pos_label=0, zero_division=0)),
            'recall': float(recall_score(
                y_true, y_pred, pos_label=0, zero_division=0)),
            'f1': float(f1_score(
                y_true, y_pred, pos_label=0, zero_division=0)),
            'auprc': float(np.mean(1.0 - y_prob)),
            'auroc': float('nan'),
            'threshold': thresh,
            'all_unmod': True,
        }

    thresh = optimal_threshold(y_true, y_prob) if threshold is None else float(threshold)
    y_pred = (y_prob >= thresh).astype(int)
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auprc': float(average_precision_score(y_true, y_prob)),
        'auroc': (float(roc_auc_score(y_true, y_prob))
                  if len(np.unique(y_true)) == 2 else float('nan')),
        'threshold': thresh,
        'all_unmod': False,
    }


def plot_training_curves(train_losses, val_losses, val_auprcs, best_epoch, out_path):
    _require_matplotlib()
    epochs = list(range(1, len(train_losses) + 1))
    if not epochs:
        print(f"  Skipping training curves; no history available for {out_path}")
        return

    best_epoch = int(best_epoch) if best_epoch else int(np.argmax(val_auprcs) + 1)
    best_epoch = max(1, min(best_epoch, len(epochs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label='Train loss', color='steelblue')
    if val_losses is not None and len(val_losses) == len(epochs):
        ax1.plot(epochs, val_losses, label='Validation loss', color='darkorange')
    ax1.axvline(best_epoch, color='red', linestyle='--', linewidth=1.0,
                label=f'Best epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend(fontsize=8)
    if len(epochs) > 1:
        ax1.set_xlim(1, max(epochs))

    best_auprc = float(val_auprcs[best_epoch - 1])
    ax2.plot(epochs, val_auprcs, label='Val AUPRC', color='seagreen')
    ax2.axvline(best_epoch, color='red', linestyle='--', linewidth=1.0,
                label=f'Best epoch ({best_epoch})  AUPRC={best_auprc:.4f}')
    ax2.scatter([best_epoch], [best_auprc], color='red', zorder=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUPRC')
    ax2.set_title('Validation AUPRC')
    if len(epochs) > 1:
        ax2.set_xlim(1, max(epochs))
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle('Training curves')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Training curves -> {out_path}")


def plot_precision_recall(y_true, y_prob, auprc, out_path):
    _require_matplotlib()
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    all_unmod = int(y_true.sum()) == 0
    if all_unmod:
        plot_true = 1 - y_true
        plot_prob = 1.0 - y_prob
        plot_auprc = average_precision_score(plot_true, plot_prob)
        curve_label = f'Unmodified AUPRC = {plot_auprc:.4f}'
        best_thresh = 0.5
        y_pred_mod = (y_prob >= best_thresh).astype(int)
        best_prec = precision_score(y_true, y_pred_mod,
                                    pos_label=0, zero_division=0)
        best_rec = recall_score(y_true, y_pred_mod,
                                pos_label=0, zero_division=0)
        best_f1 = f1_score(y_true, y_pred_mod,
                           pos_label=0, zero_division=0)
        sweep_t = np.linspace(0, 1, 300)
        sweep_f1 = [
            f1_score(y_true, (y_prob >= t).astype(int),
                     pos_label=0, zero_division=0)
            for t in sweep_t
        ]
        precisions, recalls, _ = precision_recall_curve(plot_true, plot_prob)
    else:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1s = (2 * precisions[:-1] * recalls[:-1]
               / (precisions[:-1] + recalls[:-1] + 1e-8))
        best_idx = int(np.argmax(f1s))
        best_thresh = float(thresholds[best_idx])
        best_prec = float(precisions[best_idx])
        best_rec = float(recalls[best_idx])
        best_f1 = float(f1s[best_idx])
        sweep_t = np.linspace(0, 1, 300)
        sweep_f1 = [
            f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            for t in sweep_t
        ]
        curve_label = f'AUPRC = {auprc:.4f}'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={'width_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(recalls, precisions, label=curve_label, color='steelblue')
    ax.scatter([best_rec], [best_prec], color='red', zorder=5,
               label=f'thresh={best_thresh:.3f}  '
                     f'prec={best_prec:.3f}  rec={best_rec:.3f}  F1={best_f1:.3f}')
    ax.axvline(best_rec, color='red', linewidth=0.8, linestyle=':')
    ax.axhline(best_prec, color='red', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(sweep_t, sweep_f1, color='seagreen')
    ax2.axvline(best_thresh, color='red', linestyle=':',
                label=f'optimal threshold = {best_thresh:.3f}')
    ax2.axhline(best_f1, color='red', linestyle=':',
                label=f'optimal F1 = {best_f1:.3f}')
    ax2.scatter([best_thresh], [best_f1], color='red', zorder=5)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 vs. Threshold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle('Precision-recall and threshold sweep')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"  PR curve -> {out_path}")
    print(f"  Optimal threshold (max F1): {best_thresh:.4f}")
    print(f"    Precision : {best_prec:.4f}")
    print(f"    Recall    : {best_rec:.4f}")
    print(f"    F1        : {best_f1:.4f}")
    return best_thresh


def plot_confusion_matrix(y_true, y_prob, threshold, out_path):
    _require_matplotlib()
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.maximum(row_sums, 1)
    labels = ['Unmodified', 'Modified']

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                   vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion of true class', fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix (threshold = {threshold:.4f})')

    for i in range(2):
        for j in range(2):
            col = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i - 0.1, f'{cm[i, j]:,}',
                    ha='center', va='center', fontsize=12,
                    fontweight='bold', color=col)
            ax.text(j, i + 0.18, f'({cm_norm[i, j] * 100:.1f}%)',
                    ha='center', va='center', fontsize=9, color=col)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix -> {out_path}")


def _load_history(path: Path) -> tuple:
    with np.load(path, allow_pickle=True) as d:
        tr = d['train_loss'] if 'train_loss' in d else d['train_losses']
        va = d['val_loss'] if 'val_loss' in d else d['val_losses'] if 'val_losses' in d else None
        ap = d['val_auprc'] if 'val_auprc' in d else d['val_auprcs']
        best = int(_scalar(d['best_epoch'])) if 'best_epoch' in d else 0
    return tr, va, ap, best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--out', type=Path, default=None, help='where to write the plots (default RUN_DIR/plots)')
    a = ap.parse_args(argv)
    out = a.out or a.run_dir / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for h in sorted(a.run_dir.rglob('history.npz')):
        tr, va, auprc, best = _load_history(h)
        tag = h.parent.name if h.parent != a.run_dir else 'run'
        plot_training_curves(tr, va, auprc, best, out / f'training_curves_{tag}.png'); n += 1
    rows = []
    for s in sorted(a.run_dir.rglob('scores/*.npz')):
        with np.load(s, allow_pickle=True) as d:
            if 'y_true' not in d or 'y_score' not in d:
                continue
            y, p = np.asarray(d['y_true']).astype(int), np.asarray(d['y_score']).astype(float)
        if len(y) == 0 or y.min() == y.max():
            print(f'  {s.name}: one class only, skipped'); continue
        auprc = float(average_precision_score(y, p))
        thr = plot_precision_recall(y, p, auprc, out / f'pr_{s.stem}.png')
        plot_confusion_matrix(y, p, thr, out / f'confusion_{s.stem}.png')
        m = compute_binary_metrics(y, p, thr)
        rows.append({'test': s.stem, 'n': len(y), 'n_pos': int(y.sum()), 'auroc': round(float(roc_auc_score(y, p)), 4),
                     'auprc': round(auprc, 4), 'threshold': round(thr, 4),
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
        n += 2
    if rows:
        with open(out / 'metrics.tsv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
        print(f'  metrics -> {out / "metrics.tsv"}')
    if not n:
        print(f'nothing to plot under {a.run_dir}: no history.npz or scores/*.npz', file=sys.stderr); return 1
    print(f'{n} figures in {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

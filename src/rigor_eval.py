#!/usr/bin/env python3
"""rigor_eval.py — Adds the ML rigor the paper currently lacks.

Produces, with NO Mininet required:
  1. DETECTION vs PREDICTION comparison on the large dataset:
       - label_consensus  -> congestion-STATE detection (the current F1=0.9723 claim)
       - label_predictive -> loss within the next polls = genuine EARLY WARNING
     This is the honest reframing: report both, with their true look-ahead.
  2. Standard ML figures (expected by any reviewer, currently missing):
       ROC curve, PR curve, confusion matrix, calibration plot.
  3. Cross-CC generalization: train on Reno, test on Cubic (and vice-versa),
     turning "the unified model generalizes" from an assertion into evidence.

Usage:
    python3 src/rigor_eval.py                      # all stages, sampled for speed
    python3 src/rigor_eval.py --sample 0           # use FULL large dataset (slow)
    python3 src/rigor_eval.py --stage figures      # only the figures stage
    python3 src/rigor_eval.py --stage labels       # only detection-vs-prediction
    python3 src/rigor_eval.py --stage crosscc      # only cross-CC generalization

Outputs:
    results/rigor_results.json
    results/figures/rigor/{roc,pr,confusion_matrix,calibration}.png
"""
import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             precision_recall_curve, roc_curve, auc,
                             confusion_matrix)
import lightgbm as lgb

try:
    from imblearn.over_sampling import BorderlineSMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(ROOT, 'data')
FIGDIR = os.path.join(ROOT, 'results', 'figures', 'rigor')

LEAK = ['lost', 'retrans_now', 'retrans_total', 'retrans_diff', 'bytes_retrans', 'sacked']
LABEL_COLS = ['label', 'label_basepaper', 'label_retrans', 'label_consensus', 'label_predictive']
NON_PRED = ['mss', 'pmtu', 'rcv_space']


def prep(df, target):
    """Return (X, y): drop leak cols, all label cols, non-predictive cols; numeric only;
    then variance + >0.98 correlation filtering (same as the training pipeline)."""
    y = df[target].astype(int)
    drop = [c for c in (LEAK + LABEL_COLS + NON_PRED) if c in df.columns]
    X = df.drop(columns=drop, errors='ignore').select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    sel = VarianceThreshold(1e-8)
    X = pd.DataFrame(sel.fit_transform(X), columns=X.columns[sel.get_support()], index=X.index)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if any(upper[c] > 0.98)]
    return X.drop(columns=drop_corr), y


def best_threshold(y, p):
    bt, bf = 0.5, 0.0
    for t in np.arange(0.15, 0.85, 0.01):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return bt


def smote(X, y):
    if not HAS_SMOTE or int(np.sum(y)) < 6:
        return X, y
    try:
        k = min(5, int(np.sum(y)) - 1)
        return BorderlineSMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
    except Exception:
        return X, y


def lgbm():
    return lgb.LGBMClassifier(
        n_estimators=600, max_depth=7, learning_rate=0.03, min_child_samples=12,
        subsample=0.85, colsample_bytree=0.8, is_unbalance=True,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)


def cv_oof(X, y, n_splits=5):
    """Time-series CV; return out-of-fold (y_true, y_prob) for honest evaluation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    yt, yp = [], []
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        Xtr_s, ytr_b = smote(Xtr_s, ytr)
        m = lgbm()
        m.fit(Xtr_s, ytr_b)
        yp.extend(m.predict_proba(Xte_s)[:, 1].tolist())
        yt.extend(yte.tolist())
    return np.array(yt), np.array(yp)


def metrics(y, p):
    t = best_threshold(y, p)
    pred = (p >= t).astype(int)
    prec, rec, _ = precision_recall_curve(y, p)
    return {
        'threshold': round(float(t), 3),
        'accuracy': round(float(accuracy_score(y, pred)), 4),
        'f1': round(float(f1_score(y, pred)), 4),
        'roc_auc': round(float(roc_auc_score(y, p)), 4),
        'pr_auc': round(float(auc(rec, prec)), 4),
        'positive_rate': round(float(np.mean(y)), 4),
    }


def load_large(sample):
    path = os.path.join(DATA, 'large', 'large_dataset.csv')
    if not os.path.exists(path):
        print(f'  [SKIP] {path} not found'); return None
    # contiguous head keeps temporal ordering for TimeSeriesSplit; nrows avoids
    # loading the full 634 MB file when sampling.
    df = pd.read_csv(path, nrows=sample if sample else None)
    return df.reset_index(drop=True)


# ── Stage 1: detection vs prediction ─────────────────────────────────────────
def stage_labels(sample, results):
    print('\n' + '=' * 70)
    print('  STAGE 1 — DETECTION vs PREDICTION (large dataset)')
    print('=' * 70)
    df = load_large(sample)
    if df is None:
        return
    print(f'  Rows used: {len(df):,}')
    out = {}
    for target, kind, horizon in [
        ('label_consensus', 'congestion-state DETECTION (current claim)', 'same poll (0 ms ahead)'),
        ('label_predictive', 'loss EARLY WARNING (genuine prediction)', '<=2 polls / ~40 ms ahead'),
    ]:
        if target not in df.columns:
            print(f'  [SKIP] {target} missing'); continue
        X, y = prep(df, target)
        yt, yp = cv_oof(X, y)
        m = metrics(yt, yp)
        out[target] = {'kind': kind, 'horizon': horizon, 'n_features': X.shape[1], **m}
        print(f'\n  {target}  [{kind}]')
        print(f'    horizon={horizon}  positives={m["positive_rate"]*100:.1f}%')
        print(f'    F1={m["f1"]}  ROC-AUC={m["roc_auc"]}  PR-AUC={m["pr_auc"]}  thr={m["threshold"]}')
    results['detection_vs_prediction'] = out
    print('\n  -> Use BOTH in the paper: report detection F1 AND the honest early-warning F1.')


# ── Stage 2: figures (paper dataset, matches the published F1) ────────────────
def stage_figures(results):
    print('\n' + '=' * 70)
    print('  STAGE 2 — ML FIGURES (paper dataset: enhanced_dataset.csv)')
    print('=' * 70)
    path = os.path.join(DATA, 'enhanced_dataset.csv')
    if not os.path.exists(path):
        print(f'  [SKIP] {path} not found'); return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    df = pd.read_csv(path)
    X, y = prep(df, 'label')
    yt, yp = cv_oof(X, y, n_splits=7)
    m = metrics(yt, yp)
    results['paper_dataset_lightgbm'] = m
    print(f'  LightGBM (consensus): F1={m["f1"]} ROC-AUC={m["roc_auc"]} (paper reports 0.9723 / 0.9965)')
    os.makedirs(FIGDIR, exist_ok=True)
    thr = m['threshold']

    # ROC
    fpr, tpr, _ = roc_curve(yt, yp)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f'LightGBM (AUC={m["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC — LightGBM (7-fold TimeSeriesSplit)'); plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'roc.png'), dpi=150); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(yt, yp)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, lw=2, label=f'LightGBM (AP={m["pr_auc"]:.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision–Recall — LightGBM'); plt.legend(loc='lower left')
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'pr.png'), dpi=150); plt.close()

    # Confusion matrix
    cm = confusion_matrix(yt, (yp >= thr).astype(int))
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)
    plt.xticks([0, 1], ['No Loss', 'Loss']); plt.yticks([0, 1], ['No Loss', 'Loss'])
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Confusion Matrix (thr={thr})')
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'confusion_matrix.png'), dpi=150); plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10)
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, 's-', label='LightGBM')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    plt.axvline(thr, color='r', ls=':', label=f'Deployment thr={thr}')
    plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives')
    plt.title('Calibration'); plt.legend(loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(FIGDIR, 'calibration.png'), dpi=150); plt.close()
    print(f'  -> 4 figures written to {FIGDIR}')


# ── Stage 3: cross-CC generalization ─────────────────────────────────────────
def stage_crosscc(results):
    print('\n' + '=' * 70)
    print('  STAGE 3 — CROSS-CC GENERALIZATION (train on one CC, test on the other)')
    print('=' * 70)
    reno_p = os.path.join(DATA, 'enhanced_reno.csv')
    cubic_p = os.path.join(DATA, 'enhanced_cubic.csv')
    if not (os.path.exists(reno_p) and os.path.exists(cubic_p)):
        print('  [SKIP] enhanced_reno.csv / enhanced_cubic.csv not found'); return
    reno, cubic = pd.read_csv(reno_p), pd.read_csv(cubic_p)
    out = {}
    for train_name, train_df, test_name, test_df in [
        ('reno', reno, 'cubic', cubic), ('cubic', cubic, 'reno', reno)]:
        Xtr, ytr = prep(train_df, 'label')
        Xte, yte = prep(test_df, 'label')
        common = [c for c in Xtr.columns if c in Xte.columns]
        Xtr, Xte = Xtr[common], Xte[common]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        Xtr_s, ytr_b = smote(Xtr_s, ytr)
        m = lgbm(); m.fit(Xtr_s, ytr_b)
        p = m.predict_proba(Xte_s)[:, 1]
        res = metrics(yte, p)
        out[f'train_{train_name}_test_{test_name}'] = {'n_features': len(common), **res}
        print(f'  train={train_name:5s} -> test={test_name:5s} : '
              f'F1={res["f1"]}  ROC-AUC={res["roc_auc"]}')
    results['cross_cc_generalization'] = out
    print('  -> If F1 holds across CC, the "unified model generalizes" claim is evidenced.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['all', 'labels', 'figures', 'crosscc'], default='all')
    ap.add_argument('--sample', type=int, default=300000,
                    help='Rows from large dataset for stage 1 (0 = full, slow)')
    args = ap.parse_args()

    results = {}
    if args.stage in ('all', 'labels'):
        stage_labels(args.sample, results)
    if args.stage in ('all', 'figures'):
        stage_figures(results)
    if args.stage in ('all', 'crosscc'):
        stage_crosscc(results)

    out = os.path.join(ROOT, 'results', 'rigor_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved -> {out}')


if __name__ == '__main__':
    main()

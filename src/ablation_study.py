#!/usr/bin/env python3
"""ablation_study.py — Systematically evaluate contribution of each feature group.

Ablations:
  1. All features (baseline)
  2. -Rolling statistics
  3. -Lag features
  4. -Diff/Acceleration features
  5. -Interaction/ratio features
  6. -Network condition features
  7. Raw ss features only (no engineering)
  8. No SMOTE (class balancing removed)
  9. Default hyperparameters (no Optuna tuning)
"""
import pandas as pd
import numpy as np
import json, os, warnings, time
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: pip install lightgbm"); exit(1)

try:
    from imblearn.over_sampling import BorderlineSMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARN: imbalanced-learn not found, SMOTE ablation skipped")


# ── Feature group definitions ────────────────────────────────────────────────
ROLLING_FEATURES   = ['cwnd_roll5_mean','cwnd_roll5_std','cwnd_roll3_mean',
                      'cwnd_roll10_mean','rtt_roll5_mean','rtt_roll5_std',
                      'rtt_roll3_mean','rtt_roll10_mean']

LAG_FEATURES       = ['cwnd_lag1','cwnd_lag2','cwnd_lag3',
                      'rtt_lag1','rtt_lag2','rtt_lag3']

DIFF_FEATURES      = ['cwnd_diff','cwnd_accel','rtt_diff','rtt_accel',
                      'cwnd_diff2','rtt_diff2']

INTERACTION_FEATURES = ['cwnd_rtt_interaction','send_efficiency','buffer_fill_ratio',
                         'retrans_ratio','ack_ratio','rtt_ratio','cwnd_util',
                         'throughput','bdp_estimate','rtt_zscore','cwnd_volatility',
                         'congestion_signal','rtt_range','cwnd_range']

NETWORK_COND       = ['delay_ms','bandwidth_mbit','buffer_bytes','duration_s']

EWM_FEATURES       = ['cwnd_ewm','rtt_ewm']

RUNNING_EXTREMES   = ['run_min_cwnd','run_max_cwnd','run_min_rtt','run_max_rtt',
                      'cwnd_roll5_max']

# All engineered features combined (everything except raw ss fields)
ALL_ENGINEERED = (ROLLING_FEATURES + LAG_FEATURES + DIFF_FEATURES +
                  INTERACTION_FEATURES + NETWORK_COND + EWM_FEATURES + RUNNING_EXTREMES)

# Leak features — always removed before training
LEAK_FEATURES = ['lost','retrans_now','retrans_total','retrans_diff',
                 'bytes_retrans','sacked','mss','pmtu','rcv_space']


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def get_X_y(df, drop_extra=None):
    """Return X, y after removing leak + any extra specified columns."""
    to_drop = ['label'] + [c for c in LEAK_FEATURES if c in df.columns]
    if drop_extra:
        to_drop += [c for c in drop_extra if c in df.columns]

    X = df.drop(columns=to_drop, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['label']
    return X, y


def run_cv(X, y, use_smote=True, default_params=False, n_splits=5):
    """Run TimeSeriesSplit CV and return mean F1, Acc, AUC."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if default_params:
        params = dict(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    else:
        params = dict(n_estimators=400, max_depth=7, learning_rate=0.05,
                      num_leaves=63, min_child_samples=20,
                      is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)

    fold_f1, fold_acc, fold_auc = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        # SMOTE
        if use_smote and HAS_SMOTE and y_train.sum() >= 6:
            try:
                k = min(5, int(y_train.sum()) - 1)
                sm = BorderlineSMOTE(random_state=42, k_neighbors=k)
                X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
            except Exception:
                pass

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_s, y_train,
                  eval_set=[(X_val_s, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])

        prob  = model.predict_proba(X_val_s)[:, 1]
        # Quick threshold search
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.25, 0.75, 0.05):
            p = (prob >= t).astype(int)
            f = f1_score(y_val, p, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t

        pred = (prob >= best_t).astype(int)
        fold_f1.append(f1_score(y_val, pred, zero_division=0))
        fold_acc.append(accuracy_score(y_val, pred))
        try:
            fold_auc.append(roc_auc_score(y_val, prob))
        except Exception:
            fold_auc.append(0.5)

    return {
        'f1':      round(float(np.mean(fold_f1)),  4),
        'f1_std':  round(float(np.std(fold_f1)),   4),
        'acc':     round(float(np.mean(fold_acc)), 4),
        'auc':     round(float(np.mean(fold_auc)), 4),
        'n_features': X.shape[1],
    }


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    csv_path = os.path.join(data_dir, 'enhanced_dataset.csv')

    print('=' * 72)
    print('  ABLATION STUDY — Feature Group Contribution Analysis')
    print('  Model: LightGBM | CV: 5-fold TimeSeriesSplit')
    print('=' * 72)

    df = load_data(csv_path)
    print(f'  Dataset: {len(df):,} rows | {df.shape[1]} columns\n')

    ablations = [
        ('1. All Features (Baseline)',          None,              True,  False),
        ('2. Remove Rolling Stats',             ROLLING_FEATURES,  True,  False),
        ('3. Remove Lag Features',              LAG_FEATURES,      True,  False),
        ('4. Remove Diff/Accel Features',       DIFF_FEATURES,     True,  False),
        ('5. Remove Interaction/Ratio Features',INTERACTION_FEATURES, True, False),
        ('6. Remove Network Conditions',        NETWORK_COND,      True,  False),
        ('7. Raw ss Features Only',             ALL_ENGINEERED,    True,  False),
        ('8. No SMOTE (Imbalanced Training)',   None,              False, False),
        ('9. Default Hyperparameters',          None,              True,  True ),
    ]

    results = []
    header = f"  {'Ablation':<40} {'F1':>7} {'±Std':>6} {'Acc':>7} {'AUC':>7} {'Feats':>6}"
    print(header)
    print('  ' + '-' * 76)

    baseline_f1 = None

    for name, drop_cols, use_smote, default_params in ablations:
        X, y = get_X_y(df, drop_extra=drop_cols)
        t0 = time.time()
        res = run_cv(X, y, use_smote=use_smote, default_params=default_params)
        elapsed = time.time() - t0

        if baseline_f1 is None:
            baseline_f1 = res['f1']
            delta_str = '(baseline)'
        else:
            delta = res['f1'] - baseline_f1
            sign  = '+' if delta >= 0 else ''
            delta_str = f'({sign}{delta:.4f})'

        print(f"  {name:<40} {res['f1']:>7.4f} {res['f1_std']:>6.4f} "
              f"{res['acc']:>7.4f} {res['auc']:>7.4f} {res['n_features']:>6}  "
              f"{delta_str}  [{elapsed:.0f}s]")

        results.append({'name': name, **res, 'delta_f1': round(res['f1'] - baseline_f1, 4)})

    print('\n' + '=' * 72)
    print('  FEATURE GROUP IMPACT RANKING (F1 drop when removed):')
    print('  ' + '-' * 50)

    ranked = sorted(results[1:8], key=lambda r: r['f1'])  # lower F1 = more important
    for r in ranked:
        drop = baseline_f1 - r['f1']
        bar  = '█' * int(drop * 200)
        print(f"  {r['name'][3:]:<35} -{drop:.4f}  {bar}")

    # Save results
    out_path = os.path.join(data_dir, 'ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump({'baseline_f1': baseline_f1, 'ablations': results}, f, indent=2)
    print(f'\n  Results saved → {out_path}')
    print('=' * 72)

    # Generate bar chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig_dir = os.path.join(data_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        names  = [r['name'][3:].strip() for r in results]  # strip "1. " prefix
        f1s    = [r['f1'] for r in results]
        stds   = [r['f1_std'] for r in results]
        colors = ['#2ECC71' if i == 0 else ('#E74C3C' if r['f1'] < baseline_f1 else '#F39C12')
                  for i, r in enumerate(results)]

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(range(len(names)), f1s, color=colors, edgecolor='white',
                      linewidth=1.5, yerr=stds, capsize=4, error_kw={'linewidth':1.5})
        ax.axhline(y=baseline_f1, color='#2ECC71', linestyle='--', linewidth=1.2,
                   label=f'Baseline F1={baseline_f1:.4f}')

        for bar, val, std in zip(bars, f1s, stds):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=22, ha='right', fontsize=9)
        ax.set_ylabel('F1-Score (mean ± std, 5-fold CV)')
        ax.set_title('Ablation Study — Feature Group Contribution',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_ylim(min(f1s) - 0.02, 1.0)
        plt.tight_layout()

        path = os.path.join(fig_dir, 'ablation_study.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  Figure saved → {path}')
    except Exception as e:
        print(f'  [Figure skipped: {e}]')


if __name__ == '__main__':
    main()

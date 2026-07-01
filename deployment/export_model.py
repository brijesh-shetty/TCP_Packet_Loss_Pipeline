#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os, sys, json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
import joblib

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: pip install lightgbm")
    sys.exit(1)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def main():
    # Find dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tcp_dir = os.path.dirname(script_dir)  # parent = tcp/
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Look for dataset
    for name in ['large_dataset.csv', 'enhanced_dataset.csv']:
        csv_path = os.path.join(tcp_dir, name)
        if os.path.exists(csv_path):
            break
    else:
        print("ERROR: No dataset found in tcp/ directory")
        sys.exit(1)

    print("=" * 70)
    print("  MODEL EXPORT — Training & Exporting for Ubuntu ECN")
    print("=" * 70)
    print(f"\n  Dataset: {os.path.basename(csv_path)}")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")

    # Use label_predictive for ECN (best for proactive prediction)
    label_col = 'label_predictive'
    if label_col not in df.columns:
        label_col = 'label'
    print(f"  Label: {label_col}")

    # Remove leak features
    leak_features = ['lost', 'retrans_now', 'retrans_total', 'retrans_diff',
                     'bytes_retrans', 'sacked']
    non_pred = ['mss', 'pmtu', 'rcv_space']
    all_label_cols = ['label', 'label_basepaper', 'label_retrans',
                      'label_consensus', 'label_predictive']

    drop_cols = [c for c in leak_features + non_pred + all_label_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[label_col]

    # Feature selection
    selector = VarianceThreshold(threshold=1e-8)
    X_selected = pd.DataFrame(
        selector.fit_transform(X),
        columns=X.columns[selector.get_support()],
        index=X.index
    )

    corr_matrix = X_selected.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]
    X_final = X_selected.drop(columns=to_drop)

    feature_names = X_final.columns.tolist()
    print(f"  Features: {len(feature_names)}")
    print(f"  Loss rate: {y.mean()*100:.1f}%")

    # Train/test split
    split_idx = int(len(X_final) * 0.8)
    X_train, X_test = X_final.iloc[:split_idx], X_final.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SMOTE
    if HAS_SMOTE and y_train.sum() >= 5:
        k = min(5, int(y_train.sum()) - 1)
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    # Train LightGBM
    print("\n  Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=600, max_depth=7, learning_rate=0.03,
        min_child_samples=12, subsample=0.85, colsample_bytree=0.8,
        is_unbalance=True, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_train_s, y_train)

    # Find best threshold
    y_prob = model.predict_proba(X_test_s)[:, 1]
    best_f1, best_threshold = 0, 0.5
    for t in np.arange(0.15, 0.85, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    print(f"  F1 Score: {best_f1:.4f}")
    print(f"  Threshold: {best_threshold:.2f}")

    # Also train XGBoost for base paper comparison
    try:
        import xgboost as xgb
        print("\n  Training XGBoost (for base paper comparison)...")
        scale_pw = (y.iloc[:split_idx] == 0).sum() / max((y.iloc[:split_idx] == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=600, max_depth=7, learning_rate=0.03,
            min_child_weight=6, subsample=0.85, colsample_bytree=0.8,
            scale_pos_weight=scale_pw, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train_s, y_train)

        xgb_prob = xgb_model.predict_proba(X_test_s)[:, 1]
        xgb_best_f1, xgb_threshold = 0, 0.5
        for t in np.arange(0.15, 0.85, 0.01):
            preds = (xgb_prob >= t).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > xgb_best_f1:
                xgb_best_f1, xgb_threshold = f1, t

        print(f"  XGBoost F1: {xgb_best_f1:.4f}, Threshold: {xgb_threshold:.2f}")

        # Save XGBoost
        joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.joblib'))
        print(f"  ✓ Saved xgb_model.joblib")
    except ImportError:
        xgb_best_f1, xgb_threshold = 0, 0.5
        print("  [SKIP] XGBoost not installed")

    # --- Save everything ---
    print("\n  Exporting models...")

    # Save LightGBM model
    joblib.dump(model, os.path.join(models_dir, 'lgbm_model.joblib'))
    print(f"  ✓ Saved lgbm_model.joblib")

    # Save scaler
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    print(f"  ✓ Saved scaler.joblib")

    # Save config
    config = {
        'feature_names': feature_names,
        'lgbm_threshold': float(best_threshold),
        'lgbm_f1': float(best_f1),
        'xgb_threshold': float(xgb_threshold) if 'xgb_threshold' in dir() else 0.5,
        'xgb_f1': float(xgb_best_f1) if 'xgb_best_f1' in dir() else 0,
        'label_col': label_col,
        'n_features': len(feature_names),
        'dataset': os.path.basename(csv_path),
        'dataset_rows': len(df),
        'loss_rate': float(y.mean()),
    }
    with open(os.path.join(models_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config.json")

    print(f"\n{'=' * 70}")
    print(f"  EXPORT COMPLETE!")
    print(f"  Files in: {models_dir}")
    print(f"")
    print(f"  Next: Upload the entire 'ubuntu_ecn' folder to Google Drive,")
    print(f"        then download on Ubuntu and run:")
    print(f"        sudo python3 run_experiment.py")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

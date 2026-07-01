#!/usr/bin/env python3
"""enhanced_train.py — High-performance TCP loss prediction with Optuna tuning, CatBoost, and stacking."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, f1_score, accuracy_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
import os, json, sys, time

# Optional deps
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('[WARN] xgboost not installed')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print('[WARN] lightgbm not installed')

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print('[WARN] catboost not installed. pip install catboost')

try:
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print('[WARN] imbalanced-learn not installed')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print('[WARN] optuna not installed. Using manual hyperparams.')


def load_and_prepare(csv_path):
    """Load dataset, remove leaking/non-predictive features."""
    df = pd.read_csv(csv_path)

    # Remove leaking features
    leak_features = ['lost', 'retrans_now', 'retrans_total', 'retrans_diff',
                     'bytes_retrans', 'sacked']
    drop_cols = [c for c in leak_features if c in df.columns]

    # Remove non-predictive columns
    non_pred = ['mss', 'pmtu', 'rcv_space']
    drop_cols += [c for c in non_pred if c in df.columns]

    X = df.drop(columns=['label'] + drop_cols, errors='ignore')
    y = df['label']

    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])

    # Replace inf with NaN, then fill NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Remove near-zero-variance features
    selector = VarianceThreshold(threshold=1e-8)
    X_selected = pd.DataFrame(
        selector.fit_transform(X),
        columns=X.columns[selector.get_support()],
        index=X.index
    )

    # Remove highly correlated features (>0.98)
    corr_matrix = X_selected.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    X_final = X_selected.drop(columns=to_drop)

    removed = list(set(drop_cols + to_drop))
    print(f'  Dataset: {len(df)} rows')
    print(f'  Features ({X_final.shape[1]}): {X_final.columns.tolist()}')
    print(f'  Label distribution: No Loss={int((y==0).sum())} ({(y==0).mean()*100:.1f}%), '
          f'Loss={int((y==1).sum())} ({(y==1).mean()*100:.1f}%)')
    print(f'  Dropped: {len(removed)} features')
    print()

    return X_final, y


def find_best_threshold(y_true, y_prob):
    """Find probability threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.15, 0.85, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def evaluate_model(name, y_true, y_pred, y_prob, threshold=0.5):
    """Comprehensive evaluation."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = 0.0

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(recall, precision)

    cm = confusion_matrix(y_true, y_pred)

    print(f'\n{"="*60}')
    print(f'  MODEL: {name}  (threshold={threshold:.2f})')
    print(f'{"="*60}')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  F1-Score:  {f1:.4f}')
    print(f'  ROC-AUC:   {roc:.4f}')
    print(f'  PR-AUC:    {pr_auc_val:.4f}')
    print(f'\n  Confusion Matrix:')
    print(f'    TN={cm[0,0]:6d}  FP={cm[0,1]:6d}')
    print(f'    FN={cm[1,0]:6d}  TP={cm[1,1]:6d}')
    print(f'\n{classification_report(y_true, y_pred, target_names=["No Loss", "Loss"], digits=4)}')

    return {
        'name': name, 'accuracy': round(acc, 4), 'f1': round(f1, 4),
        'roc_auc': round(roc, 4), 'pr_auc': round(pr_auc_val, 4), 'threshold': round(threshold, 2)
    }


def get_feature_importance(model, feature_names, name):
    """Extract and print top-15 feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    else:
        return

    idx = np.argsort(importances)[::-1]
    print(f'\n  Feature Importances ({name}):')
    print(f'  {"Rank":<5} {"Feature":<28} {"Importance":>12}')
    print(f'  {"-"*48}')
    for rank, j in enumerate(idx[:15], 1):
        bar = '#' * int(importances[j] / max(importances[idx[0]], 1e-9) * 20)
        print(f'  {rank:<5} {feature_names[j]:<28} {importances[j]:>12.4f}  {bar}')


def apply_smote(X_train, y_train):
    """Apply BorderlineSMOTE or fallback to regular SMOTE."""
    if not HAS_SMOTE or y_train.sum() < 5:
        return X_train, y_train

    try:
        k = min(5, int(y_train.sum()) - 1)
        try:
            sm = BorderlineSMOTE(random_state=42, k_neighbors=k)
            return sm.fit_resample(X_train, y_train)
        except:
            sm = SMOTE(random_state=42, k_neighbors=k)
            return sm.fit_resample(X_train, y_train)
    except:
        return X_train, y_train


def optuna_tune_xgb(X, y, n_trials=30):
    """Tune XGBoost hyperparameters using Optuna."""
    if not HAS_OPTUNA or not HAS_XGB:
        return None

    scale_pw = (y == 0).sum() / max((y == 1).sum(), 1)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            'scale_pos_weight': scale_pw,
            'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            X_tr_s, y_tr = apply_smote(X_tr_s, y_tr)

            m = xgb.XGBClassifier(**params)
            m.fit(X_tr_s, y_tr)
            y_prob = m.predict_proba(X_te_s)[:, 1]
            t, f1 = find_best_threshold(y_te, y_prob)
            scores.append(f1)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best['scale_pos_weight'] = scale_pw
    best['eval_metric'] = 'logloss'
    best['random_state'] = 42
    best['n_jobs'] = -1
    return best


def optuna_tune_lgb(X, y, n_trials=30):
    """Tune LightGBM hyperparameters using Optuna."""
    if not HAS_OPTUNA or not HAS_LGB:
        return None

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 25),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            'is_unbalance': True,
            'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            X_tr_s, y_tr = apply_smote(X_tr_s, y_tr)

            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr_s, y_tr)
            y_prob = m.predict_proba(X_te_s)[:, 1]
            t, f1 = find_best_threshold(y_te, y_prob)
            scores.append(f1)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best['is_unbalance'] = True
    best['random_state'] = 42
    best['n_jobs'] = -1
    best['verbose'] = -1
    return best


def train_and_evaluate(X, y, output_dir, dataset_name='Combined', use_optuna=True, n_trials=30):
    """Train models with SMOTE + Optuna + threshold tuning + stacking ensemble."""
    n_splits = 7
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    # --- Optuna Hyperparameter Tuning ---
    xgb_params = None
    lgb_params = None

    if use_optuna and HAS_OPTUNA:
        print(f'\n  [OPTUNA] Tuning XGBoost ({n_trials} trials)...')
        t0 = time.time()
        xgb_params = optuna_tune_xgb(X, y, n_trials)
        if xgb_params:
            print(f'    Best params found in {time.time()-t0:.1f}s')

        print(f'  [OPTUNA] Tuning LightGBM ({n_trials} trials)...')
        t0 = time.time()
        lgb_params = optuna_tune_lgb(X, y, n_trials)
        if lgb_params:
            print(f'    Best params found in {time.time()-t0:.1f}s')

    # Define models
    models = {}

    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        min_samples_leaf=12, subsample=0.85, max_features='sqrt',
        random_state=42
    )

    models['RandomForest'] = RandomForestClassifier(
        n_estimators=600, max_depth=12, min_samples_leaf=6,
        class_weight='balanced', max_features='sqrt',
        random_state=42, n_jobs=-1
    )

    if HAS_XGB:
        if xgb_params:
            models['XGBoost'] = xgb.XGBClassifier(**xgb_params)
        else:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=600, max_depth=7, learning_rate=0.03,
                min_child_weight=6, subsample=0.85, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight, reg_alpha=0.1, reg_lambda=1.0,
                eval_metric='logloss', random_state=42, n_jobs=-1
            )

    if HAS_LGB:
        if lgb_params:
            models['LightGBM'] = lgb.LGBMClassifier(**lgb_params)
        else:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=600, max_depth=7, learning_rate=0.03,
                min_child_samples=12, subsample=0.85, colsample_bytree=0.8,
                is_unbalance=True, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbose=-1
            )

    if HAS_CB:
        models['CatBoost'] = cb.CatBoostClassifier(
            iterations=600, depth=7, learning_rate=0.05,
            auto_class_weights='Balanced',
            random_seed=42, verbose=0
        )

    # ── Baseline models for academic comparison ──────────────────────────────
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB

    models['KNN'] = KNeighborsClassifier(
        n_neighbors=7, weights='distance', metric='minkowski',
        algorithm='auto', n_jobs=-1
    )
    models['DecisionTree'] = DecisionTreeClassifier(
        max_depth=10, min_samples_leaf=10,
        class_weight='balanced', random_state=42
    )
    models['NaiveBayes'] = GaussianNB(var_smoothing=1e-9)
    # ─────────────────────────────────────────────────────────────────────────

    results = []
    feature_names = X.columns.tolist()

    # --- Train individual models ---
    for model_name, model in models.items():
        print(f'\n{"#"*65}')
        print(f'  Training: {model_name}  ({n_splits}-fold TimeSeriesSplit + SMOTE)')
        print(f'{"#"*65}')

        all_y_true, all_y_pred, all_y_prob = [], [], []
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # SMOTE on training data only
            X_train_s, y_train = apply_smote(X_train_s, y_train)

            # Train
            m = type(model)(**model.get_params())
            m.fit(X_train_s, y_train)

            y_prob = m.predict_proba(X_test_s)[:, 1]
            best_t, _ = find_best_threshold(y_test, y_prob)
            y_pred = (y_prob >= best_t).astype(int)

            fold_f1 = f1_score(y_test, y_pred)
            fold_acc = accuracy_score(y_test, y_pred)
            print(f'  Fold {fold}: acc={fold_acc:.4f} f1={fold_f1:.4f} thresh={best_t:.2f}')
            fold_metrics.append({'fold': fold, 'acc': round(fold_acc, 4),
                                 'f1': round(fold_f1, 4), 'threshold': round(best_t, 2)})

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())

        global_thresh, _ = find_best_threshold(all_y_true, all_y_prob)
        final_pred = (np.array(all_y_prob) >= global_thresh).astype(int)

        result = evaluate_model(model_name, all_y_true, final_pred, all_y_prob, global_thresh)
        result['fold_metrics'] = fold_metrics
        results.append(result)

        # Feature importance
        scaler_full = StandardScaler()
        X_full_s = scaler_full.fit_transform(X)
        final_m = type(model)(**model.get_params())
        X_fit, y_fit = apply_smote(X_full_s, y)
        final_m.fit(X_fit, y_fit)
        get_feature_importance(final_m, feature_names, model_name)

    # --- Stacking Ensemble ---
    estimators = []
    if HAS_XGB:
        xp = xgb_params if xgb_params else {
            'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05,
            'scale_pos_weight': scale_pos_weight, 'eval_metric': 'logloss',
            'random_state': 42, 'n_jobs': -1
        }
        estimators.append(('xgb', xgb.XGBClassifier(**xp)))
    if HAS_LGB:
        lp = lgb_params if lgb_params else {
            'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05,
            'is_unbalance': True, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }
        estimators.append(('lgb', lgb.LGBMClassifier(**lp)))

    estimators.append(('gb', GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
    )))

    if HAS_CB:
        estimators.append(('cb', cb.CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            auto_class_weights='Balanced', random_seed=42, verbose=0
        )))

    if len(estimators) >= 3:
        print(f'\n{"#"*65}')
        print(f'  Training: StackingEnsemble  ({n_splits}-fold TimeSeriesSplit)')
        print(f'{"#"*65}')

        all_y_true, all_y_pred, all_y_prob = [], [], []
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

            scaler = StandardScaler()
            X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
            X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

            X_train_s, y_train = apply_smote(X_train_s, y_train)

            stack = StackingClassifier(
                estimators=[(n, type(m)(**m.get_params())) for n, m in estimators],
                final_estimator=LogisticRegression(
                    class_weight='balanced', max_iter=1000, C=1.0
                ),
                cv=3, stack_method='predict_proba', n_jobs=-1
            )

            stack.fit(X_train_s, y_train)
            y_prob = stack.predict_proba(X_test_s)[:, 1]
            best_t, _ = find_best_threshold(y_test, y_prob)
            y_pred = (y_prob >= best_t).astype(int)

            fold_f1 = f1_score(y_test, y_pred)
            fold_acc = accuracy_score(y_test, y_pred)
            print(f'  Fold {fold}: acc={fold_acc:.4f} f1={fold_f1:.4f} thresh={best_t:.2f}')
            fold_metrics.append({'fold': fold, 'acc': round(fold_acc, 4),
                                 'f1': round(fold_f1, 4), 'threshold': round(best_t, 2)})

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())

        global_thresh, _ = find_best_threshold(all_y_true, all_y_prob)
        final_pred = (np.array(all_y_prob) >= global_thresh).astype(int)
        result = evaluate_model('StackingEnsemble', all_y_true, final_pred, all_y_prob, global_thresh)
        result['fold_metrics'] = fold_metrics
        results.append(result)

    # --- Summary ---
    print(f'\n\n{"="*75}')
    print(f'  FINAL MODEL COMPARISON — {dataset_name}')
    print(f'{"="*75}')
    print(f'  {"Model":<22} {"Accuracy":>10} {"F1":>10} {"ROC-AUC":>10} {"PR-AUC":>10} {"Thresh":>8}')
    print(f'  {"-"*72}')
    for r in results:
        print(f'  {r["name"]:<22} {r["accuracy"]:>10.4f} {r["f1"]:>10.4f} '
              f'{r["roc_auc"]:>10.4f} {r["pr_auc"]:>10.4f} {r["threshold"]:>8.2f}')

    best = max(results, key=lambda x: x['f1'])
    print(f'\n  ★ Best model by F1: {best["name"]} (F1={best["f1"]:.4f}, Acc={best["accuracy"]:.4f})')

    return results


if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    out_file = os.path.join(data_dir, 'enhanced_results.txt')

    import io

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    log_f = open(out_file, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_f)

    print('=' * 75)
    print('  TCP PACKET LOSS PREDICTION — ENHANCED PIPELINE')
    print('  Network conditions + Advanced features + Optuna + CatBoost + Stacking')
    print('=' * 75)

    all_results = {}

    # Combined dataset
    combined_path = os.path.join(data_dir, 'enhanced_dataset.csv')
    if os.path.exists(combined_path):
        print('\n\n>>> COMBINED DATASET (Reno + Cubic)')
        X, y = load_and_prepare(combined_path)
        all_results['combined'] = train_and_evaluate(
            X, y, data_dir, 'Combined', use_optuna=True, n_trials=30
        )

    # Per-CC datasets
    for cc in ['reno', 'cubic']:
        cc_path = os.path.join(data_dir, f'{cc}_enhanced.csv')
        if os.path.exists(cc_path):
            cc_df = pd.read_csv(cc_path)
            if len(cc_df) > 100:
                print(f'\n\n>>> {cc.upper()} DATASET')
                X_cc, y_cc = load_and_prepare(cc_path)
                all_results[cc] = train_and_evaluate(
                    X_cc, y_cc, data_dir, cc.upper(), use_optuna=False
                )

    # Save JSON results
    json_path = os.path.join(data_dir, 'enhanced_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n\nResults saved to:')
    print(f'  {out_file}')
    print(f'  {json_path}')

    log_f.close()
    sys.stdout = sys.__stdout__
    print(f'\nDone! Results saved to {out_file}')

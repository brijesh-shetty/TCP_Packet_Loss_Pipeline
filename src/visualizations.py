#!/usr/bin/env python3
"""visualizations.py — Generate all paper-ready figures for TCP Packet Loss Prediction.

Generates:
  Fig 1: Model Comparison — F1, Accuracy, ROC-AUC, PR-AUC bar chart
  Fig 2: Feature Importance — Top-15 horizontal bar chart (LightGBM)
  Fig 3: Fold-wise Performance — F1 stability across 7 folds
  Fig 4: Class Distribution — Loss vs No-Loss counts
  Fig 5: Feature Correlation Heatmap — Top-20 features
  Fig 6: Confusion Matrices — Grid for all models
  Fig 7: CC Algorithm Comparison — Reno vs Cubic performance
"""
import pandas as pd
import numpy as np
import os, sys, json, warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('[ERROR] matplotlib required: pip install matplotlib')
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print('[WARN] seaborn not installed, using matplotlib defaults')


def setup_style():
    """Configure publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    if HAS_SNS:
        sns.set_style('whitegrid')
        sns.set_context('paper', font_scale=1.1)


# Professional color palette
COLORS = {
    'GradientBoosting': '#E74C3C',
    'RandomForest': '#2ECC71',
    'XGBoost': '#3498DB',
    'LightGBM': '#9B59B6',
    'CatBoost': '#F39C12',
    'StackingEnsemble': '#1ABC9C',
    'KNN': '#E67E22',
    'LogisticRegression': '#95A5A6',
    'DecisionTree': '#34495E',
    'NaiveBayes': '#BDC3C7',
}


def fig_model_comparison(results, fig_dir, dataset_name='Combined'):
    """Bar chart comparing all models across F1, Accuracy, ROC-AUC, PR-AUC."""
    models = [r['name'] for r in results]
    metrics = ['f1', 'accuracy', 'roc_auc', 'pr_auc']
    labels = ['F1-Score', 'Accuracy', 'ROC-AUC', 'PR-AUC']

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(models))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    bar_colors = ['#3498DB', '#2ECC71', '#9B59B6', '#F39C12']

    for i, (metric, label, color) in enumerate(zip(metrics, labels, bar_colors)):
        values = [r[metric] for r in results]
        bars = ax.bar(x + offsets[i] * width, values, width, label=label,
                      color=color, edgecolor='white', linewidth=0.8, alpha=0.85)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6.5,
                    fontweight='bold', rotation=45)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance Comparison — {dataset_name} Dataset',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0.85, 1.02)
    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Highlight best model
    best = max(results, key=lambda r: r['f1'])
    best_idx = models.index(best['name'])
    ax.annotate(f'★ Best: {best["name"]}\nF1={best["f1"]:.4f}',
                xy=(best_idx, best['f1']), xytext=(best_idx + 1.5, 0.88),
                fontsize=9, fontweight='bold', color='#9B59B6',
                arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1.5))

    plt.tight_layout()
    path = os.path.join(fig_dir, f'model_comparison_{dataset_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ Model comparison: {path}')


def fig_feature_importance(csv_path, fig_dir, dataset_name='Combined'):
    """Horizontal bar chart of top-15 feature importances from LightGBM."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold

    try:
        import lightgbm as lgb
    except ImportError:
        print('  [SKIP] LightGBM not installed, skipping feature importance')
        return

    df = pd.read_csv(csv_path)
    leak = ['lost', 'retrans_now', 'retrans_total', 'retrans_diff',
            'bytes_retrans', 'sacked', 'mss', 'pmtu', 'rcv_space']
    X = df.drop(columns=['label'] + [c for c in leak if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['label']

    model = lgb.LGBMClassifier(n_estimators=400, max_depth=7, is_unbalance=True,
                                random_state=42, n_jobs=-1, verbose=-1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model.fit(X_s, y)

    importances = model.feature_importances_
    idx = np.argsort(importances)[-15:]  # Top 15

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 15))

    ax.barh(range(15), importances[idx], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(15))
    ax.set_yticklabels([X.columns[i] for i in idx])
    ax.set_xlabel('Feature Importance (split count)')
    ax.set_title(f'Top-15 Feature Importances — {dataset_name}',
                 fontsize=13, fontweight='bold')

    # Add value labels
    for i, (val, name) in enumerate(zip(importances[idx], [X.columns[j] for j in idx])):
        ax.text(val + max(importances) * 0.01, i, f'{int(val)}',
                va='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(fig_dir, f'feature_importance_{dataset_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ Feature importance: {path}')


def fig_fold_stability(results, fig_dir, dataset_name='Combined'):
    """Line chart showing F1 score across 7 folds per model."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        if 'fold_metrics' not in r:
            continue
        folds = [fm['fold'] for fm in r['fold_metrics']]
        f1s = [fm['f1'] for fm in r['fold_metrics']]
        color = COLORS.get(r['name'], '#333333')
        ax.plot(folds, f1s, 'o-', label=r['name'], color=color,
                linewidth=1.5, markersize=5, alpha=0.85)

    ax.set_xlabel('Fold Number')
    ax.set_ylabel('F1-Score')
    ax.set_title(f'Cross-Validation Stability — {dataset_name}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.set_xticks(range(1, 8))
    ax.set_ylim(0.88, 1.0)
    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(fig_dir, f'fold_stability_{dataset_name.lower()}.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ Fold stability: {path}')


def fig_class_distribution(csv_path, fig_dir):
    """Bar chart showing Loss vs No-Loss distribution."""
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['No Loss (0)', 'Loss (1)'], counts.values,
                  color=['#2ECC71', '#E74C3C'], edgecolor='white', linewidth=2,
                  width=0.5)

    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution (Multi-Signal Consensus Labels)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.15)

    plt.tight_layout()
    path = os.path.join(fig_dir, 'class_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ Class distribution: {path}')


def fig_correlation_heatmap(csv_path, fig_dir):
    """Correlation heatmap of top-20 features."""
    if not HAS_SNS:
        print('  [SKIP] seaborn required for heatmap')
        return

    df = pd.read_csv(csv_path)
    leak = ['lost', 'retrans_now', 'retrans_total', 'retrans_diff',
            'bytes_retrans', 'sacked', 'mss', 'pmtu', 'rcv_space']
    X = df.drop(columns=['label'] + [c for c in leak if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Select top-20 features by variance
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(20).index.tolist()
    corr = X[top_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={'size': 7}, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap (Top-20 by Variance)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(fig_dir, 'correlation_heatmap.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ Correlation heatmap: {path}')


def fig_cc_comparison(fig_dir):
    """Compare Reno vs Cubic vs Combined performance."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # Try to read results from enhanced_results.txt
    results_file = os.path.join(data_dir, 'enhanced_results.txt')
    if not os.path.exists(results_file):
        print('  [SKIP] enhanced_results.txt not found for CC comparison')
        return

    # Extract best F1 per dataset from results text
    datasets = {'Combined': {}, 'Reno': {}, 'Cubic': {}}
    current_dataset = None

    with open(results_file, 'r') as f:
        for line in f:
            if '>>> COMBINED' in line.upper():
                current_dataset = 'Combined'
            elif '>>> RENO' in line.upper():
                current_dataset = 'Reno'
            elif '>>> CUBIC' in line.upper():
                current_dataset = 'Cubic'
            elif '★ Best model by F1' in line and current_dataset:
                import re
                m = re.search(r'F1=([\d.]+), Acc=([\d.]+)', line)
                if m:
                    datasets[current_dataset] = {
                        'f1': float(m.group(1)),
                        'acc': float(m.group(2))
                    }

    if not all(datasets.values()):
        print('  [SKIP] Incomplete CC comparison data')
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    names = list(datasets.keys())
    f1s = [datasets[n].get('f1', 0) for n in names]
    accs = [datasets[n].get('acc', 0) for n in names]

    x = np.arange(len(names))
    width = 0.3

    bars1 = ax.bar(x - width/2, f1s, width, label='F1-Score',
                   color='#3498DB', edgecolor='white')
    bars2 = ax.bar(x + width/2, accs, width, label='Accuracy',
                   color='#2ECC71', edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'{bar.get_height():.4f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Congestion Control Algorithm Comparison (Best Model: LightGBM)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Combined\n(Reno + Cubic)', 'TCP Reno\nOnly', 'TCP Cubic\nOnly'])
    ax.legend()
    ax.set_ylim(0.93, 0.985)

    plt.tight_layout()
    path = os.path.join(fig_dir, 'cc_algorithm_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ CC comparison: {path}')


def fig_labeling_comparison(csv_path, fig_dir):
    """Show base paper naive vs enhanced multi-signal labeling."""
    df = pd.read_csv(csv_path)

    # If 'lost' column exists, compare naive vs consensus labels
    if 'lost' not in df.columns:
        print('  [SKIP] No "lost" column for labeling comparison')
        return

    naive_loss = (df['lost'] > 0).sum()
    consensus_loss = (df['label'] == 1).sum()
    total = len(df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Naive labeling
    naive_counts = [total - naive_loss, naive_loss]
    axes[0].bar(['No Loss', 'Loss'], naive_counts,
                color=['#2ECC71', '#E74C3C'], edgecolor='white', width=0.5)
    axes[0].set_title('Base Paper: Naive Labeling\n(lost > 0)', fontsize=11, fontweight='bold')
    for i, v in enumerate(naive_counts):
        axes[0].text(i, v + 200, f'{v:,}\n({v/total*100:.1f}%)',
                     ha='center', fontweight='bold')

    # Consensus labeling
    cons_counts = [total - consensus_loss, consensus_loss]
    axes[1].bar(['No Loss', 'Loss'], cons_counts,
                color=['#2ECC71', '#E74C3C'], edgecolor='white', width=0.5)
    axes[1].set_title('Our Method: Multi-Signal Consensus\n(≥2/3 signals)', fontsize=11, fontweight='bold')
    for i, v in enumerate(cons_counts):
        axes[1].text(i, v + 200, f'{v:,}\n({v/total*100:.1f}%)',
                     ha='center', fontweight='bold')

    plt.suptitle('Labeling Strategy Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'labeling_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ Labeling comparison: {path}')


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 75)
    print('  PAPER VISUALIZATIONS — TCP Packet Loss Prediction')
    print('  Generating publication-quality figures')
    print('=' * 75)

    setup_style()

    csv_path = os.path.join(data_dir, 'enhanced_dataset.csv')
    json_path = os.path.join(data_dir, 'enhanced_results.json')

    # Load results
    results = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)
        # Handle both dict and list formats
        if isinstance(results, dict):
            if 'combined' in results:
                results = results['combined']
            else:
                results = list(results.values())[0] if results else []

    print(f'\n  Dataset: {csv_path}')
    print(f'  Results: {json_path}')
    print(f'  Output:  {fig_dir}/')
    print()

    # Generate each figure
    if results and isinstance(results, list):
        fig_model_comparison(results, fig_dir, 'Combined')
        fig_fold_stability(results, fig_dir, 'Combined')
    else:
        print('  [SKIP] No results JSON found for model comparison/fold charts')

    if os.path.exists(csv_path):
        fig_feature_importance(csv_path, fig_dir, 'Combined')
        fig_class_distribution(csv_path, fig_dir)
        fig_correlation_heatmap(csv_path, fig_dir)
        fig_labeling_comparison(csv_path, fig_dir)

    fig_cc_comparison(fig_dir)

    print(f'\n{"="*75}')
    print(f'  All figures saved to: {fig_dir}/')
    print(f'{"="*75}')


if __name__ == '__main__':
    main()

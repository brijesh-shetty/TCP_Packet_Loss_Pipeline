# Proactive TCP Congestion Control Using ML-Driven Artificial ECN Signals

LightGBM predicts impending TCP congestion from live `ss` state and drives **proportional RED/ECN** queue adjustments in Mininet, evaluated on TCP Reno and Cubic. Extends Welzl et al., *"Real-Time TCP Packet Loss Prediction Using Machine Learning"* (IEEE Access, 2024).

**Headline results** (verified against files in `results/`):
- LightGBM **F1 = 0.9723, ROC-AUC = 0.9965** on the 57,808-sample combined Reno+Cubic dataset.
- Mininet deployment: **+15.5% throughput** (Reno), **755 → 0 retransmissions** (Cubic).

---

## Repository Layout

```
TCP_ECN_Project/
├── src/                    # Data parsing, feature engineering, model training
│   ├── enhanced_parser.py      # Raw ss .txt → feature-engineered CSV (71 feats, 4 label strategies)
│   ├── enhanced_train.py       # 6-model training, 7-fold TimeSeriesSplit, SMOTE, Optuna
│   ├── parse_combined_ecn.py   # Parser for the combined ECN capture
│   ├── ablation_study.py       # Feature-group / labeling ablations
│   ├── check_dataset.py        # Dataset sanity checks
│   ├── artificial_ecn.py       # Offline ECN-response simulation
│   ├── visualizations.py       # Paper figures (model comparison, feature importance, etc.)
│   ├── trace_labels.py         # Labeling helpers
│   └── main.py                 # Entry point: --parse / --train / --all
│
├── deployment/             # Real-time Mininet ECN controller (Linux only)
│   ├── run_experiment.py       # Builds topology, runs baseline / binary-ECN / proportional-ECN
│   ├── ml_ecn_controller.py    # 20 ms poll → LightGBM inference → tc qdisc RED adjustment
│   ├── ecn_topology.py         # Dumbbell Mininet topology
│   ├── export_model.py         # Re-export trained model to joblib
│   ├── compare_results.py      # Aggregates the 3 experiment conditions
│   ├── setup_ubuntu.sh         # Environment setup
│   └── models/                 # lgbm_model.joblib, xgb_model.joblib, scaler.joblib, config.json
│
├── data/
│   ├── enhanced_dataset.csv     # PRIMARY training set — 57,808 rows (Reno+Cubic) ★
│   ├── enhanced_reno.csv        # Reno split
│   ├── enhanced_cubic.csv       # Cubic split
│   └── raw_experiments/         # ss -tin captures (provenance for enhanced_dataset.csv)
│
├── results/
│   ├── enhanced_results.json    # Model metrics → paper Tables I/II/IV
│   ├── enhanced_results.txt     # Full training log
│   ├── all_results.json         # Deployment metrics → paper Tables V–VIII
│   ├── {reno,cubic}_{baseline,basepaper_ecn,our_ecn}_*.json / *.npz   # Raw traces
│   └── figures/                 # ML figures + reno/ and cubic/ deployment figures
│
├── paper/
│   ├── final_paper.pdf          # Submitted paper
│   ├── base_paper.pdf           # Welzl et al. 2024 (reference)
│   ├── paper_part1.tex, paper_part2.tex, figure_inserts.tex
│   └── paper_issues.md          # Known fixes pending (duplicate table, etc.)
│
├── requirements.txt
├── run.bat
└── .gitignore
```

★ `enhanced_dataset.csv` is the set behind every reported model number. The 1.28M-row
`large_dataset.csv` from the earlier exploration is **not** used in the paper and is not included.

---

## Quick Start

### Inference / training (any OS)
```bash
pip install -r requirements.txt

# Train all models on the primary dataset (no Optuna ≈ fast)
python src/main.py --train --dataset data/enhanced_dataset.csv --no-optuna

# Full pipeline (re-parse raw captures, then train)
python src/main.py --all --no-optuna
```

### Real-time Mininet deployment (Linux + Mininet required)
```bash
cd deployment
sudo python3 run_experiment.py        # runs baseline / binary-ECN / proportional-ECN
python3 compare_results.py            # aggregates into all_results.json
```

---

## Method Summary

1. **Leak-free labeling** — multi-signal consensus (2-of-3: CWND drop ≥30%, ssthresh engaged,
   retransmission), yielding a naturally balanced 48/52 split. All leakage columns dropped before training.
2. **Features** — 71 engineered (rolling stats, differentials, lags, ratios, interactions) → 45 retained
   after variance + correlation (>0.98) filtering.
3. **Model** — LightGBM, 7-fold TimeSeriesSplit, BorderlineSMOTE on train folds, Optuna (XGB+LGBM only).
   Train threshold 0.38, re-calibrated to **0.56** for deployment.
4. **Proportional ECN** — every 20 ms, `factor = clamp(1 − (conf − 0.3)×1.2, 0.2, 0.7)` scales RED
   min/max thresholds; restored after ~100 ms.

## Reference
M. Welzl, S. Islam, M. von Stephanides, "Real-Time TCP Packet Loss Prediction Using Machine Learning,"
*IEEE Access*, vol. 12, pp. 159622–159634, 2024. DOI: 10.1109/ACCESS.2024.3488511

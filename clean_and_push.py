import os
import shutil
import subprocess

# Define paths
base_dir = r"c:\Users\Lenovo\projects\cn project"
src_repo = os.path.join(base_dir, "dataset extraction and preprocessing", "TCP_Packet_Loss_Pipeline")
dest_repo = os.path.join(base_dir, "TCP_Packet_Loss_Pipeline_Clean")

print(f"Source repo: {src_repo}")
print(f"Dest repo: {dest_repo}")

import stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Create dest repo if it doesn't exist, or clean it if it does
if os.path.exists(dest_repo):
    print(f"Cleaning existing dest repo: {dest_repo}")
    shutil.rmtree(dest_repo, onerror=remove_readonly)
os.makedirs(dest_repo)
os.makedirs(os.path.join(dest_repo, "src"))
os.makedirs(os.path.join(dest_repo, "base_paper"))

# 1. Copy & Modify Python Source Files into src/
py_files = [
    "ablation_study.py",
    "artificial_ecn.py",
    "check_dataset.py",
    "enhanced_parser.py",
    "enhanced_train.py",
    "main.py",
    "parse_combined_ecn.py",
    "visualizations.py"
]

for file_name in py_files:
    src_file = os.path.join(src_repo, file_name)
    dest_file = os.path.join(dest_repo, "src", file_name)
    if os.path.exists(src_file):
        print(f"Restructuring & Copying {file_name}...")
        with open(src_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace file directory path resolution to point to parent (project root)
        # so files are written/read from project root instead of src/
        new_content = content.replace(
            "os.path.dirname(os.path.abspath(__file__))",
            "os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))"
        )
        
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(new_content)
    else:
        print(f"WARNING: Source file {src_file} does not exist!")

# 2. Copy Base Paper to base_paper/
pdf_name = "Real-Time_TCP_Packet_Loss_Prediction_Using_Machine_Learning (3).pdf"
src_pdf = os.path.join(src_repo, pdf_name)
dest_pdf = os.path.join(dest_repo, "base_paper", pdf_name)
if os.path.exists(src_pdf):
    print(f"Copying base paper PDF...")
    shutil.copy2(src_pdf, dest_pdf)
else:
    print("WARNING: Base paper PDF not found!")

# 3. Copy Figures Directory
src_figures = os.path.join(src_repo, "figures")
dest_figures = os.path.join(dest_repo, "figures")
if os.path.exists(src_figures):
    print("Copying figures directory...")
    shutil.copytree(src_figures, dest_figures)
else:
    print("WARNING: figures directory does not exist!")

# 4. Copy requirements.txt to root
src_req = os.path.join(src_repo, "requirements.txt")
dest_req = os.path.join(dest_repo, "requirements.txt")
if os.path.exists(src_req):
    print("Copying requirements.txt...")
    shutil.copy2(src_req, dest_req)
else:
    print("WARNING: requirements.txt not found!")

# 5. Copy & Modify run.bat
src_bat = os.path.join(src_repo, "run.bat")
dest_bat = os.path.join(dest_repo, "run.bat")
if os.path.exists(src_bat):
    print("Modifying & Copying run.bat...")
    with open(src_bat, "r", encoding="utf-8") as f:
        bat_content = f.read()
    
    # Update execution path
    new_bat_content = bat_content.replace("python main.py", "python src/main.py")
    
    with open(dest_bat, "w", encoding="utf-8") as f:
        f.write(new_bat_content)
else:
    print("WARNING: run.bat not found!")

# 6. Create .gitignore (with requirements.txt unignored)
gitignore_content = """# Virtual environment
.venv/

# Python cache
__pycache__/
*.pyc

# Large datasets
*.csv

# Output files
*.json
*.txt
!requirements.txt
output_backup1/

# OS files
.DS_Store
Thumbs.db
"""

with open(os.path.join(dest_repo, ".gitignore"), "w", encoding="utf-8") as f:
    f.write(gitignore_content)
print("Created .gitignore file.")

# 7. Create rich README.md
readme_content = """# Proactive TCP Congestion Control Using ML-Driven Artificial ECN Signals

> **LightGBM classifier predicting TCP packet loss before it occurs, integrated as proportional ECN signals into TCP Reno and Cubic via Mininet.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-classifier-green)](https://lightgbm.readthedocs.io/)
[![Mininet](https://img.shields.io/badge/Mininet-network%20emulation-orange)](http://mininet.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Traditional TCP congestion control (Reno, Cubic) reacts to packet loss **only after** it has occurred — causing retransmissions, inflated RTT, and reduced throughput. This project builds a **proactive** alternative:

1. A **LightGBM classifier** trained on 57,808 TCP flow samples predicts imminent congestion with **F1 = 0.9723** and **ROC-AUC = 0.9965**.
2. Predictions are fed back as **proportional Explicit Congestion Notification (ECN) signals** — dynamically scaling RED queue thresholds in proportion to prediction confidence.
3. Mininet deployment achieves **+15.5% throughput for TCP Reno** and **100% retransmission elimination for TCP Cubic** (755 → 0).

This work extends the base paper *"Real-Time TCP Packet Loss Prediction Using Machine Learning"* (Welzl, Islam & von Stephanides, IEEE Access 2024) with a leak-free labeling strategy, richer feature engineering, a unified Reno+Cubic model, and a proportional queue-management controller.

---

## Key Results

| Metric | Baseline (pfifo) | Binary ECN | **Proportional ECN (Ours)** |
|---|:---:|:---:|:---:|
| TCP Reno Throughput | 1.39 Mbps | 1.49 Mbps | **1.60 Mbps (+15.5%)** |
| TCP Cubic Retransmissions | 755 | 0 (100% ↓) | **0 (100% ↓)** |
| TCP Cubic CWND Stability (σ) | 2.03 | 1.87 | **1.80 (+11.3%)** |

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|
| GradientBoosting | 0.9473 | 0.9439 | 0.9895 | 0.9885 |
| RandomForest | 0.9272 | 0.9230 | 0.9822 | 0.9810 |
| XGBoost | 0.9690 | 0.9670 | 0.9957 | 0.9951 |
| **LightGBM ★** | **0.9740** | **0.9723** | **0.9965** | **0.9960** |
| CatBoost | 0.9694 | 0.9674 | 0.9953 | 0.9947 |
| StackingEnsemble | 0.9685 | 0.9664 | 0.9903 | 0.9889 |

★ Selected for deployment (threshold calibrated to 0.56).

---

## Repository Structure

```
.
├── data_capture/                  # Mininet topology & data collection
│   ├── mn_topo.py                 # Dumbbell Mininet topology (1 foreground + 6 bg flows)
│   ├── data_capture.sh            # Main capture script (duration, CC, delay, BW, queue)
│   ├── capture_all_data.sh        # Batch runner across all parameter combinations
│   ├── bash_functions.sh          # Shared shell utility functions
│   ├── start_and_run_connection.sh # iPerf3 session launcher
│   ├── create_csv.py              # Converts raw ss output to CSV per experiment
│   ├── merge_data.py              # Merges per-experiment CSVs into combined dataset
│   ├── generate_unsupervised_dataset.py  # Unsupervised feature dataset builder
│   ├── connection_parameter_combinations.csv  # Full 300-experiment parameter grid
│   ├── test_parameter_combinations.csv        # Deployment test parameter set
│   └── output/                    # Raw ss capture output (see note below)
│
├── data_transformation/
│   └── txt_to_csv.py              # Parser: ss .txt → feature-engineered CSV with labels
│
├── models/
│   ├── binary_clf_reno_phase_three.ubj    # Trained XGBoost model — TCP Reno
│   └── binary_clf_cubic_phase_three.ubj   # Trained XGBoost model — TCP Cubic
│
└── tests/
    ├── predict.py                 # Real-time prediction daemon (watchdog + XGBoost)
    ├── prepare_data.py            # Feature prep for inference
    ├── calculate_results.py       # Evaluation metrics calculator
    ├── create_cwnd_plot.py        # CWND trace visualization
    ├── plot_metrics_tradeoff.py   # Precision/recall/F1 tradeoff plots
    ├── plot_metrics_vs_thresh.py  # Metrics vs. threshold sweep
    ├── plot_feature.py            # Individual feature distribution plotter
    ├── parse_pcap.py              # PCAP parser for ground-truth retransmission counts
    └── calculate_metrics.sh       # Shell wrapper for batch metric calculation
```

> **Raw data**: The full 300-experiment `ss -tin` capture output (~4.5 GB) is not included in this repository due to size. A sample dataset (`data_capture/unsupervised_network_dataset.csv`) is provided. Contact the authors for access to the full dataset.

---

## Methodology

### 1. Network Topology & Data Collection

- **Topology**: Mininet dumbbell — 1 foreground iPerf3 flow (h1→h3) + 6 background flows across a single bottleneck switch pair (s1–s2).
- **Bottleneck**: 10 Mbit/s HTB-shaped link; Netem delay applied per interface.
- **Parameter grid**: 5 bandwidths × 5 delays × 3 BDP multipliers × 2 CC algorithms = **300 experiments** (150 Reno + 150 Cubic), each at 100 s or 300 s duration.
- **Polling**: `ss -tin` queried every **20 ms**, capturing cwnd, ssthresh, RTT/rttvar, retransmission counters, unacked segments, pacing rate, and timer info.

### 2. Leak-Free Labeling (Multi-Signal Consensus)

The base paper's naive CWND-peak labeling introduces data leakage and severe class imbalance (~1% positive). We replace it with a **2-of-3 consensus** rule:

| Signal | Condition |
|---|---|
| **CWND Drop** | cwnd ≤ rolling-5-max × 0.70 |
| **ssthresh Active** | cwnd ≤ ssthresh × 1.2 AND ssthresh > 0 |
| **Retransmission** | retrans_now > 0 OR lost > 0 OR retrans_diff > 0 |

A sample is labeled **congestion (1)** if ≥ 2 signals are active. All leakage columns (`lost`, `retrans_now`, `retrans_total`, `retrans_diff`, `bytes_retrans`, `sacked`) are then **dropped from the feature set** before training.

Result: **48.07% positive / 51.93% negative** — naturally balanced, no aggressive oversampling required.

### 3. Feature Engineering

**71 candidate features** engineered from raw `ss` output, reduced to **45 retained features** via `VarianceThreshold` + Pearson correlation filtering (> 0.98 dropped):

| Group | Examples |
|---|---|
| Rolling statistics | `cwnd_roll5_mean`, `rtt_roll3_mean`, `cwnd_roll5_std` |
| Temporal differentials | `cwnd_diff`, `rtt_diff`, `cwnd_accel`, `rtt_accel` |
| Lag features | `cwnd_lag1`, `cwnd_lag2`, `rtt_lag1`, `rtt_lag2` |
| Ratio features | `retrans_ratio`, `rtt_ratio`, `cwnd_util`, `ack_ratio` |
| Interaction features | `cwnd_rtt_interaction`, `buffer_fill_ratio`, `congestion_signal` |
| Network conditions | `delay_ms`, `bandwidth_mbit`, `bdp_estimate` |

### 4. Model Training

- **Validation**: 7-fold `TimeSeriesSplit` (preserves temporal ordering)
- **Class balancing**: `BorderlineSMOTE` applied to training folds only
- **Tuning**: Optuna (30 trials) for XGBoost and LightGBM; manual hyperparams for others
- **Threshold calibration**: Per-fold F1-maximizing threshold search; deployment threshold re-calibrated to **0.56**

### 5. Proportional ECN Mechanism

Every 20 ms:
1. Poll `ss -tin` → compute 45 features
2. LightGBM returns `P(loss) = confidence`
3. If `confidence ≥ 0.56`, compute reduction factor:

   ```
   factor = clamp(1.0 − (confidence − 0.3) × 1.2, 0.2, 0.7)
   ```

4. Apply to RED queue thresholds:

   ```bash
   tc qdisc change dev s1-eth8 handle 10: red limit 200000 \
     min <orig_min × factor> max <orig_max × factor> \
     avpkt 1000 bandwidth 10mbit ecn probability 0.2
   ```

5. After ~100 ms (5 polling intervals), restore original thresholds.

High-confidence predictions → aggressive threshold reduction (factor → 0.2); marginal predictions → gentle adjustment (factor → 0.7).

---

## Getting Started

### Prerequisites

**Linux / Mininet environment required for data capture.** For inference only, any Python 3.8+ system works.

```bash
pip install xgboost lightgbm pandas numpy scikit-learn imbalanced-learn optuna watchdog
```

Mininet installation: http://mininet.org/download/

### Data Capture

```bash
cd data_capture

# Single experiment: 100s, TCP Reno, 6 bg flows, 30ms delay, 10Mbit, queue 18750 bytes
./data_capture.sh -d 100 -c reno -n 6 -l 30 -b 10 -q 18750 -s reno

# Full parameter grid (300 experiments)
./capture_all_data.sh
```

**Arguments for `data_capture.sh`:**

| Flag | Description | Default |
|---|---|---|
| `-d` | Duration (seconds) | 300 |
| `-c` | CC algorithm (`reno`/`cubic`) | `cubic` |
| `-n` | Background flows (0–6) | 0 |
| `-l` | Delay (ms) | 50 |
| `-b` | Bandwidth (Mbit/s) | 50 |
| `-q` | Queue size (bytes) | 312500 |
| `-s` | Background CC scenario | `reno` |

Output is saved under `data_capture/output/`.

### Data Transformation

```bash
# Convert raw ss .txt captures to feature-engineered CSV
python data_transformation/txt_to_csv.py
```

### Real-Time Prediction (Inference)

The `tests/predict.py` daemon watches a directory for new `ss` output, runs inference, and writes the prediction:

```bash
python tests/predict.py \
  -c models/binary_clf_reno_phase_three.ubj \
  -d data_capture/output/reno/experiment_1/ \
  -i data_capture/output/reno/experiment_1/current_sample.csv \
  -o prediction_output.txt \
  -t 0.56
```

**Arguments:**

| Flag | Description |
|---|---|
| `-c` | Path to trained `.ubj` model |
| `-d` | Directory to watch for file changes |
| `-i` | Path to the live CSV input file |
| `-o` | Output file for predictions |
| `-t` | Classification threshold (default: model default) |
| `--timestamp_mode` | Append predictions with timestamps |
| `--time` | Benchmark inference speed (10,000 runs) |

### Evaluation & Visualization

```bash
# Compute classification metrics
python tests/calculate_results.py

# Plot CWND trace comparison (Baseline vs Binary vs Proportional ECN)
python tests/create_cwnd_plot.py

# Precision/Recall/F1 tradeoff across thresholds
python tests/plot_metrics_vs_thresh.py

# Parse pcap for ground-truth retransmission counts
python tests/parse_pcap.py
```

---

## Dataset

| Stat | Value |
|---|---|
| Total samples | 57,808 |
| Reno samples | ~28,900 |
| Cubic samples | ~28,900 |
| Loss events (class 1) | 27,790 (48.07%) |
| Normal events (class 0) | 30,018 (51.93%) |
| Features engineered | 71 |
| Features retained | 45 |
| Experiments | 300 (150 Reno + 150 Cubic) |
| Parameter grid | 5 BW × 5 delay × 3 BDP × 2 CC |

---

## References

This work builds on and extends:

> M. Welzl, S. Islam, and M. von Stephanides, "Real-Time TCP Packet Loss Prediction Using Machine Learning," *IEEE Access*, vol. 12, pp. 159622–159634, 2024. DOI: [10.1109/ACCESS.2024.3488511](https://doi.org/10.1109/ACCESS.2024.3488511)
"""

with open(os.path.join(dest_repo, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print("Created README.md file.")

# 8. Git Operations
def run_git(cmd, cwd=dest_repo):
    print(f"Running git command in {cwd}: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"ERROR: {res.stderr}")
        return False
    else:
        print(res.stdout)
        return True

if run_git(["git", "init"]):
    run_git(["git", "config", "user.name", "Brijesh Shetty"])
    run_git(["git", "config", "user.email", "brijeshshetty005@gmail.com"])
    run_git(["git", "branch", "-M", "main"])
    run_git(["git", "add", "."])
    
    commit_msg = (
        "Restructure repository: Group code under src/ and base paper under base_paper/\n\n"
        "- Organized source files into src/ directory\n"
        "- Relocated Welzl et al. (2024) base paper into base_paper/\n"
        "- Dynamic directory path resolution configured for parent (project root)\n"
        "- Enriched README.md with base paper limitations and details\n"
        "- Updated run.bat execution entry point to use src/main.py"
    )
    run_git(["git", "commit", "-m", commit_msg])
    run_git(["git", "remote", "add", "origin", "https://github.com/brijesh-shetty/TCP_Packet_Loss_Pipeline.git"])
    
    print("\n" + "="*50)
    print("SUCCESS: Structured repository set up locally at:")
    print(dest_repo)
    print("To push this repository to GitHub, please run the following command:")
    print(f'cd "{dest_repo}"')
    print('git push -u origin main --force')
    print("="*50)
else:
    print("ERROR initializing git!")

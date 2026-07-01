# Project Explained — End to End

Proactive TCP Congestion Control Using ML-Driven Artificial ECN Signals.
Base paper → implementation → improvements → results, with an honest assessment.
(Written as viva/interview prep and repo documentation.)

---

## 1. The Base Paper

**Welzl, Islam & von Stephanides, "Real-Time TCP Packet Loss Prediction Using Machine Learning," IEEE Access 2024.**

**Core idea:** TCP (Reno/Cubic) is *reactive* — it backs off only *after* packets drop. If loss can be *predicted* a moment earlier, the sender can slow down and avoid the drop. They showed ML can do this prediction in real time.

**What they did:**
- Emulated Reno and Cubic flows in **Mininet**; polled TCP state with the Linux `ss` tool.
- **Labeling:** a **cwnd-peak rule** — if cwnd rises then drops next step (`cwnd[i-1] ≤ cwnd[i]` and `cwnd[i+1] < cwnd[i]`), label that peak as pre-loss. Produced **~1% positive class** of ~14M samples — extreme imbalance.
- **Models:** gradient-boosted trees (XGBoost, Random Forest).
- **Results:** XGBoost **F1 ≈ 0.28–0.40** — modest, due to the imbalance and noisy labeling.
- **Actuation:** predictions fed back as **binary ECN** signals at the source.

**Their repo:** `data_capture/` (Mininet topology + `ss` capture + 300-experiment grid), `data_transformation/` (txt→CSV), `models/` (per-CC XGBoost `.ubj`), `tests/` (`predict.py` daemon, metric calculators, plots, pcap parser).

**Open limitations:** low F1, severe imbalance, leakage-prone labeling, separate per-algorithm models, binary-only ECN.

---

## 2. This Project's Implementation (end to end)

Five stages. Stages 1–4 build/validate the model; stage 5 deploys it live.

### Stage 1 — Data Collection
- **Topology:** Mininet **dumbbell** — one foreground iperf3 flow `h1→h3` across switches `s1–s2`, with **6 background flows** competing for the bottleneck.
- **Link:** 10 Mbit/s bottleneck, **HTB** shaping; **Netem** delay (15 ms/side → 30 ms RTT; varied 30–70 ms during capture).
- **Parameter grid:** 5 bandwidths (10–50 Mbit) × 5 delays (30–70 ms) × 3 BDP multipliers (0.5/1/2×) × 2 CC (Reno/Cubic) = **300 experiments** (150 Reno + 150 Cubic), at 100 s and 300 s.
- **Sampling:** `ss -tin` every **20 ms** → cwnd, ssthresh, RTT/rttvar, retransmission counters, unacked, bytes, pacing/delivery rate, timers.
- **Output:** primary **57,808-sample** dataset (`enhanced_dataset.csv`) + a larger **1.28 M-row** dataset.

### Stage 2 — Parsing & Feature Engineering (`src/enhanced_parser.py`)
- Parses raw `ss` text into structured per-experiment records, **grouped by flow** so rolling/diff features never leak across experiments.
- **71 candidate features** in five groups:
  - **Rolling stats** — `cwnd_roll5_mean/std`, `rtt_roll3_mean`, …
  - **Temporal differentials** — `cwnd_diff`, `rtt_diff`, `cwnd_accel`, `rtt_accel`
  - **Lags** — `cwnd_lag1/2`, `rtt_lag1/2`
  - **Ratios** — `retrans_ratio`, `cwnd_util`, `rtt_ratio`, `ack_ratio`
  - **Interactions / conditions** — `cwnd_rtt_interaction`, `buffer_fill_ratio`, `congestion_signal`, `delay_ms`, `bandwidth_mbit`, `bdp_estimate`
- Reduced to **45 features** via VarianceThreshold + Pearson correlation > 0.98 removal.

### Stage 3 — Leak-Free Labeling (the central contribution)
- Base paper labeling is leaky and ~1% positive. Replaced with **Multi-Signal Consensus**: "congestion (1)" if **≥2 of 3** signals fire:
  1. **CWND drop** — cwnd ≤ 70% of its rolling-5 max
  2. **ssthresh active** — cwnd ≤ ssthresh×1.2 and ssthresh > 0
  3. **Retransmission** — retrans/lost activity
- **All leakage columns dropped before training** (`lost`, `retrans_now/total/diff`, `bytes_retrans`, `sacked`) — model learns only from flow *dynamics*.
- Result: **naturally balanced 48% / 52%** — no aggressive oversampling.
- Three alternative labels also defined: `label_basepaper` (their rule), `label_retrans` (ground-truth loss), `label_predictive` (loss + next 1–2 steps = genuine early warning).

### Stage 4 — Model Training (`src/enhanced_train.py`)
- **8 classifiers** evaluated, **6 reported**: GradientBoosting, RandomForest, XGBoost, LightGBM, CatBoost, StackingEnsemble (Logistic-Regression meta). (KNN/DecisionTree/NaiveBayes dropped for low F1.)
- **7-fold TimeSeriesSplit** (preserves temporal order, prevents future leakage).
- **BorderlineSMOTE** on training folds only.
- **Optuna** (30 trials) for XGBoost + LightGBM; manual hyperparameters otherwise.
- **Per-fold F1-maximizing threshold calibration** (not fixed 0.5). Train threshold 0.38 → re-calibrated **0.56** for deployment.

### Stage 5 — Real-Time ECN Deployment (`deployment/ml_ecn_controller.py` + `run_experiment.py`)
A live **closed loop**:
1. Every 20 ms, poll `ss` for the foreground flow.
2. Compute the 45-feature vector from current sample + rolling history.
3. LightGBM returns `P(loss)`.
4. If `P ≥ 0.56`, **proportionally** lower the bottleneck's RED thresholds via `tc qdisc change` (`factor = clamp(1 − (conf − 0.3)×1.2, 0.4, 0.7)`) and raise the ECN marking probability → kernel marks packets **ECN-CE** → TCP reduces cwnd **before** the drop.
5. Restore thresholds after a cooldown.
- Compares **3 conditions**: baseline (pfifo, no ECN), base-paper (XGBoost @0.83, binary), ours (LightGBM @0.56, proportional).
- Hardened with **debounce** (require consecutive positive predictions), a **refractory window** (gap between ECN episodes), and a **gentler reduction floor (0.4)** so it stays stable instead of choking the flow to cwnd=2.

---

## 3. Improvements Over the Base Paper

| # | Base paper | This project |
|---|---|---|
| 1 | Leaky cwnd-peak labeling, ~1% positive | **Leak-free multi-signal consensus**, balanced 48/52 |
| 2 | Smaller feature set | **71 engineered → 45 retained**, 5 groups |
| 3 | Separate per-CC models | **Single unified Reno+Cubic model** |
| 4 | ~2 models, basic tuning | **8 models, Optuna + per-fold threshold calibration** |
| 5 | **Binary** ECN toggle | **Proportional** RED-threshold ECN, confidence-scaled |
| 6 | F1 0.28–0.40 | F1 0.97 (detection) / **0.87 (genuine prediction)** |
| 7 | — | **Cross-CC generalization** + **sub-2 ms** inference, measured |

---

## 4. Results (verified, with honest caveats)

### Model results — solid and verified
All match `results/enhanced_results.json`:

| Model | F1 | ROC-AUC |
|---|---|---|
| **LightGBM ★** | **0.9723** | **0.9965** |
| XGBoost | 0.9670 | 0.9957 |
| CatBoost | 0.9674 | 0.9953 |
| StackingEnsemble | 0.9664 | 0.9903 |
| GradientBoosting | 0.9439 | 0.9895 |
| RandomForest | 0.9230 | 0.9822 |

### Added rigor (validated, see `results/rigor_results.json` + `results/figures/rigor/`)
- **Detection vs prediction (honest framing):** congestion-state detection **F1 = 0.973**; genuine loss prediction **~40 ms ahead F1 = 0.872**. The 0.97 is *detection*; the honest *predictive* number is 0.87 — still far above the base paper's 0.28–0.40 on a harder task.
- **Cross-CC generalization:** train Reno→test Cubic **F1 = 0.94**; Cubic→Reno **F1 = 0.95**.
- **Inference latency:** **~1.3 ms mean, <2 ms p99** — negligible vs the 20 ms loop.
- **Figures:** ROC, PR, confusion matrix, calibration.

### Deployment results — honest status
- A single original run showed Reno **+15.5% throughput** and Cubic **755→0 retransmissions**.
- **Under repetition these do not replicate.** Throughput swings widely (−79% to +120%) because the foreground flow either sits stable (nothing to predict; 0 ECN signals fire) or overflows — depending on random background-flow timing. In a VM the 20 ms loop also slips (CPU-starved), adding noise.
- **Conclusion:** the deployment is a **proof-of-concept** that the real-time ML→ECN→kernel loop *works and is stable* — **not** a robust throughput benchmark. The "+15.5%" was a lucky single draw.

---

## 5. The End-to-End Story (one paragraph)

Traditional TCP reacts to loss only after it happens. This project builds a proactive alternative: 300 Mininet experiments generate 57,808 samples of live TCP state; a leak-free multi-signal consensus labeling scheme (the key contribution) produces a balanced dataset where the base paper's had ~1% positives; 71 engineered features are reduced to 45; a single LightGBM model — generalizing across both Reno and Cubic — detects congestion at F1 0.97 and predicts loss ~40 ms ahead at F1 0.87, at sub-2 ms inference; and a real-time controller closes the loop, feeding predictions as proportional ECN signals via Linux `tc`/RED to make TCP back off before drops occur. The ML and systems contributions are strong and validated; the deployment demonstrates the closed loop works, while honest repeated testing shows throughput gains are environment-dependent rather than guaranteed.

---

## 6. What is Solid vs. What is a Proof-of-Concept

**Strong, defensible (lead with these):**
- Leak-free labeling methodology (the real novelty).
- Unified model across Reno/Cubic, with cross-CC generalization evidence.
- Detection F1 0.97 / honest prediction F1 0.87 at a stated horizon.
- Sub-2 ms real-time inference.
- A working, stable real-time ML→ECN control loop.

**Proof-of-concept (state honestly):**
- Mininet ECN deployment demonstrates the closed loop functions; throughput/retransmission gains are environment-sensitive and not claimed as a robust benchmark.

**Viva one-liner:** *"My contribution is a leak-free labeling method and a unified, low-latency predictive model that generalizes across congestion-control algorithms, wired into a working real-time ECN controller. The model results are rigorous; the deployment is a validated proof-of-concept, and I'm upfront that the throughput numbers are environment-sensitive."*

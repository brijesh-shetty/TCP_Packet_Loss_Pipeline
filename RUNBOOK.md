# RUNBOOK — Strengthening the project (rigor + resume)

Everything here addresses the three open weaknesses:
1. **n=1 deployment** → repeat across configs, report mean ± std (Part B).
2. **throughput-metric fragility** → capture & report iperf goodput too (Part B).
3. **detection-vs-prediction framing of F1** → report a genuine early-warning F1 (Part A).

Run Part A anywhere (no Mininet). Run Part B on Ubuntu with Mininet.

---

## Setup

```bash
cd TCP_ECN_Project
python3 -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
pip install matplotlib joblib            # if not already pulled in
```

For Part B also install Mininet + iperf3:
```bash
sudo apt update && sudo apt install -y mininet iperf3
```

---

## PART A — ML rigor (no Mininet, ~10–20 min total)

### A1. Detection vs Prediction + figures + cross-CC
```bash
# Quick pass (sampled, ~3 min) to confirm it runs:
python3 src/rigor_eval.py --sample 50000

# Reported numbers — full large dataset for the labels stage (slower, more RAM):
python3 src/rigor_eval.py --stage labels  --sample 0
python3 src/rigor_eval.py --stage figures
python3 src/rigor_eval.py --stage crosscc
```
Produces:
- `results/rigor_results.json` — all numbers below in machine form.
- `results/figures/rigor/{roc,pr,confusion_matrix,calibration}.png`

What you get and where it goes:
| Output | Use in paper |
|---|---|
| `detection_vs_prediction` (consensus F1≈0.97 **and** predictive F1≈0.87 @ ~40 ms) | Add a row/sentence: report BOTH. Reword abstract/intro from "predict impending loss" so the 0.97 is called **detection** and the predictive horizon is stated explicitly. |
| `roc.png`, `pr.png` | New figure — standard, currently missing. |
| `confusion_matrix.png` | New figure — referenced but not shown. |
| `calibration.png` | Justifies the 0.56 deployment threshold (the 0.38→0.56 gap). |
| `cross_cc_generalization` (train Reno→test Cubic F1≈0.94, and vice-versa) | Turns "the unified model generalizes" from claim → evidence. Add one table. |

> The labels stage uses the contiguous head of `large_dataset.csv`. With `--sample 0`
> it uses everything (Reno+Cubic); a small sample is Reno-only, so only quote `--sample 0`
> numbers in the paper.

### A2. Inference latency
```bash
python3 src/benchmark_latency.py --n 20000
```
Gives mean/p50/p95/p99 single-sample latency (≈1–2 ms here). Bullet:
*"Low-millisecond inference — a small fraction of the 20 ms poll loop."*

---

## PART B — Deployment rigor (Ubuntu + Mininet, hours)

### B0. Make sure the models are present
`deployment/models/` already contains `lgbm_model.joblib`, `xgb_model.joblib`,
`scaler.joblib`, `config.json`. If you retrain, re-export with:
```bash
python3 deployment/export_model.py     # note: it should use label_consensus (see paper/paper_corrections.md #9)
```

### B1. Run the sweep (repetitions × configs)
```bash
cd deployment
chmod +x run_sweep.sh
sudo bash run_sweep.sh        # edit DURATION / REPS / CONFIGS at the top first
```
Default = 3 configs × 5 reps × (2 CC × 3 conditions) = 90 runs (~3 h). Each run’s
JSON/NPZ are moved into `deployment/sweep/<config>/rep<k>/` so nothing is overwritten.
Safe to re-run — it skips configs/reps already done.

### B2. Aggregate to mean ± std
```bash
python3 aggregate_sweep.py
```
Prints, per config × CC × condition: throughput (**ss-derived AND iperf goodput**),
retransmissions, CWND std, ECN signals — each mean ± std — plus the
`our_ecn vs baseline` throughput gain as **"+X% ± Y% over N reps"**.
Writes `deployment/sweep/sweep_summary.json`.

What you get and where it goes:
| Output | Use in paper |
|---|---|
| `+X% ± Y%` throughput gain over N reps | Replace the bare "+15.5%" in Tables V/VIII + abstract. |
| iperf goodput column | Report iperf as primary; fixes the metric-choice objection. If `our_ecn` iperf shows `n/a`, the warning tells you the `/tmp/iperf_result.json` capture failed — see note below. |
| multiple configs | Lets you honestly write "across N topologies", not one point. |

> **iperf capture for our_ecn:** in the single original run this came back empty.
> If the aggregator flags it missing, check that `iperf3` finishes before `net.stop()`
> in `run_experiment.py` (the post-loop `time.sleep(2)`); bump it if needed, and verify
> `cat /tmp/iperf_result.json` is non-empty after a manual run.

---

## Suggested order
1. **A1 + A2** today (no Mininet) → new figures, the detection-vs-prediction table, cross-CC, latency. These alone make the paper meaningfully more defensible.
2. Apply the wording fixes (`paper/paper_corrections.md`, `paper/paper_issues.md`): detection-vs-prediction reframe, duplicate Table II/IV, author names, threshold rationale.
3. **B1 + B2** when you have a Mininet box free overnight → statistical deployment results.
4. Submit to arXiv / a student venue. "Published / under review" is the biggest single resume multiplier.

## Resume bullet (after the above)
> Built a real-time ML-ECN congestion controller (LightGBM, ~1 ms inference inside a 20 ms
> poll loop) that **detects congestion at F1 0.97 and predicts loss ~40 ms ahead at F1 0.87**;
> over **N Mininet runs** it cut TCP Cubic retransmissions to zero and improved TCP Reno
> goodput **+X% ± Y%** vs. baseline. Unified model generalizes across Reno/Cubic (cross-CC F1 ≈ 0.94).

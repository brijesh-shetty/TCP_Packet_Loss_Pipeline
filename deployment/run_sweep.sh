#!/usr/bin/env bash
# run_sweep.sh — Repeat the 3-condition ECN experiment across configs AND repetitions,
# so the headline gains get mean ± std instead of a single n=1 number.
#
# Run on Ubuntu with Mininet installed, as root:
#     cd deployment
#     sudo bash run_sweep.sh
#
# Each (config × repetition) runs baseline / basepaper_ecn / our_ecn for BOTH Reno and
# Cubic, then its result files are moved into sweep/<config>/rep<k>/ so nothing is
# overwritten. Afterwards run:  python3 aggregate_sweep.py
#
# Tune the three knobs below. Default = 3 configs × 5 reps × (2 CC × 3 conditions)
# = 90 runs. At ~130 s/run that's ~3.3 h — start it before you leave.
set -u

DURATION=120          # seconds per condition (paper used 120)
REPS=5                # repetitions per config (>=5 gives a usable std)
# Each entry: "BW_MBIT DELAY_MS"   (delay is per-interface; RTT = 2 x delay)
CONFIGS=(
  "10 15"             # 10 Mbit, 30 ms RTT  (the paper's point)
  "20 15"             # 20 Mbit, 30 ms RTT
  "10 30"             # 10 Mbit, 60 ms RTT
)

HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$HERE/results"
SWEEP="$HERE/sweep"
mkdir -p "$SWEEP"

if [ ! -f "$HERE/models/lgbm_model.joblib" ]; then
  echo "ERROR: deployment/models/lgbm_model.joblib missing. Run export_model.py first." >&2
  exit 1
fi

total=$(( ${#CONFIGS[@]} * REPS ))
count=0
start=$(date +%s)

for cfg in "${CONFIGS[@]}"; do
  read -r BW DELAY <<< "$cfg"
  for r in $(seq 1 "$REPS"); do
    count=$((count + 1))
    dest="$SWEEP/bw${BW}_d${DELAY}/rep${r}"
    if [ -f "$dest/all_results.json" ]; then
      echo "[$count/$total] skip (exists): $dest"
      continue
    fi
    mkdir -p "$dest"
    echo "============================================================"
    echo "[$count/$total] bw=${BW}Mbit delay=${DELAY}ms rep=${r}  ($(date +%H:%M:%S))"
    echo "============================================================"

    mn -c >/dev/null 2>&1 || true   # clean any stale Mininet state

    python3 "$HERE/run_experiment.py" \
        --experiment all --cc both \
        --duration "$DURATION" --bw "$BW" --delay "$DELAY" \
        2>&1 | tail -n 40

    # Move this run's outputs out of the way so the next run can't overwrite them.
    mv "$RESULTS"/*_results.json "$dest/" 2>/dev/null || true
    mv "$RESULTS"/*_traces.npz  "$dest/" 2>/dev/null || true
    mv "$RESULTS"/all_results.json "$dest/" 2>/dev/null || true

    echo "  saved → $dest"
  done
done

elapsed=$(( $(date +%s) - start ))
echo
echo "Sweep done: $count runs in $((elapsed/60)) min. Results under $SWEEP/"
echo "Next:  python3 $HERE/aggregate_sweep.py"

#!/usr/bin/env python3
"""Trace labeling signals through actual experiment data to analyze slow-start behavior."""
import re, os

file_path = r'output_backup1\text\cubic\6bg_flows\30ms_10mbit_18750bytes_100s\ss_data.txt'
lines = open(file_path, 'r', encoding='utf-8', errors='ignore').readlines()

print(f"{'Row':>4} {'cwnd':>5} {'ssth':>5} {'retN':>5} {'lost':>5} {'retD':>5}  {'S1':>2} {'S2':>2} {'S3':>2} -> Label")
print("-" * 70)

prev_rt = 0
cwnd_vals = []
loss_count = 0
no_loss_count = 0

for i in range(len(lines)):
    line = lines[i].strip()
    if 'ESTAB' not in line or i + 1 >= len(lines):
        continue
    nxt = lines[i + 1].strip()
    cwnd_m = re.search(r'cwnd:(\d+)', nxt)
    ss_m = re.search(r'\bssthresh:(\d+)', nxt)
    rn_m = re.search(r'retrans:(\d+)/(\d+)', nxt)
    lost_m = re.search(r'\blost:(\d+)', nxt)
    if not cwnd_m:
        continue

    cwnd = int(cwnd_m.group(1))
    ssth = int(ss_m.group(1)) if ss_m else 0
    rn = int(rn_m.group(1)) if rn_m else 0
    rt = int(rn_m.group(2)) if rn_m else 0
    lost = int(lost_m.group(1)) if lost_m else 0
    rd = max(0, rt - prev_rt)
    prev_rt = rt

    cwnd_vals.append(cwnd)

    # Calculate signals (exactly as in enhanced_parser.py)
    roll5_max = max(cwnd_vals[-5:]) if len(cwnd_vals) > 0 else cwnd
    s1 = 1 if (roll5_max > 1 and cwnd <= roll5_max * 0.7) else 0
    s2 = 1 if (ssth > 0 and cwnd <= ssth * 1.2) else 0
    s3 = 1 if (rn > 0 or lost > 0 or rd > 0) else 0
    label = 1 if (s1 + s2 + s3 >= 2) else 0

    if label:
        loss_count += 1
    else:
        no_loss_count += 1

    marker = " <-- LABELED LOSS" if label else ""
    actual = " [ACTUAL LOSS]" if (rn > 0 or lost > 0 or rd > 0) else ""
    print(f"{len(cwnd_vals):4d} {cwnd:5d} {ssth:5d} {rn:5d} {lost:5d} {rd:5d}   {s1:1d}  {s2:1d}  {s3:1d} ->  {label}{marker}{actual}")

total = loss_count + no_loss_count
print(f"\n{'='*70}")
print(f"SUMMARY for this experiment:")
print(f"  Total rows:  {total}")
print(f"  Loss labels: {loss_count} ({loss_count/total*100:.1f}%)")
print(f"  No-loss:     {no_loss_count} ({no_loss_count/total*100:.1f}%)")
print(f"\nSLOW-START ANALYSIS:")
print(f"  First 10 records (likely slow-start/early phase):")
print(f"    Labels: {['LOSS' if i < loss_count else 'OK' for i in range(min(10, total))]}")

# Now count FALSE POSITIVES: labeled as loss but no actual retrans 
print(f"\nFALSE POSITIVE ANALYSIS:")
fp = 0
tp = 0
fn = 0
tn = 0
prev_rt2 = 0
cwnd_vals2 = []
for i in range(len(lines)):
    line = lines[i].strip()
    if 'ESTAB' not in line or i + 1 >= len(lines):
        continue
    nxt = lines[i + 1].strip()
    cwnd_m = re.search(r'cwnd:(\d+)', nxt)
    ss_m = re.search(r'\bssthresh:(\d+)', nxt)
    rn_m = re.search(r'retrans:(\d+)/(\d+)', nxt)
    lost_m = re.search(r'\blost:(\d+)', nxt)
    if not cwnd_m:
        continue
    cwnd = int(cwnd_m.group(1))
    ssth = int(ss_m.group(1)) if ss_m else 0
    rn = int(rn_m.group(1)) if rn_m else 0
    rt = int(rn_m.group(2)) if rn_m else 0
    lost = int(lost_m.group(1)) if lost_m else 0
    rd = max(0, rt - prev_rt2)
    prev_rt2 = rt
    cwnd_vals2.append(cwnd)
    
    roll5_max = max(cwnd_vals2[-5:])
    s1 = 1 if (roll5_max > 1 and cwnd <= roll5_max * 0.7) else 0
    s2 = 1 if (ssth > 0 and cwnd <= ssth * 1.2) else 0
    s3 = 1 if (rn > 0 or lost > 0 or rd > 0) else 0
    consensus_label = 1 if (s1 + s2 + s3 >= 2) else 0
    actual_loss = 1 if (rd > 0 or lost > 0) else 0
    
    if consensus_label == 1 and actual_loss == 0:
        fp += 1
    elif consensus_label == 1 and actual_loss == 1:
        tp += 1
    elif consensus_label == 0 and actual_loss == 1:
        fn += 1
    else:
        tn += 1

print(f"  True Positives  (consensus=1, actual=1): {tp}")
print(f"  False Positives (consensus=1, actual=0): {fp}  <-- labeled loss but NO actual loss!")
print(f"  False Negatives (consensus=0, actual=1): {fn}")
print(f"  True Negatives  (consensus=0, actual=0): {tn}")
if tp + fp > 0:
    print(f"  Precision: {tp/(tp+fp)*100:.1f}%")
if tp + fn > 0:
    print(f"  Recall:    {tp/(tp+fn)*100:.1f}%")

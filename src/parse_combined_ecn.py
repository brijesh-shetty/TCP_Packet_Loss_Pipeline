#!/usr/bin/env python3
"""parse_combined_ecn.py — Parse the combined_congestion1.txt ECN dataset.

This file is a large single ss capture with ECN enabled (ecn cubic).
Parses it using the same logic as enhanced_parser.py to produce a compatible
dataset that can be merged with enhanced_dataset.csv.
"""
import re
import pandas as pd
import numpy as np
import os

INPUT_FILE = r"combined_congestion1 (1).txt"
OUTPUT_FILE = "ecn_dataset.csv"

def parse_ss_block(line1, line2):
    """Parse a pair of ss output lines into a feature dict."""
    d = {}
    # Combine both lines
    text = line1 + " " + line2

    # State
    if "ESTAB" not in line1:
        return None

    # Skip SYN-SENT
    if "SYN-SENT" in line1 or "SYN_SENT" in line1:
        return None

    # ECN flag
    d['ecn_enabled'] = 1 if 'ecn' in text.lower() else 0

    # CC algo
    for algo in ['cubic', 'reno', 'bbr']:
        if algo in text:
            d['cc_algo_name'] = algo
            break
    else:
        d['cc_algo_name'] = 'unknown'

    def extract_int(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    def extract_float(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0

    # Core TCP metrics
    d['cwnd']           = extract_int(r'\bcwnd:(\d+)')
    d['ssthresh']       = extract_int(r'\bssthresh:(\d+)')
    d['rtt']            = extract_float(r'\brtt:([\d.]+)/')
    d['rtt_var']        = extract_float(r'\brtt:[\d.]+/([\d.]+)')
    d['rto']            = extract_int(r'\brto:(\d+)')
    d['mss']            = extract_int(r'\bmss:(\d+)')
    d['unacked']        = extract_int(r'\bunacked:(\d+)')
    d['lost']           = extract_int(r'\blost:(\d+)')
    d['sacked']         = extract_int(r'\bsacked:(\d+)')
    d['bytes_sent']     = extract_int(r'\bbytes_sent:(\d+)')
    d['bytes_acked']    = extract_int(r'\bbytes_acked:(\d+)')
    d['bytes_retrans']  = extract_int(r'\bbytes_retrans:(\d+)')
    d['segs_out']       = extract_int(r'\bsegs_out:(\d+)')
    d['segs_in']        = extract_int(r'\bsegs_in:(\d+)')
    d['notsent']        = extract_int(r'\bnotsent:(\d+)')

    # Delivery rate (Mbps)
    m = re.search(r'delivery_rate\s+([\d.]+)(Mbps|kbps|Gbps)', text)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == 'kbps': val /= 1000
        elif unit == 'Gbps': val *= 1000
        d['delivery_rate_mbps'] = val
    else:
        d['delivery_rate_mbps'] = 0.0

    # Pacing rate
    m = re.search(r'pacing_rate\s+([\d.]+)(Mbps|kbps|Gbps)', text)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == 'kbps': val /= 1000
        elif unit == 'Gbps': val *= 1000
        d['pacing_rate_mbps'] = val
    else:
        d['pacing_rate_mbps'] = 0.0

    # Send rate
    m = re.search(r'send\s+([\d.]+)(Mbps|kbps|Gbps)', text)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == 'kbps': val /= 1000
        elif unit == 'Gbps': val *= 1000
        d['send_rate_mbps'] = val
    else:
        d['send_rate_mbps'] = 0.0

    # Retrans
    m = re.search(r'\bretrans:(\d+)/(\d+)', text)
    if m:
        d['retrans_now'] = int(m.group(1))
        d['retrans_total'] = int(m.group(2))
    else:
        d['retrans_now'] = 0
        d['retrans_total'] = 0

    d['minrtt'] = extract_float(r'\bminrtt:([\d.]+)')

    # Skip rows with no cwnd
    if d['cwnd'] == 0:
        return None

    return d


def add_features_and_labels(df):
    """Add engineered features and multi-signal consensus labels."""
    df = df.sort_index().reset_index(drop=True)

    # Diffs
    df['cwnd_diff'] = df['cwnd'].diff().fillna(0)
    df['rtt_diff']  = df['rtt'].diff().fillna(0)
    df['cwnd_accel'] = df['cwnd_diff'].diff().fillna(0)

    # Rolling
    for w in [3, 5, 10]:
        df[f'cwnd_roll{w}_mean'] = df['cwnd'].rolling(w, min_periods=1).mean()
        df[f'rtt_roll{w}_mean']  = df['rtt'].rolling(w, min_periods=1).mean()
    df['cwnd_roll5_std'] = df['cwnd'].rolling(5, min_periods=1).std().fillna(0)
    df['rtt_roll5_std']  = df['rtt'].rolling(5, min_periods=1).std().fillna(0)

    # Lags
    for lag in [1, 2, 3]:
        df[f'cwnd_lag{lag}'] = df['cwnd'].shift(lag).fillna(0)
        df[f'rtt_lag{lag}']  = df['rtt'].shift(lag).fillna(0)

    # EWM
    df['cwnd_ewm'] = df['cwnd'].ewm(span=5).mean()
    df['rtt_ewm']  = df['rtt'].ewm(span=5).mean()

    # Interactions
    df['cwnd_rtt_interaction'] = df['cwnd'] * df['rtt']
    df['rtt_ratio'] = df['rtt'] / df['rtt_roll5_mean'].replace(0, 1)

    # Rolling peak for drop detection
    df['cwnd_roll5_max'] = df['cwnd'].rolling(5, min_periods=1).max()
    df['cwnd_drop_pct']  = (df['cwnd_roll5_max'] - df['cwnd']) / df['cwnd_roll5_max'].replace(0, 1)

    # Min/max running stats
    df['run_min_cwnd'] = df['cwnd'].expanding().min()
    df['run_max_cwnd'] = df['cwnd'].expanding().max()
    df['run_min_rtt']  = df['rtt'].expanding().min()
    df['run_max_rtt']  = df['rtt'].expanding().max()

    # Multi-signal consensus labeling (same as enhanced_parser.py)
    # Signal 1: CWND dropped >= 30% from rolling peak
    sig1 = (df['cwnd_drop_pct'] >= 0.30).astype(int)

    # Signal 2: ssthresh throttling (cwnd <= ssthresh * 1.2, ssthresh < 65535)
    sig2 = ((df['cwnd'] <= df['ssthresh'] * 1.2) & (df['ssthresh'] < 65535)).astype(int)

    # Signal 3: Retransmission activity
    sig3 = (df['retrans_now'] > 0).astype(int)

    df['signal_sum'] = sig1 + sig2 + sig3
    df['label'] = (df['signal_sum'] >= 2).astype(int)

    # Drop raw leak features before saving
    drop_cols = ['lost', 'retrans_now', 'retrans_total', 'sacked',
                 'bytes_retrans', 'signal_sum', 'cwnd_drop_pct']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    input_path = os.path.join(data_dir, INPUT_FILE)
    output_path = os.path.join(data_dir, OUTPUT_FILE)

    print('=' * 65)
    print('  Parsing ECN Dataset: combined_congestion1.txt')
    print('=' * 65)

    if not os.path.exists(input_path):
        print(f'ERROR: File not found: {input_path}')
        return

    file_size_mb = os.path.getsize(input_path) / 1e6
    print(f'  File size: {file_size_mb:.0f} MB')
    print('  Parsing lines (this may take a minute)...')

    records = []
    skipped = 0
    processed = 0

    with open(input_path, 'r', errors='replace') as f:
        lines = []
        for line in f:
            lines.append(line.strip())

    print(f'  Total lines: {len(lines):,}')

    i = 0
    while i < len(lines) - 1:
        l1 = lines[i]
        l2 = lines[i+1] if i+1 < len(lines) else ''

        if 'Netid' in l1:
            i += 1
            continue

        if l1.startswith('tcp') and 'ESTAB' in l1:
            # Try to get second info line
            j = i + 1
            info_line = ''
            while j < len(lines) and j < i + 3:
                if lines[j].startswith('tcp') or 'Netid' in lines[j]:
                    break
                info_line = lines[j]
                j += 1

            rec = parse_ss_block(l1, info_line)
            if rec:
                records.append(rec)
                processed += 1
            else:
                skipped += 1
            i = j
        else:
            i += 1

        if processed % 50000 == 0 and processed > 0:
            print(f'  Processed: {processed:,} records...')

    print(f'\n  Parsed: {processed:,} records, skipped: {skipped:,}')

    if not records:
        print('  ERROR: No records parsed!')
        return

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Add engineered features and labels
    print('  Adding features and labels...')
    df = add_features_and_labels(df)

    loss_pct = df['label'].mean() * 100
    print(f'\n  Final dataset: {len(df):,} rows')
    print(f'  Loss rate: {loss_pct:.2f}%')
    print(f'  ECN enabled: {df["ecn_enabled"].mean()*100:.0f}% of rows')
    print(f'  Features: {df.shape[1]}')

    df.to_csv(output_path, index=False)
    print(f'\n  Saved → {output_path}')
    print('=' * 65)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""enhanced_parser.py — Maximum data retention parser with network condition features.

Parses ALL 300 raw experiment files from output_backup1, extracts network condition
metadata from folder names, and engineers advanced features for TCP loss prediction.
"""
import re, os, sys, glob
import pandas as pd
import numpy as np


def parse_network_conditions(folder_name):
    """Extract delay, bandwidth, buffer_size, duration from folder name.
    Example: '30ms_10mbit_18750bytes_100s' → (30, 10, 18750, 100)
    """
    m = re.match(r'(\d+)ms_(\d+)mbit_(\d+)bytes_(\d+)s', folder_name)
    if m:
        return {
            'delay_ms': int(m.group(1)),
            'bandwidth_mbit': int(m.group(2)),
            'buffer_bytes': int(m.group(3)),
            'duration_s': int(m.group(4)),
        }
    return None


def parse_ss_file(file_path, expected_cc, experiment_id, net_cond):
    """Parse raw ss output. Keep ALL valid ESTAB+data pairs."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    records = []
    prev_retrans_total = 0
    prev_data_segs = 0
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if 'ESTAB' not in line:
            i += 1
            continue

        if i + 1 >= len(lines):
            i += 1
            continue

        next_line = lines[i + 1].strip()

        # Verify CC algorithm
        if expected_cc == 'reno' and 'reno' not in next_line:
            i += 2
            continue
        if expected_cc == 'cubic' and 'cubic' not in next_line:
            i += 2
            continue

        # Must have rtt and cwnd
        rtt_m = re.search(r'rtt:(\d+\.?\d*)/(\d+\.?\d*)', next_line)
        cwnd_m = re.search(r'cwnd:(\d+)', next_line)
        if not rtt_m or not cwnd_m:
            i += 2
            continue

        row = {}
        row['cc'] = expected_cc
        row['experiment_id'] = experiment_id

        # Network conditions
        if net_cond:
            row['delay_ms'] = net_cond['delay_ms']
            row['bandwidth_mbit'] = net_cond['bandwidth_mbit']
            row['buffer_bytes'] = net_cond['buffer_bytes']
            row['duration_s'] = net_cond['duration_s']

        # --- Core TCP features ---
        row['rtt'] = float(rtt_m.group(1))
        row['rtt_variance'] = float(rtt_m.group(2))
        row['cwnd'] = int(cwnd_m.group(1))

        m = re.search(r'\bssthresh:(\d+)', next_line)
        row['ssthresh'] = int(m.group(1)) if m else 0

        m = re.search(r'rto:(\d+)', next_line)
        row['rto'] = int(m.group(1)) if m else 0

        for field in ['bytes_sent', 'bytes_acked', 'bytes_retrans']:
            m = re.search(rf'{field}:(\d+)', next_line)
            row[field] = int(m.group(1)) if m else 0

        for field in ['segs_out', 'segs_in', 'data_segs_out']:
            m = re.search(rf'{field}:(\d+)', next_line)
            row[field] = int(m.group(1)) if m else 0

        for field in ['unacked', 'lost', 'sacked']:
            m = re.search(rf'\b{field}:(\d+)', next_line)
            row[field] = int(m.group(1)) if m else 0

        m = re.search(r'delivered:(\d+)', next_line)
        row['delivered'] = int(m.group(1)) if m else 0

        m = re.search(r'notsent:(\d+)', next_line)
        row['notsent'] = int(m.group(1)) if m else 0

        m = re.search(r'rcv_space:(\d+)', next_line)
        row['rcv_space'] = int(m.group(1)) if m else 0

        m = re.search(r'minrtt:(\d+\.?\d*)', next_line)
        row['minrtt'] = float(m.group(1)) if m else 0

        m = re.search(r'mss:(\d+)', next_line)
        row['mss'] = int(m.group(1)) if m else 1448

        m = re.search(r'pmtu:(\d+)', next_line)
        row['pmtu'] = int(m.group(1)) if m else 0

        m = re.search(r'retrans:(\d+)/(\d+)', next_line)
        if m:
            row['retrans_now'] = int(m.group(1))
            row['retrans_total'] = int(m.group(2))
        else:
            row['retrans_now'] = 0
            row['retrans_total'] = 0

        # Rates
        def parse_rate(text, key):
            m = re.search(rf'{key}\s+([\d.e+\-]+)\s*(Gbps|Mbps|kbps|bps)', text)
            if not m:
                m = re.search(rf'{key}\s+([\d.e+\-]+)(Gbps|Mbps|kbps|bps)', text)
            if m:
                val = float(m.group(1))
                unit = m.group(2)
                if unit == 'Gbps': val *= 1e9
                elif unit == 'Mbps': val *= 1e6
                elif unit == 'kbps': val *= 1e3
                return val
            return 0

        row['send_rate'] = parse_rate(next_line, 'send')
        row['pacing_rate'] = parse_rate(next_line, 'pacing_rate')
        row['delivery_rate'] = parse_rate(next_line, 'delivery_rate')

        m = re.search(r'lastsnd:(\d+)', next_line)
        row['last_send'] = int(m.group(1)) if m else 0

        # Timer info
        timer_m = re.search(r'timer:\((\w+),(\d+)ms?,(\d+)\)', line)
        if timer_m:
            row['expire_time'] = int(timer_m.group(2))
            row['timer_retrans'] = int(timer_m.group(3))
        else:
            row['expire_time'] = 0
            row['timer_retrans'] = 0

        # Per-interval retrans diff
        current_dso = row['data_segs_out']
        row['data_segs_sent'] = max(0, current_dso - prev_data_segs)
        prev_data_segs = current_dso

        row['retrans_diff'] = max(0, row['retrans_total'] - prev_retrans_total)
        prev_retrans_total = row['retrans_total']

        records.append(row)
        i += 2

    return records


def add_features_and_labels(df):
    """Add derived features and multi-signal consensus labels."""
    if len(df) == 0:
        return df

    grp = ['cc', 'experiment_id']

    # --- Rolling features (per experiment) ---
    for col in ['cwnd', 'rtt']:
        df[f'{col}_roll5_mean'] = df.groupby(grp)[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{col}_roll5_std'] = df.groupby(grp)[col].transform(
            lambda x: x.rolling(5, min_periods=1).std().fillna(0))
        df[f'{col}_roll3_mean'] = df.groupby(grp)[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean())

    # 1st-order diffs
    df['cwnd_diff'] = df.groupby(grp)['cwnd'].diff().fillna(0)
    df['rtt_diff'] = df.groupby(grp)['rtt'].diff().fillna(0)

    # 2nd-order diffs (acceleration)
    df['cwnd_accel'] = df.groupby(grp)['cwnd_diff'].diff().fillna(0)
    df['rtt_accel'] = df.groupby(grp)['rtt_diff'].diff().fillna(0)

    # Lag features
    df['cwnd_lag1'] = df.groupby(grp)['cwnd'].shift(1).fillna(method='bfill')
    df['rtt_lag1'] = df.groupby(grp)['rtt'].shift(1).fillna(method='bfill')
    df['cwnd_lag2'] = df.groupby(grp)['cwnd'].shift(2).fillna(method='bfill')
    df['rtt_lag2'] = df.groupby(grp)['rtt'].shift(2).fillna(method='bfill')

    # Running min/max per experiment
    df['run_min_cwnd'] = df.groupby(grp)['cwnd'].transform('cummin')
    df['run_max_cwnd'] = df.groupby(grp)['cwnd'].transform('cummax')
    df['run_min_rtt'] = df.groupby(grp)['rtt'].transform('cummin')
    df['run_max_rtt'] = df.groupby(grp)['rtt'].transform('cummax')
    df['run_min_ssthresh'] = df.groupby(grp)['ssthresh'].transform(
        lambda x: x.where(x > 0).expanding().min().fillna(0))
    df['run_max_ssthresh'] = df.groupby(grp)['ssthresh'].transform(
        lambda x: x.where(x > 0).expanding().max().fillna(0))

    # Ratios
    df['retrans_ratio'] = df['bytes_retrans'] / df['bytes_sent'].replace(0, 1)
    df['ack_ratio'] = df['bytes_acked'] / df['bytes_sent'].replace(0, 1)
    df['rtt_ratio'] = df['rtt'] / df['minrtt'].replace(0, 1)
    df['cwnd_util'] = df['unacked'] / df['cwnd'].replace(0, 1)

    # EWM momentum
    df['cwnd_ewm'] = df.groupby(grp)['cwnd'].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean())
    df['rtt_ewm'] = df.groupby(grp)['rtt'].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean())

    # --- NEW ADVANCED FEATURES ---

    # Interaction features
    df['cwnd_rtt_interaction'] = df['cwnd'] * df['rtt']
    df['send_efficiency'] = df['bytes_acked'] / df['bytes_sent'].replace(0, 1)

    # Buffer fill ratio
    df['buffer_fill_ratio'] = df['notsent'] / (df['cwnd'].replace(0, 1) * df['mss'].replace(0, 1448))

    # Retransmission burst (rolling 3-window sum)
    df['retrans_burst'] = df.groupby(grp)['retrans_diff'].transform(
        lambda x: x.rolling(3, min_periods=1).sum())

    # RTT z-score within experiment
    rtt_exp_mean = df.groupby(grp)['rtt'].transform('mean')
    rtt_exp_std = df.groupby(grp)['rtt'].transform('std').replace(0, 1)
    df['rtt_zscore'] = (df['rtt'] - rtt_exp_mean) / rtt_exp_std

    # CWND volatility (coefficient of variation over rolling window)
    df['cwnd_volatility'] = df['cwnd_roll5_std'] / df['cwnd_roll5_mean'].replace(0, 1)

    # Congestion signal (composite)
    df['congestion_signal'] = (
        (df['ssthresh'] > 0).astype(int) +
        (df['rtt_ratio'] > 2).astype(int) +
        (df['rto'] > 500).astype(int)
    )

    # Throughput estimate
    df['throughput'] = df['delivery_rate'] * df['cwnd']

    # RTT range within experiment
    df['rtt_range'] = df['run_max_rtt'] - df['run_min_rtt']

    # CWND range within experiment
    df['cwnd_range'] = df['run_max_cwnd'] - df['run_min_cwnd']

    # Bandwidth-delay product estimate
    if 'bandwidth_mbit' in df.columns and 'delay_ms' in df.columns:
        df['bdp_estimate'] = df['bandwidth_mbit'] * 1e6 * df['delay_ms'] / 1000 / 8

    # Rolling peak for labeling
    df['cwnd_roll5_max'] = df.groupby(grp)['cwnd'].transform(
        lambda x: x.rolling(5, min_periods=1).max())

    # --- MULTI-SIGNAL LABELING ---
    sig_cwnd = ((df['cwnd_roll5_max'] > 1) &
                (df['cwnd'] <= df['cwnd_roll5_max'] * 0.7)).astype(int)
    sig_ssthresh = ((df['ssthresh'] > 0) &
                    (df['cwnd'] <= df['ssthresh'] * 1.2)).astype(int)
    sig_retrans = ((df['retrans_now'] > 0) | (df['lost'] > 0) |
                   (df['retrans_diff'] > 0)).astype(int)

    signal_sum = sig_cwnd + sig_ssthresh + sig_retrans
    df['label'] = (signal_sum >= 2).astype(int)

    # Drop intermediate columns
    df.drop(columns=['cwnd_roll5_max'], inplace=True)

    return df


def parse_all_experiments(backup_dir):
    """Parse all experiments from the backup directory structure."""
    all_records = []
    experiment_id = 0

    for cc in ['reno', 'cubic']:
        base = os.path.join(backup_dir, 'text', cc, '6bg_flows')
        if not os.path.isdir(base):
            print(f'  [SKIP] Directory not found: {base}')
            continue

        experiment_dirs = sorted([d for d in os.listdir(base)
                                  if os.path.isdir(os.path.join(base, d))])

        print(f'\n  Parsing {cc.upper()}: {len(experiment_dirs)} experiments')
        for exp_dir_name in experiment_dirs:
            ss_file = os.path.join(base, exp_dir_name, 'ss_data.txt')
            if not os.path.exists(ss_file):
                continue

            net_cond = parse_network_conditions(exp_dir_name)
            recs = parse_ss_file(ss_file, cc, experiment_id, net_cond)
            all_records.extend(recs)
            experiment_id += 1

        print(f'    → {experiment_id} experiments parsed, {len(all_records)} total records so far')

    return all_records


def main():
    backup_dir = r'c:\Users\Lenovo\Downloads\output_backup1-20260327T023325Z-1-001\output_backup1'
    out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    print('=' * 70)
    print('  ENHANCED TCP PACKET LOSS PARSER')
    print('  Parsing all 300 raw experiments + network condition features')
    print('=' * 70)

    # Parse from individual experiment files
    all_records = parse_all_experiments(backup_dir)

    if not all_records:
        # Fallback to combined files if backup dir not accessible
        print('\n  [FALLBACK] Trying combined files...')
        for fpath, cc in [
            (os.path.join(out_dir, '_com1.txt'), 'reno'),
            (os.path.join(out_dir, '_com2.txt'), 'cubic'),
        ]:
            if os.path.exists(fpath):
                print(f'  Parsing {os.path.basename(fpath)} as {cc}...')
                recs = parse_ss_file(fpath, cc, 0, None)
                all_records.extend(recs)
                print(f'    → {len(recs)} records')

    if not all_records:
        print('ERROR: No records parsed!')
        sys.exit(1)

    df = pd.DataFrame(all_records)
    print(f'\n  Raw records: {len(df)}')

    # Add features and labels
    print('  Engineering features...')
    df = add_features_and_labels(df)

    # Replace inf with NaN, then fill NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split by CC
    reno_df = df[df['cc'] == 'reno'].copy()
    cubic_df = df[df['cc'] == 'cubic'].copy()

    # Drop helper columns for output
    drop_cols = ['cc', 'experiment_id', 'mss']
    reno_out = reno_df.drop(columns=[c for c in drop_cols if c in reno_df.columns])
    cubic_out = cubic_df.drop(columns=[c for c in drop_cols if c in cubic_df.columns])
    combined_out = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Save
    combined_path = os.path.join(out_dir, 'enhanced_dataset.csv')
    reno_path = os.path.join(out_dir, 'reno_enhanced.csv')
    cubic_path = os.path.join(out_dir, 'cubic_enhanced.csv')

    combined_out.to_csv(combined_path, index=False)
    reno_out.to_csv(reno_path, index=False)
    cubic_out.to_csv(cubic_path, index=False)

    # Summary
    print('\n' + '=' * 70)
    print('  PARSING SUMMARY')
    print('=' * 70)
    n_features = len([c for c in combined_out.columns if c != 'label'])
    print(f'  Features: {n_features}')
    print(f'  Reno:     {len(reno_out):>6} rows | Loss: {(reno_out["label"]==1).sum():>5} ({(reno_out["label"]==1).mean()*100:.1f}%)')
    print(f'  Cubic:    {len(cubic_out):>6} rows | Loss: {(cubic_out["label"]==1).sum():>5} ({(cubic_out["label"]==1).mean()*100:.1f}%)')
    print(f'  Combined: {len(combined_out):>6} rows | Loss: {(combined_out["label"]==1).sum():>5} ({(combined_out["label"]==1).mean()*100:.1f}%)')
    print(f'\n  Feature list: {[c for c in combined_out.columns if c != "label"]}')
    print(f'\n  Saved:')
    print(f'    {combined_path}')
    print(f'    {reno_path}')
    print(f'    {cubic_path}')
    print('=' * 70)

    return combined_path


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""aggregate_sweep.py — Turn the repeated sweep runs into mean ± std statistics.

Reads every sweep/<config>/rep<k>/all_results.json produced by run_sweep.sh and reports,
per (config, CC, condition): throughput (BOTH ss-derived and iperf goodput),
retransmissions, CWND, CWND std, RTT, ECN signals — each as mean ± std over reps.
Also computes the our_ecn-vs-baseline throughput gain per rep, then mean ± std, so the
"+15.5%" claim becomes "+X% ± Y% over N runs".

Usage:
    python3 aggregate_sweep.py                 # reads ./sweep
    python3 aggregate_sweep.py --sweep PATH
"""
import os
import json
import glob
import argparse
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CONDITIONS = ['baseline', 'basepaper_ecn', 'our_ecn']


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None, 0)
    if len(vals) == 1:
        return (vals[0], 0.0, 1)
    return (st.mean(vals), st.pstdev(vals), len(vals))


def fmt(mean, sd, scale=1.0, unit='', dec=2):
    if mean is None:
        return 'n/a'
    return f'{mean/scale:.{dec}f}±{sd/scale:.{dec}f}{unit}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default=os.path.join(HERE, 'sweep'))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.sweep, '*', 'rep*', 'all_results.json')))
    if not files:
        print(f'No runs found under {args.sweep}. Run run_sweep.sh first.'); return
    print(f'Found {len(files)} sweep runs.\n')

    # acc[config][cc][cond][metric] = list of values across reps
    acc, gains = {}, {}
    for fp in files:
        config = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        with open(fp) as f:
            data = json.load(f)
        for cc in data:
            for cond in data[cc]:
                r = data[cc][cond]
                m = r.get('metrics', {})
                ip = r.get('iperf', {}) or {}
                d = acc.setdefault(config, {}).setdefault(cc, {}).setdefault(cond, {})
                d.setdefault('thr_ss', []).append(m.get('avg_throughput_bps'))
                d.setdefault('thr_iperf', []).append(ip.get('bits_per_second') or None)
                d.setdefault('retrans', []).append(m.get('total_retransmissions'))
                d.setdefault('cwnd', []).append(m.get('avg_cwnd'))
                d.setdefault('cwnd_std', []).append(m.get('cwnd_std'))
                d.setdefault('rtt', []).append(m.get('avg_rtt_ms'))
                d.setdefault('ecn', []).append(len(r.get('ecn_events', [])))
            # per-rep gain our_ecn vs baseline (ss throughput)
            try:
                b = data[cc]['baseline']['metrics']['avg_throughput_bps']
                o = data[cc]['our_ecn']['metrics']['avg_throughput_bps']
                if b:
                    gains.setdefault(config, {}).setdefault(cc, []).append((o - b) / b * 100)
            except KeyError:
                pass

    for config in sorted(acc):
        print('=' * 78)
        print(f'  CONFIG: {config}')
        print('=' * 78)
        for cc in sorted(acc[config]):
            print(f'\n  {cc.upper()}')
            print(f'  {"condition":<15}{"thr_ss(Mb)":>14}{"thr_iperf(Mb)":>16}'
                  f'{"retrans":>12}{"cwnd_std":>11}{"ECN":>8}')
            for cond in CONDITIONS:
                if cond not in acc[config][cc]:
                    continue
                d = acc[config][cc][cond]
                tss = ms(d['thr_ss']); tip = ms(d['thr_iperf'])
                rt = ms(d['retrans']); cs = ms(d['cwnd_std']); ec = ms(d['ecn'])
                print(f'  {cond:<15}{fmt(*tss[:2],1e6):>14}{fmt(*tip[:2],1e6):>16}'
                      f'{fmt(*rt[:2],1,"",1):>12}{fmt(*cs[:2],1,"",2):>11}'
                      f'{fmt(*ec[:2],1,"",1):>8}')
                if cond == 'our_ecn' and tip[0] is None:
                    print('     ⚠ iperf goodput missing for our_ecn — check /tmp/iperf_result.json capture.')
            if config in gains and cc in gains[config]:
                g = ms(gains[config][cc])
                print(f'    → our_ecn throughput gain vs baseline: '
                      f'{g[0]:+.1f}% ± {g[1]:.1f}%  (n={g[2]} reps)')

    # machine-readable summary
    summary = {}
    for config in acc:
        summary[config] = {}
        for cc in acc[config]:
            summary[config][cc] = {}
            for cond in acc[config][cc]:
                d = acc[config][cc][cond]
                summary[config][cc][cond] = {
                    k: {'mean': ms(v)[0], 'std': ms(v)[1], 'n': ms(v)[2]} for k, v in d.items()}
            if config in gains and cc in gains.get(config, {}):
                g = ms(gains[config][cc])
                summary[config][cc]['throughput_gain_pct'] = {'mean': g[0], 'std': g[1], 'n': g[2]}

    out = os.path.join(args.sweep, 'sweep_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary written → {out}')


if __name__ == '__main__':
    main()

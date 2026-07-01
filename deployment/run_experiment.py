#!/usr/bin/env python3
"""run_experiment.py — Run all 3 Artificial ECN experiments on Mininet.

Usage (on Ubuntu, as root):
    sudo python3 run_experiment.py                    # Run all 3 experiments
    sudo python3 run_experiment.py --experiment baseline   # Run only baseline
    sudo python3 run_experiment.py --experiment basepaper  # Run only base paper ECN
    sudo python3 run_experiment.py --experiment ours       # Run only our ECN
    sudo python3 run_experiment.py --duration 60           # 60-second experiments
    sudo python3 run_experiment.py --cc cubic              # Use Cubic (default: both)

Experiments:
  1. Baseline:     Standard TCP — no ECN, no ML (control group)
  2. Base Paper:   XGBoost prediction + binary ECN (reproducing their approach)
  3. Ours:         LightGBM prediction + proportional ECN (our improvement)
"""
import os
import sys
import time
import json
import signal
import argparse
import threading
import subprocess
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        module='sklearn')

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mininet.log import setLogLevel, info
from mininet.clean import cleanup

from ecn_topology import (create_topology, set_congestion_control,
                           setup_baseline_qdisc, setup_red_ecn_qdisc)
from ml_ecn_controller import TCPStateMonitor, ArtificialECNController


class ExperimentRunner:
    """Runs a single ECN experiment and collects results."""

    def __init__(self, experiment_name, duration_s=120, cc_algo='cubic',
                 bw_mbit=10, delay_ms=30, n_bg_flows=6, polling_ms=20,
                 model_dir=None):
        self.experiment_name = experiment_name
        self.duration_s = duration_s
        self.cc_algo = cc_algo
        self.bw_mbit = bw_mbit
        self.delay_ms = delay_ms
        self.n_bg_flows = n_bg_flows
        self.polling_ms = polling_ms
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models')

        # Results
        self.cwnd_trace = []
        self.rtt_trace = []
        self.throughput_trace = []
        self.retrans_trace = []
        self.timestamps = []
        self.ecn_events = []

    def run(self):
        """Execute the experiment."""
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT: {self.experiment_name}")
        print(f"  CC: {self.cc_algo}, Duration: {self.duration_s}s, "
              f"BW: {self.bw_mbit}Mbit, Delay: {self.delay_ms}ms")
        print(f"{'='*70}")

        # Clean up any previous Mininet
        cleanup()
        time.sleep(1)

        # Create topology
        print("\n  Creating Mininet topology...")
        red_min = 5000
        red_max = 15000
        net, config = create_topology(
            bw_mbit=self.bw_mbit,
            delay_ms=self.delay_ms,
            n_bg_flows=self.n_bg_flows,
            red_min=red_min,
            red_max=red_max,
        )

        h1 = config['h1']  # Main sender
        h3 = config['h3']  # Main receiver
        s1 = config['s1']  # Switch with bottleneck
        bg_senders = config['bg_senders']
        bg_receivers = config['bg_receivers']
        s1_to_s2_intf = config['s1_to_s2_intf']

        try:
            # Set congestion control
            print(f"  Setting CC algorithm: {self.cc_algo}")
            set_congestion_control(h1, self.cc_algo)
            set_congestion_control(h3, self.cc_algo)
            for hs in bg_senders:
                set_congestion_control(hs, self.cc_algo)

            # For baseline experiment — disable ECN and use pfifo instead of RED
            if self.experiment_name == 'baseline':
                print("  Disabling ECN (baseline experiment)")
                for host in [h1, h3] + bg_senders + bg_receivers:
                    host.cmd('sysctl -w net.ipv4.tcp_ecn=0')
                # Replace RED with pfifo (keeps htb rate limit)
                if s1_to_s2_intf:
                    setup_baseline_qdisc(s1, s1_to_s2_intf, self.bw_mbit)
                s2_to_s1_intf = config.get('s2_to_s1_intf')
                if s2_to_s1_intf:
                    setup_baseline_qdisc(config['s2'], s2_to_s1_intf, self.bw_mbit)

            # Print tc diagnostic
            tc_show = s1.cmd(f'tc qdisc show dev {s1_to_s2_intf}')
            print(f"  tc qdisc on bottleneck ({s1_to_s2_intf}):")
            for line in tc_show.strip().split('\n'):
                print(f"    {line}")

            # Start iperf3 servers
            print("  Starting iperf3 servers...")
            h3.cmd('iperf3 -s -D -p 5201')
            time.sleep(0.5)
            for i, hr in enumerate(bg_receivers):
                hr.cmd(f'iperf3 -s -D -p {5210 + i}')
                time.sleep(0.2)

            # Start background flows
            print(f"  Starting {self.n_bg_flows} background flows...")
            for i, (hs, hr) in enumerate(zip(bg_senders, bg_receivers)):
                hs.cmd(f'iperf3 -c {hr.IP()} -p {5210 + i} '
                       f'-t {self.duration_s + 10} -b 2M &')
                time.sleep(0.2)

            # Wait for background flows to ramp up
            time.sleep(3)

            # Verify background flows are running
            bg_check = bg_senders[0].cmd('ss -tn state established')
            bg_count = bg_check.count('ESTAB') + bg_check.count('10.0.0.')
            print(f"  Background flow check: {bg_count} connections active")

            # Initialize ML-ECN controller if needed
            ecn_controller = None
            tcp_monitor = None

            if self.experiment_name in ['basepaper_ecn', 'our_ecn']:
                model_type = 'xgb' if self.experiment_name == 'basepaper_ecn' else 'lgbm'
                model_file = ('xgb_model.joblib' if model_type == 'xgb'
                              else 'lgbm_model.joblib')

                if os.path.exists(os.path.join(self.model_dir, model_file)):
                    tcp_monitor = TCPStateMonitor(
                        h1.cmd, h3.IP(), self.polling_ms)
                    ecn_controller = ArtificialECNController(
                        self.model_dir, s1.cmd, s1_to_s2_intf,
                        orig_red_min=red_min, orig_red_max=red_max,
                        bw_mbit=self.bw_mbit, model_type=model_type)
                    print(f"  ML-ECN Controller initialized ({model_type.upper()})")
                else:
                    print(f"  WARNING: {model_file} not found! "
                          f"Running without ML-ECN.")

            # Start main iperf3 flow
            print(f"\n  Starting main flow (h1 → h3, {self.duration_s}s)...")
            h1.cmd(f'iperf3 -c {h3.IP()} -p 5201 -t {self.duration_s} '
                   f'-J > /tmp/iperf_result.json &')

            # Wait for connection to establish, then verify
            time.sleep(2)
            debug_ss = h1.cmd('ss -tin')
            print(f"  DEBUG ss output (first 500 chars):")
            print(f"  {debug_ss[:500]}")
            print()

            # --- Main monitoring loop ---
            print(f"  Monitoring TCP state every {self.polling_ms}ms...")
            start_time = time.time()
            tick = 0

            while time.time() - start_time < self.duration_s:
                loop_start = time.time()

                # Poll TCP state
                raw_state = None
                if tcp_monitor:
                    raw_state = tcp_monitor.poll_once()
                else:
                    # Even without ML, we poll ss for metrics
                    try:
                        # Use -tin for numeric output (no DNS delays)
                        # Filter by port 5201 (main iperf flow) to avoid bg flows
                        raw = h1.cmd('ss -tin sport 5201 or dport 5201')
                        raw_state = self._quick_parse_ss(raw)
                    except:
                        raw_state = None

                if raw_state and raw_state.get('cwnd', 0) > 0:
                    elapsed = time.time() - start_time
                    self.timestamps.append(elapsed)
                    self.cwnd_trace.append(raw_state.get('cwnd', 0))
                    self.rtt_trace.append(raw_state.get('rtt', 0))
                    self.retrans_trace.append(raw_state.get('retrans', 0))

                    # Compute throughput from send rate
                    send_rate = raw_state.get('send', 0)
                    self.throughput_trace.append(send_rate)

                    # ML-ECN controller tick
                    if ecn_controller and tcp_monitor:
                        tcp_monitor.history.append(raw_state)
                        if len(tcp_monitor.history) > tcp_monitor.max_history:
                            tcp_monitor.history.pop(0)

                        features = tcp_monitor.compute_features(raw_state)
                        result = ecn_controller.tick(features, elapsed)

                        if result['action'] == 'ecn_injected':
                            self.ecn_events.append({
                                'time': elapsed,
                                'probability': result['probability'],
                                'cwnd': result['cwnd'],
                            })

                    tick += 1
                    if tick % 250 == 0:
                        ecn_info = ""
                        if ecn_controller:
                            summary = ecn_controller.get_summary()
                            ecn_info = (f" | ECN signals: {summary['ecn_signals_sent']}"
                                        f" | Pred rate: {summary['prediction_rate']:.1%}")
                        print(f"    [{elapsed:.1f}s] cwnd={raw_state.get('cwnd',0):.0f} "
                              f"rtt={raw_state.get('rtt',0):.1f}ms{ecn_info}")

                # Sleep for remaining interval
                elapsed_loop = time.time() - loop_start
                sleep_time = max(0, (self.polling_ms / 1000.0) - elapsed_loop)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # --- Collect iperf3 results ---
            time.sleep(2)
            iperf_json = h1.cmd('cat /tmp/iperf_result.json')
            iperf_results = self._parse_iperf(iperf_json)

            # Get final retransmission count
            final_ss = h1.cmd(f'ss -ti dst {h3.IP()}')
            final_retrans = self._extract_retrans(final_ss)

        finally:
            # Cleanup
            print("  Stopping network...")
            net.stop()
            time.sleep(1)

        # --- Compile results ---
        results = {
            'experiment': self.experiment_name,
            'cc_algo': self.cc_algo,
            'duration_s': self.duration_s,
            'bw_mbit': self.bw_mbit,
            'delay_ms': self.delay_ms,
            'n_bg_flows': self.n_bg_flows,
            'metrics': {
                'avg_throughput_bps': np.mean(self.throughput_trace) if self.throughput_trace else 0,
                'max_throughput_bps': max(self.throughput_trace) if self.throughput_trace else 0,
                'total_retransmissions': int(final_retrans),
                'avg_cwnd': np.mean(self.cwnd_trace) if self.cwnd_trace else 0,
                'cwnd_std': np.std(self.cwnd_trace) if self.cwnd_trace else 0,
                'avg_rtt_ms': np.mean(self.rtt_trace) if self.rtt_trace else 0,
                'rtt_std_ms': np.std(self.rtt_trace) if self.rtt_trace else 0,
                'n_samples': len(self.cwnd_trace),
            },
            'iperf': iperf_results,
            'traces': {
                'timestamps': self.timestamps,
                'cwnd': self.cwnd_trace,
                'rtt': self.rtt_trace,
                'throughput': self.throughput_trace,
            },
            'ecn_events': self.ecn_events,
        }

        if ecn_controller:
            results['ecn_summary'] = ecn_controller.get_summary()
            results['ecn_prediction_log'] = ecn_controller.prediction_log

        # Print summary
        m = results['metrics']
        print(f"\n  {'─'*50}")
        print(f"  RESULTS: {self.experiment_name}")
        print(f"  {'─'*50}")
        print(f"  Avg Throughput:   {m['avg_throughput_bps']/1e6:.2f} Mbps")
        print(f"  Retransmissions:  {m['total_retransmissions']}")
        print(f"  Avg CWND:         {m['avg_cwnd']:.1f} segments")
        print(f"  CWND Stability:   {m['cwnd_std']:.2f} (std dev)")
        print(f"  Avg RTT:          {m['avg_rtt_ms']:.2f} ms")
        print(f"  Samples:          {m['n_samples']}")
        if self.ecn_events:
            print(f"  ECN Signals:      {len(self.ecn_events)}")

        return results

    def _quick_parse_ss(self, raw):
        """Quick ss parser for non-ML experiments."""
        import re
        state = {}
        patterns = {
            'cwnd': r'cwnd:(\d+)',
            'rtt': r'rtt:([0-9.]+)',
            'ssthresh': r'ssthresh:(\d+)',
            'send': r'send\s+([0-9.]+)([KMG]?)bps',
            'retrans': r'retrans:\d+/(\d+)',
            'lost': r'lost:(\d+)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, raw)
            if match:
                if key == 'send':
                    val = float(match.group(1))
                    s = match.group(2)
                    if s == 'K': val *= 1e3
                    elif s == 'M': val *= 1e6
                    elif s == 'G': val *= 1e9
                    state[key] = val
                else:
                    state[key] = float(match.group(1))
            else:
                state[key] = 0.0
        state['timestamp'] = time.time()
        return state

    def _extract_retrans(self, ss_output):
        """Extract total retransmissions from ss output."""
        import re
        match = re.search(r'retrans:\d+/(\d+)', ss_output)
        return int(match.group(1)) if match else 0

    def _parse_iperf(self, json_str):
        """Parse iperf3 JSON output."""
        try:
            data = json.loads(json_str)
            end = data.get('end', {})
            sum_sent = end.get('sum_sent', {})
            return {
                'bits_per_second': sum_sent.get('bits_per_second', 0),
                'bytes': sum_sent.get('bytes', 0),
                'retransmits': sum_sent.get('retransmits', 0),
                'seconds': sum_sent.get('seconds', 0),
            }
        except:
            return {'bits_per_second': 0, 'bytes': 0, 'retransmits': 0, 'seconds': 0}


def main():
    parser = argparse.ArgumentParser(description='Artificial ECN Experiments')
    parser.add_argument('--experiment', choices=['all', 'baseline', 'basepaper', 'ours'],
                        default='all', help='Which experiment to run')
    parser.add_argument('--duration', type=int, default=120,
                        help='Experiment duration in seconds (default: 120)')
    parser.add_argument('--cc', choices=['reno', 'cubic', 'both'],
                        default='both', help='Congestion control algorithm')
    parser.add_argument('--bw', type=int, default=10,
                        help='Bottleneck bandwidth in Mbit (default: 10)')
    parser.add_argument('--delay', type=int, default=30,
                        help='Link delay in ms (default: 30)')
    parser.add_argument('--bg-flows', type=int, default=6,
                        help='Number of background flows (default: 6)')
    args = parser.parse_args()

    setLogLevel('warning')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    model_dir = os.path.join(script_dir, 'models')
    os.makedirs(results_dir, exist_ok=True)

    # Check model exists
    if not os.path.exists(os.path.join(model_dir, 'lgbm_model.joblib')):
        print("ERROR: Model files not found in models/ directory!")
        print("       Run export_model.py on Windows first, then transfer the files.")
        sys.exit(1)

    # Determine CC algorithms to test
    cc_algos = ['reno', 'cubic'] if args.cc == 'both' else [args.cc]

    # Determine experiments to run
    if args.experiment == 'all':
        experiments = ['baseline', 'basepaper_ecn', 'our_ecn']
    elif args.experiment == 'basepaper':
        experiments = ['basepaper_ecn']
    elif args.experiment == 'ours':
        experiments = ['our_ecn']
    else:
        experiments = [args.experiment]

    print("=" * 70)
    print("  ARTIFICIAL ECN — EXPERIMENT SUITE")
    print(f"  Experiments: {experiments}")
    print(f"  CC Algorithms: {cc_algos}")
    print(f"  Duration: {args.duration}s per experiment")
    print(f"  Bottleneck: {args.bw}Mbit/{args.delay}ms")
    print("=" * 70)

    all_results = {}

    for cc in cc_algos:
        all_results[cc] = {}

        for exp_name in experiments:
            runner = ExperimentRunner(
                experiment_name=exp_name,
                duration_s=args.duration,
                cc_algo=cc,
                bw_mbit=args.bw,
                delay_ms=args.delay,
                n_bg_flows=args.bg_flows,
                model_dir=model_dir,
            )

            results = runner.run()
            all_results[cc][exp_name] = results

            # Save individual result
            result_file = os.path.join(results_dir,
                                       f'{cc}_{exp_name}_results.json')
            # Remove large trace data for JSON (save separately)
            result_save = {k: v for k, v in results.items() if k != 'traces'}
            with open(result_file, 'w') as f:
                json.dump(result_save, f, indent=2, default=str)

            # Save traces as numpy
            trace_file = os.path.join(results_dir, f'{cc}_{exp_name}_traces.npz')
            np.savez(trace_file,
                     timestamps=results['traces']['timestamps'],
                     cwnd=results['traces']['cwnd'],
                     rtt=results['traces']['rtt'],
                     throughput=results['traces']['throughput'])

            print(f"\n  Saved: {result_file}")
            time.sleep(3)  # Cool down between experiments

    # --- Print comparison table ---
    print(f"\n\n{'='*80}")
    print(f"  FINAL COMPARISON — ARTIFICIAL ECN EXPERIMENTS")
    print(f"{'='*80}")

    for cc in cc_algos:
        print(f"\n  CC Algorithm: {cc.upper()}")
        print(f"  {'Experiment':<20} {'Throughput':>12} {'Retrans':>10} "
              f"{'Avg CWND':>10} {'CWND Std':>10} {'Avg RTT':>10}")
        print(f"  {'─'*74}")

        for exp_name in experiments:
            if exp_name in all_results.get(cc, {}):
                r = all_results[cc][exp_name]
                m = r['metrics']
                throughput_mbps = m['avg_throughput_bps'] / 1e6
                print(f"  {exp_name:<20} {throughput_mbps:>10.2f}M "
                      f"{m['total_retransmissions']:>10d} "
                      f"{m['avg_cwnd']:>10.1f} "
                      f"{m['cwnd_std']:>10.2f} "
                      f"{m['avg_rtt_ms']:>8.1f}ms")

        # Calculate improvements
        if 'baseline' in all_results.get(cc, {}) and 'our_ecn' in all_results.get(cc, {}):
            base = all_results[cc]['baseline']['metrics']
            ours = all_results[cc]['our_ecn']['metrics']

            tp_imp = ((ours['avg_throughput_bps'] - base['avg_throughput_bps'])
                      / max(base['avg_throughput_bps'], 1) * 100)
            rt_red = ((base['total_retransmissions'] - ours['total_retransmissions'])
                      / max(base['total_retransmissions'], 1) * 100)
            cs_imp = ((base['cwnd_std'] - ours['cwnd_std'])
                      / max(base['cwnd_std'], 1) * 100)

            print(f"\n  Our ECN Improvement over Baseline:")
            print(f"    Throughput:    {tp_imp:+.1f}%")
            print(f"    Retransmissions: {rt_red:+.1f}% reduction")
            print(f"    CWND Stability:  {cs_imp:+.1f}% improvement")

    # Save combined results
    combined_file = os.path.join(results_dir, 'all_results.json')
    # Strip traces for the combined file
    combined = {}
    for cc in all_results:
        combined[cc] = {}
        for exp in all_results[cc]:
            combined[cc][exp] = {k: v for k, v in all_results[cc][exp].items()
                                 if k not in ['traces', 'ecn_prediction_log']}
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n\n  All results saved to: {results_dir}/")
    print(f"  Combined results: {combined_file}")
    print(f"\n  Next: Run 'python3 compare_results.py' to generate figures")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""compare_results.py — Generate paper-ready figures from experiment results.

Run this after run_experiment.py has completed.
Reads from results/ directory and saves figures to results/figures/.

Usage:
    python3 compare_results.py
    python3 compare_results.py --results-dir results/
"""
import os
import sys
import json
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib is required to generate figures")
    sys.exit(1)


def load_results(results_dir):
    """Load json and npz results from directory."""
    all_results = {}
    
    # Try to load combined results
    combined_file = os.path.join(results_dir, 'all_results.json')
    if os.path.exists(combined_file):
        with open(combined_file, 'r') as f:
            all_results = json.load(f)
            
    # If not found, build from individual files
    if not all_results:
        for f in os.listdir(results_dir):
            if f.endswith('_results.json'):
                parts = f.replace('_results.json', '').split('_', 1)
                if len(parts) == 2:
                    cc, exp = parts
                    if cc not in all_results:
                        all_results[cc] = {}
                    with open(os.path.join(results_dir, f), 'r') as jf:
                        all_results[cc][exp] = json.load(jf)

    # Load traces (too large for JSON)
    traces = {}
    for f in os.listdir(results_dir):
        if f.endswith('_traces.npz'):
            parts = f.replace('_traces.npz', '').split('_', 1)
            if len(parts) == 2:
                cc, exp = parts
                if cc not in traces:
                    traces[cc] = {}
                npz = np.load(os.path.join(results_dir, f))
                traces[cc][exp] = {
                    'timestamps': npz['timestamps'],
                    'cwnd': npz['cwnd'],
                    'rtt': npz['rtt'],
                    'throughput': npz['throughput']
                }

    return all_results, traces


def generate_figures(results, traces, output_dir):
    """Generate all comparison figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'sans-serif',
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })

    colors = {
        'baseline': '#E74C3C',        # Red
        'basepaper_ecn': '#F39C12',   # Orange
        'our_ecn': '#2ECC71',         # Green
    }
    
    labels = {
        'baseline': 'Standard TCP',
        'basepaper_ecn': 'XGBoost ECN (Base Paper)',
        'our_ecn': 'LightGBM ECN (Ours)'
    }

    # Generate figures for each CC algorithm
    for cc in results.keys():
        if not traces.get(cc):
            continue
            
        print(f"\nGenerating figures for {cc.upper()}...")
        cc_dir = os.path.join(output_dir, cc)
        os.makedirs(cc_dir, exist_ok=True)
        
        # Determine which experiments exist
        exps_available = [e for e in ['baseline', 'basepaper_ecn', 'our_ecn'] 
                         if e in results[cc] and e in traces[cc]]
        
        if not exps_available:
            continue

        # --- 1. Bar Charts: Metrics Comparison ---
        if len(exps_available) >= 2:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            x_labels = [labels[e].replace(' (', '\n(') for e in exps_available]
            x_pos = np.arange(len(exps_available))
            
            # Throughput
            tp_vals = [results[cc][e]['metrics']['avg_throughput_bps']/1e6 for e in exps_available]
            bars0 = axes[0].bar(x_pos, tp_vals, color=[colors[e] for e in exps_available], edgecolor='white', linewidth=1.5)
            axes[0].set_title(f'Average Throughput ({cc.upper()})')
            axes[0].set_ylabel('Mbps')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(x_labels, rotation=15, ha='right')
            
            for bar, val in zip(bars0, tp_vals):
                axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.02,
                        f'{val:.2f}M', ha='center', va='bottom', fontweight='bold')

            # Retransmissions
            rt_vals = [results[cc][e]['metrics']['total_retransmissions'] for e in exps_available]
            bars1 = axes[1].bar(x_pos, rt_vals, color=[colors[e] for e in exps_available], edgecolor='white', linewidth=1.5)
            axes[1].set_title(f'Total Retransmissions ({cc.upper()})')
            axes[1].set_ylabel('Count (Lower is Better)')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(x_labels, rotation=15, ha='right')
            
            for bar, val in zip(bars1, rt_vals):
                axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.02,
                        f'{int(val)}', ha='center', va='bottom', fontweight='bold')

            # CWND Stability
            std_vals = [results[cc][e]['metrics']['cwnd_std'] for e in exps_available]
            bars2 = axes[2].bar(x_pos, std_vals, color=[colors[e] for e in exps_available], edgecolor='white', linewidth=1.5)
            axes[2].set_title(f'CWND Stability ({cc.upper()})')
            axes[2].set_ylabel('Std Dev (Lower is Better)')
            axes[2].set_xticks(x_pos)
            axes[2].set_xticklabels(x_labels, rotation=15, ha='right')
            
            for bar, val in zip(bars2, std_vals):
                axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.02,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            out_file = os.path.join(cc_dir, f'{cc}_metrics_comparison.png')
            plt.savefig(out_file)
            print(f"  Saved {out_file}")
            plt.close()

        # --- 2. Time Series: CWND Traces ---
        if len(exps_available) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            for e in exps_available:
                tr = traces[cc][e]
                # Plot every Nth point to make the graph readable
                step = max(1, len(tr['timestamps']) // 1000)
                
                ax.plot(tr['timestamps'][::step], tr['cwnd'][::step], 
                        color=colors[e], label=labels[e], linewidth=1.0, alpha=0.8)
                
            ax.set_title(f'Congestion Window Dynamics ({cc.upper()})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('CWND (segments)')
            ax.legend(loc='upper right', framealpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            out_file = os.path.join(cc_dir, f'{cc}_cwnd_trace.png')
            plt.savefig(out_file)
            print(f"  Saved {out_file}")
            plt.close()

            # Create individual zoomed-in CWND panels for better multi-figure view
            n_plots = len(exps_available)
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
            if n_plots == 1:
                axes = [axes]
                
            for idx, e in enumerate(exps_available):
                tr = traces[cc][e]
                step = max(1, len(tr['timestamps']) // 1500)
                
                axes[idx].plot(tr['timestamps'][::step], tr['cwnd'][::step], 
                              color=colors[e], linewidth=1.2, alpha=0.9)
                axes[idx].set_title(labels[e], fontsize=12)
                axes[idx].set_ylabel('CWND')
                axes[idx].grid(True, linestyle='--', alpha=0.5)
                axes[idx].fill_between(tr['timestamps'][::step], 0, tr['cwnd'][::step],
                                      color=colors[e], alpha=0.1)
                
            axes[-1].set_xlabel('Time (seconds)')
            plt.suptitle(f'Detailed CWND Traces ({cc.upper()})', fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()
            out_file = os.path.join(cc_dir, f'{cc}_cwnd_panels.png')
            plt.savefig(out_file)
            print(f"  Saved {out_file}")
            plt.close()

        # --- 3. Throughput CDF ---
        if len(exps_available) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for e in exps_available:
                # Convert throughput trace to Mbps and compute CDF
                tp = traces[cc][e]['throughput'] / 1e6
                if len(tp) > 0:
                    sorted_tp = np.sort(tp)
                    yvals = np.arange(len(sorted_tp)) / float(len(sorted_tp) - 1)
                    
                    ax.plot(sorted_tp, yvals, color=colors[e], label=labels[e], linewidth=2.0)
                
            ax.set_title(f'Throughput CDF ({cc.upper()})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Throughput (Mbps)')
            ax.set_ylabel('Cumulative Probability')
            ax.legend(loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            out_file = os.path.join(cc_dir, f'{cc}_throughput_cdf.png')
            plt.savefig(out_file)
            print(f"  Saved {out_file}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate ECN Comparison Figures')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing experiment results')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.results_dir)
    figures_dir = os.path.join(results_dir, 'figures')

    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory '{results_dir}' not found!")
        print("Run 'sudo python3 run_experiment.py' first.")
        sys.exit(1)

    print("=" * 60)
    print("  ARTIFICIAL ECN — FIGURE GENERATOR")
    print("=" * 60)
    
    # Load data
    results, traces = load_results(results_dir)
    
    if not results or not traces:
        print("ERROR: No valid results or traces found in the results directory.")
        sys.exit(1)
        
    print(f"Loaded results for CC algorithms: {list(results.keys())}")
    
    # Generate figures
    generate_figures(results, traces, figures_dir)
    
    print("\n" + "=" * 60)
    print(f"  All figures saved to: {figures_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""benchmark_latency.py — Measure single-sample inference latency of the deployed model.

The whole premise is "predict within one RTT (~20-30 ms)". This proves the model
inference itself is a negligible fraction of that budget — a strong systems bullet.

Usage:
    python3 src/benchmark_latency.py
    python3 src/benchmark_latency.py --n 20000 --model deployment/models/lgbm_model.joblib
"""
import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'deployment', 'models', 'lgbm_model.joblib'))
    ap.add_argument('--scaler', default=os.path.join(ROOT, 'deployment', 'models', 'scaler.joblib'))
    ap.add_argument('--n', type=int, default=10000, help='Number of timed single-sample predictions')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f'ERROR: model not found: {args.model}'); sys.exit(1)

    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler) if os.path.exists(args.scaler) else None

    n_feat = getattr(model, 'n_features_in_', None) or (
        scaler.n_features_in_ if scaler is not None else 45)
    print(f'  Model: {os.path.basename(args.model)}  | features: {n_feat}  | runs: {args.n:,}')

    rng = np.random.default_rng(42)
    sample = rng.standard_normal((1, n_feat)).astype(np.float64)

    # warmup
    for _ in range(100):
        x = scaler.transform(sample) if scaler is not None else sample
        model.predict_proba(x)

    lat = np.empty(args.n)
    for i in range(args.n):
        t0 = time.perf_counter()
        x = scaler.transform(sample) if scaler is not None else sample
        model.predict_proba(x)
        lat[i] = (time.perf_counter() - t0) * 1000.0  # ms

    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print('\n  Single-sample inference latency (scaler + predict_proba):')
    print(f'    mean : {lat.mean():.4f} ms')
    print(f'    p50  : {p50:.4f} ms')
    print(f'    p95  : {p95:.4f} ms')
    print(f'    p99  : {p99:.4f} ms')
    print(f'    max  : {lat.max():.4f} ms')
    budget = 20.0
    print(f'\n  Poll budget: {budget:.0f} ms  ->  p99 inference is '
          f'{p99 / budget * 100:.2f}% of one polling interval.')
    print('  Bullet: "low-millisecond inference, a small fraction of the 20 ms poll loop".')


if __name__ == '__main__':
    main()

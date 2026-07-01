#!/usr/bin/env python3
"""ml_ecn_controller.py — Real-Time ML-Based Artificial ECN Controller.

THE CORE SCRIPT: This is what makes our implementation real (not simulation).

How it works:
1. Polls TCP state via 'ss -ti' every 20ms (same as base paper)
2. Computes features from raw TCP metrics
3. Runs LightGBM inference → predicts if loss is imminent
4. If loss predicted → dynamically lowers RED queue thresholds on bottleneck switch
   → This causes the Linux kernel to mark packets with ECN
   → TCP stack sees ECN → naturally reduces cwnd (proactive, before actual loss)
5. After cooldown → restores original RED thresholds

This is called by run_experiment.py — not run standalone.
"""
import subprocess
import time
import re
import json
import os
import sys
import threading
import signal
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import joblib


class TCPStateMonitor:
    """Polls TCP state using ss -ti every polling_interval_ms."""

    def __init__(self, host_cmd_func, target_ip, polling_ms=20):
        """
        Args:
            host_cmd_func: Function to run commands on the Mininet host (h1.cmd)
            target_ip: IP of the receiver (to filter ss output)
            polling_ms: Polling interval in milliseconds
        """
        self.host_cmd = host_cmd_func
        self.target_ip = target_ip
        self.polling_ms = polling_ms
        self.history = []  # List of parsed TCP states
        self.max_history = 100  # Keep last 100 samples for feature engineering

    def poll_once(self):
        """Poll ss -ti once and parse the output."""
        try:
            # Use -tin for numeric output (no DNS delays)
            # Filter by port 5201 to capture main iperf3 flow only
            raw = self.host_cmd('ss -tin sport 5201 or dport 5201')
            return self._parse_ss_output(raw)
        except Exception as e:
            return None

    def _parse_ss_output(self, raw_output):
        """Parse ss -ti output into a dict of TCP metrics."""
        state = {}

        # Parse key TCP metrics from ss output
        patterns = {
            'cwnd': r'cwnd:(\d+)',
            'rtt': r'rtt:([0-9.]+)/([0-9.]+)',  # rtt/rttvar
            'ssthresh': r'ssthresh:(\d+)',
            'bytes_sent': r'bytes_sent:(\d+)',
            'bytes_acked': r'bytes_acked:(\d+)',
            'bytes_received': r'bytes_received:(\d+)',
            'segs_out': r'segs_out:(\d+)',
            'segs_in': r'segs_in:(\d+)',
            'data_segs_out': r'data_segs_out:(\d+)',
            'data_segs_in': r'data_segs_in:(\d+)',
            'send': r'send\s+([0-9.]+)([KMG]?)bps',
            'retrans': r'retrans:\d+/(\d+)',
            'lost': r'lost:(\d+)',
            'sacked': r'sacked:(\d+)',
            'rcv_space': r'rcv_space:(\d+)',
            'delivery_rate': r'delivery_rate\s+([0-9.]+)([KMG]?)bps',
            'busy': r'busy:(\d+)ms',
            'unacked': r'unacked:(\d+)',
            'rcv_rtt': r'rcv_rtt:([0-9.]+)',
            'minrtt': r'minrtt:([0-9.]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, raw_output)
            if match:
                if key == 'rtt':
                    state['rtt'] = float(match.group(1))
                    state['rttvar'] = float(match.group(2))
                elif key == 'send' or key == 'delivery_rate':
                    val = float(match.group(1))
                    suffix = match.group(2)
                    if suffix == 'K':
                        val *= 1e3
                    elif suffix == 'M':
                        val *= 1e6
                    elif suffix == 'G':
                        val *= 1e9
                    state[key] = val
                else:
                    state[key] = float(match.group(1))
            else:
                if key == 'rtt':
                    state['rtt'] = 0.0
                    state['rttvar'] = 0.0
                else:
                    state[key] = 0.0

        state['timestamp'] = time.time()
        return state

    def compute_features(self, current_state):
        """Compute engineered features from current state + history.

        Mirrors the feature engineering in enhanced_parser.py.
        """
        features = {}

        # Raw metrics
        for key in ['cwnd', 'rtt', 'rttvar', 'ssthresh', 'bytes_sent',
                     'bytes_acked', 'unacked', 'send', 'delivery_rate',
                     'rcv_space', 'minrtt']:
            features[key] = current_state.get(key, 0.0)

        # Derived features
        cwnd = current_state.get('cwnd', 1)
        rtt = current_state.get('rtt', 1)
        ssthresh = current_state.get('ssthresh', 0)
        minrtt = current_state.get('minrtt', 0.001)

        # BDP and ratios
        features['bdp'] = cwnd * 1448 / max(rtt, 0.001) * 1000
        features['cwnd_to_ssthresh'] = cwnd / max(ssthresh, 1)
        features['rtt_ratio'] = rtt / max(minrtt, 0.001)
        features['rtt_inflation'] = (rtt - minrtt) / max(minrtt, 0.001)

        # Temporal features from history
        if len(self.history) >= 2:
            prev = self.history[-1]
            features['cwnd_diff'] = cwnd - prev.get('cwnd', cwnd)
            features['rtt_diff'] = rtt - prev.get('rtt', rtt)
            features['cwnd_growth'] = features['cwnd_diff'] / max(prev.get('cwnd', 1), 1)
            features['rtt_growth'] = features['rtt_diff'] / max(prev.get('rtt', 1), 0.001)
        else:
            features['cwnd_diff'] = 0
            features['rtt_diff'] = 0
            features['cwnd_growth'] = 0
            features['rtt_growth'] = 0

        # Rolling statistics (last 5 samples)
        if len(self.history) >= 5:
            recent_cwnd = [h.get('cwnd', 0) for h in self.history[-5:]]
            recent_rtt = [h.get('rtt', 0) for h in self.history[-5:]]
            features['cwnd_mean_5'] = np.mean(recent_cwnd)
            features['cwnd_std_5'] = np.std(recent_cwnd)
            features['cwnd_max_5'] = max(recent_cwnd)
            features['cwnd_min_5'] = min(recent_cwnd)
            features['rtt_mean_5'] = np.mean(recent_rtt)
            features['rtt_std_5'] = np.std(recent_rtt)
            features['cwnd_range_5'] = features['cwnd_max_5'] - features['cwnd_min_5']
            features['cwnd_drop_from_peak'] = (features['cwnd_max_5'] - cwnd) / max(features['cwnd_max_5'], 1)
        else:
            for key in ['cwnd_mean_5', 'cwnd_std_5', 'cwnd_max_5', 'cwnd_min_5',
                         'rtt_mean_5', 'rtt_std_5', 'cwnd_range_5', 'cwnd_drop_from_peak']:
                features[key] = 0.0

        # Rolling statistics (last 10 samples)
        if len(self.history) >= 10:
            recent_cwnd = [h.get('cwnd', 0) for h in self.history[-10:]]
            recent_rtt = [h.get('rtt', 0) for h in self.history[-10:]]
            features['cwnd_mean_10'] = np.mean(recent_cwnd)
            features['cwnd_std_10'] = np.std(recent_cwnd)
            features['rtt_mean_10'] = np.mean(recent_rtt)
            features['rtt_std_10'] = np.std(recent_rtt)
            features['cwnd_trend_10'] = (cwnd - recent_cwnd[0]) / max(abs(recent_cwnd[0]), 1)
            features['rtt_trend_10'] = (rtt - recent_rtt[0]) / max(abs(recent_rtt[0]), 0.001)
        else:
            for key in ['cwnd_mean_10', 'cwnd_std_10', 'rtt_mean_10', 'rtt_std_10',
                         'cwnd_trend_10', 'rtt_trend_10']:
                features[key] = 0.0

        # Interaction features
        features['cwnd_x_rtt'] = cwnd * rtt
        features['send_x_rtt'] = current_state.get('send', 0) * rtt

        return features


class ArtificialECNController:
    """The main ECN controller — loads model, monitors TCP, injects ECN."""

    def __init__(self, model_dir, switch_cmd_func, intf_name,
                 orig_red_min=5000, orig_red_max=15000, bw_mbit=10,
                 model_type='lgbm',
                 trigger_streak=3, refractory_ticks=10,
                 min_factor=0.4, max_factor=0.7,
                 cooldown_ticks=5, marking_prob=0.2):
        """
        Args:
            model_dir: Path to models/ directory (contains lgbm_model.joblib etc.)
            switch_cmd_func: Function to run commands on Mininet switch (s1.cmd)
            intf_name: Bottleneck interface name on s1
            orig_red_min/max: Original RED queue thresholds
            bw_mbit: Bottleneck bandwidth
            model_type: 'lgbm' for our model, 'xgb' for base paper comparison

        Stability controls (added to stop the controller pinning cwnd at the
        minimum when predictions fire on every poll):
            trigger_streak:   require this many CONSECUTIVE positive predictions
                              before injecting ECN (debounce against single spikes)
            refractory_ticks: minimum ticks to wait after restoring before a new
                              ECN episode may start (prevents always-on throttling)
            min_factor/max_factor: clamp range for the proportional reduction.
                              min_factor=0.4 is far gentler than the old 0.2 floor.
            cooldown_ticks:   ticks ECN stays active before thresholds are restored
            marking_prob:     RED ECN marking probability while ECN is active
        """
        self.switch_cmd = switch_cmd_func
        self.intf_name = intf_name
        self.orig_red_min = orig_red_min
        self.orig_red_max = orig_red_max
        self.bw_mbit = bw_mbit
        self.model_type = model_type
        self.trigger_streak = trigger_streak
        self.refractory_ticks = refractory_ticks
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.marking_prob = marking_prob

        # Load model and config
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path) as f:
            self.config = json.load(f)

        if model_type == 'lgbm':
            model_path = os.path.join(model_dir, 'lgbm_model.joblib')
            self.threshold = self.config['lgbm_threshold']
        else:
            model_path = os.path.join(model_dir, 'xgb_model.joblib')
            self.threshold = self.config.get('xgb_threshold', 0.5)

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        self.feature_names = self.config['feature_names']

        print(f"  Loaded {model_type.upper()} model "
              f"(threshold={self.threshold:.2f}, "
              f"features={len(self.feature_names)})")

        # ECN state tracking
        self.ecn_active = False
        self.ecn_cooldown = 0  # Ticks remaining before restoring thresholds
        self.cooldown_ticks = cooldown_ticks  # Active duration per ECN episode
        self.pos_streak = 0    # Consecutive positive predictions (debounce)
        self.refractory = 0    # Ticks remaining before a new episode may start

        # Metrics
        self.total_predictions = 0
        self.positive_predictions = 0
        self.ecn_signals_sent = 0
        self.prediction_log = []

    def predict(self, features_dict):
        """Run ML prediction on current TCP features.

        Returns (predicted_loss: bool, probability: float)
        """
        # Build feature vector in the correct order
        feature_vector = []
        for fname in self.feature_names:
            feature_vector.append(features_dict.get(fname, 0.0))

        X = np.array([feature_vector])

        # Handle any NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        prob = self.model.predict_proba(X_scaled)[0, 1]
        predicted_loss = prob >= self.threshold

        self.total_predictions += 1
        if predicted_loss:
            self.positive_predictions += 1

        return predicted_loss, prob

    def inject_ecn(self, confidence):
        """Inject artificial ECN by lowering RED queue thresholds.

        PROPORTIONAL response (our improvement over base paper):
        - Higher confidence → more aggressive threshold reduction
        - This causes the RED queue to mark more packets with ECN
        - The kernel TCP stack then reduces cwnd naturally
        """
        if self.ecn_active:
            return  # Already in ECN mode

        # Proportional reduction based on confidence, clamped to [min_factor, max_factor].
        # Gentler floor (0.4) than before (0.2) so a confident prediction tightens the
        # queue without strangling the flow down to cwnd=2.
        reduction_factor = 1.0 - (confidence - 0.3) * 1.2
        reduction_factor = max(self.min_factor, min(self.max_factor, reduction_factor))

        new_min = int(self.orig_red_min * reduction_factor)
        new_max = int(self.orig_red_max * reduction_factor)

        # Apply new RED thresholds (change handle 10: = the RED qdisc)
        self.switch_cmd(
            f'tc qdisc change dev {self.intf_name} handle 10: red '
            f'limit 200000 min {new_min} max {new_max} '
            f'avpkt 1000 bandwidth {self.bw_mbit}mbit ecn probability {self.marking_prob}'
        )

        self.ecn_active = True
        self.ecn_cooldown = self.cooldown_ticks
        self.ecn_signals_sent += 1

    def restore_normal(self):
        """Restore original RED thresholds."""
        self.switch_cmd(
            f'tc qdisc change dev {self.intf_name} handle 10: red '
            f'limit 200000 min {self.orig_red_min} max {self.orig_red_max} '
            f'avpkt 1000 bandwidth {self.bw_mbit}mbit ecn probability 0.1'
        )
        self.ecn_active = False

    def tick(self, features_dict, timestamp):
        """Called every polling interval. Makes prediction and acts on it.

        Returns dict with prediction info for logging.
        """
        # Handle ECN cooldown → restore, then open a refractory window so we don't
        # immediately re-trigger on the next noisy positive prediction.
        if self.ecn_active:
            self.ecn_cooldown -= 1
            if self.ecn_cooldown <= 0:
                self.restore_normal()
                self.refractory = self.refractory_ticks
        elif self.refractory > 0:
            self.refractory -= 1

        # Make prediction
        predicted_loss, probability = self.predict(features_dict)

        # Debounce: only a sustained run of positive predictions counts as real
        # congestion, not a single-poll spike (common under VM timing jitter).
        if predicted_loss:
            self.pos_streak += 1
        else:
            self.pos_streak = 0

        # Act on prediction
        action = 'none'
        if self.ecn_active:
            action = 'ecn_active'
        elif self.refractory > 0:
            action = 'refractory'
        elif self.pos_streak >= self.trigger_streak:
            self.inject_ecn(probability)
            self.pos_streak = 0
            action = 'ecn_injected'

        log_entry = {
            'timestamp': timestamp,
            'predicted_loss': bool(predicted_loss),
            'probability': float(probability),
            'action': action,
            'ecn_active': self.ecn_active,
            'cwnd': features_dict.get('cwnd', 0),
            'rtt': features_dict.get('rtt', 0),
        }
        self.prediction_log.append(log_entry)

        return log_entry

    def get_summary(self):
        """Return summary metrics."""
        return {
            'total_predictions': self.total_predictions,
            'positive_predictions': self.positive_predictions,
            'ecn_signals_sent': self.ecn_signals_sent,
            'prediction_rate': (self.positive_predictions / max(self.total_predictions, 1)),
            'model_type': self.model_type,
            'threshold': self.threshold,
        }

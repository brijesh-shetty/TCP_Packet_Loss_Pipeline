#!/usr/bin/env python3
"""main.py — TCP Packet Loss Prediction Pipeline Entry Point.

Usage:
    python main.py --parse           Parse raw data → enhanced dataset
    python main.py --train           Train models on enhanced dataset
    python main.py --all             Parse + Train end-to-end
    python main.py --dataset FILE    Use custom dataset for training
    python main.py --no-optuna       Skip Optuna hyperparameter tuning
"""
import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description='TCP Packet Loss Prediction — ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all               Full pipeline (parse + train)
  python main.py --parse             Parse raw data only
  python main.py --train             Train on existing enhanced_dataset.csv
  python main.py --train --no-optuna Fast training without Optuna tuning
  python main.py --dataset dataset_optimized.csv  Train on specific dataset
        """
    )
    parser.add_argument('--parse', action='store_true', help='Parse raw data into enhanced dataset')
    parser.add_argument('--train', action='store_true', help='Train models on dataset')
    parser.add_argument('--all', action='store_true', help='Parse + Train end-to-end')
    parser.add_argument('--dataset', type=str, default=None, help='Custom dataset CSV path')
    parser.add_argument('--no-optuna', action='store_true', help='Skip Optuna tuning (faster)')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials (default: 30)')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')),
                        help='Data directory')

    args = parser.parse_args()

    # Default to --all if no action specified
    if not args.parse and not args.train and not args.all and not args.dataset:
        args.all = True

    data_dir = args.data_dir
    start_time = time.time()

    print('=' * 75)
    print('  TCP PACKET LOSS PREDICTION')
    print('  Based on: "Real-time Prediction of TCP Packet Loss using ML"')
    print('  IEEE Ref: 10738808 | Enhanced Implementation')
    print('=' * 75)

    dataset_path = None

    # --- PARSE ---
    if args.parse or args.all:
        print('\n>>> STEP 1: PARSING RAW DATA')
        print('-' * 50)

        from enhanced_parser import main as parse_main
        dataset_path = parse_main()

        print(f'\n  Parsing completed in {time.time()-start_time:.1f}s')

    # --- TRAIN ---
    if args.train or args.all or args.dataset:
        print('\n>>> STEP 2: TRAINING MODELS')
        print('-' * 50)

        from enhanced_train import load_and_prepare, train_and_evaluate

        # Determine dataset path
        if args.dataset:
            dataset_path = args.dataset
            if not os.path.isabs(dataset_path):
                dataset_path = os.path.join(data_dir, dataset_path)
        elif dataset_path is None:
            dataset_path = os.path.join(data_dir, 'enhanced_dataset.csv')

        if not os.path.exists(dataset_path):
            print(f'  ERROR: Dataset not found: {dataset_path}')
            print('  Run with --parse first, or specify --dataset')
            sys.exit(1)

        use_optuna = not args.no_optuna

        # Redirect to log file
        import io
        out_file = os.path.join(data_dir, 'enhanced_results.txt')
        log_f = open(out_file, 'w', encoding='utf-8')

        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()

        sys.stdout = Tee(sys.__stdout__, log_f)

        # Train on combined dataset
        print(f'\n  Dataset: {os.path.basename(dataset_path)}')
        print(f'  Optuna: {"ON" if use_optuna else "OFF"}')
        if use_optuna:
            print(f'  Trials: {args.trials}')
        print()

        X, y = load_and_prepare(dataset_path)
        results = train_and_evaluate(
            X, y, data_dir, 'Combined',
            use_optuna=use_optuna, n_trials=args.trials
        )

        # Also train per-CC if available
        for cc in ['reno', 'cubic']:
            cc_path = os.path.join(data_dir, f'{cc}_enhanced.csv')
            if os.path.exists(cc_path):
                import pandas as pd
                cc_df = pd.read_csv(cc_path)
                if len(cc_df) > 100:
                    print(f'\n\n>>> {cc.upper()} DATASET')
                    X_cc, y_cc = load_and_prepare(cc_path)
                    train_and_evaluate(
                        X_cc, y_cc, data_dir, cc.upper(),
                        use_optuna=False
                    )

        # Save JSON results
        import json
        json_path = os.path.join(data_dir, 'enhanced_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        log_f.close()
        sys.stdout = sys.__stdout__

        print(f'\n  Results saved to:')
        print(f'    {out_file}')
        print(f'    {json_path}')

    elapsed = time.time() - start_time
    print(f'\n{"="*75}')
    print(f'  DONE! Total time: {elapsed/60:.1f} minutes')
    print(f'{"="*75}')


if __name__ == '__main__':
    main()

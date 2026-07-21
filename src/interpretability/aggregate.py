"""
Aggregate results across seeds and folds for a given experiment tag.

Usage:
  python -m src.interpretability.aggregate --tag bbvalfix_d07_lr5e4_6seeds
  python -m src.interpretability.aggregate --tag bbvalfix_d07_lr5e4_6seeds --save
"""
import os
import sys
import glob
import json
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from interpretability.base import RESULTS_ROOT


def find_experiment_dirs(tag):
    pattern = os.path.join(RESULTS_ROOT, f'mhcmattention_sngkf_seed*_iseed_*_outerf5_innerf5_tag{tag}')
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f'No experiment dirs found for tag="{tag}"')
        print(f'Pattern: {pattern}')
    return dirs


def parse_seed_from_dir(dirname):
    parts = os.path.basename(dirname).split('_')
    for i, p in enumerate(parts):
        if p == 'seed' and i + 1 < len(parts):
            return int(parts[i + 1])
    return None


def main():
    p = argparse.ArgumentParser(description='Aggregate results across seeds')
    p.add_argument('--tag', required=True, help='Experiment tag')
    p.add_argument('--save', action='store_true', help='Save CSV to figures dir')
    args = p.parse_args()

    dirs = find_experiment_dirs(args.tag)
    if not dirs:
        sys.exit(1)

    print(f'\nFound {len(dirs)} experiment directories\n')

    all_rows = []
    for exp_dir in dirs:
        seed = parse_seed_from_dir(os.path.basename(exp_dir))
        results_path = os.path.join(exp_dir, 'results.json')
        if not os.path.exists(results_path):
            print(f'  No results.json in {exp_dir} — skipping')
            continue

        with open(results_path) as f:
            results = json.load(f)

        folds = results.get('folds', [])
        for fold_data in folds:
            row = {
                'seed': seed,
                'fold': fold_data['fold'],
                'test_bacc': fold_data['test_bacc'],
                'test_auc': fold_data.get('test_auc', ''),
                'test_acc': fold_data['test_metrics']['acc'],
                'test_f1': fold_data['test_metrics']['f1'],
                'test_sens': fold_data['test_metrics']['sens'],
                'test_spec': fold_data['test_metrics']['spec'],
                'inner_cv_val_bacc': fold_data['inner_cv_val_bacc'],
                'eeg_backbone_val_loss': fold_data.get('eeg_backbone_val_loss', ''),
                'aud_backbone_val_loss': fold_data.get('aud_backbone_val_loss', ''),
                'n_test': fold_data.get('n_test', ''),
                'final_fusion_epochs': fold_data.get('final_fusion_epochs', ''),
            }
            all_rows.append(row)

    if not all_rows:
        print('  No fold data found')
        sys.exit(1)

    def fmt_val(v, spec='>8.4f', fallback=''):
        return f'{v:{spec}}' if isinstance(v, (int, float)) else fallback

    # Print table
    print(f"{'seed':>4s} | {'fold':>4s} | {'bacc':>6s} | {'auc':>6s} | {'acc':>6s} | {'f1':>6s} | {'sens':>6s} | {'spec':>6s} | {'inner_vl':>8s} | {'eeg_bb':>8s} | {'aud_bb':>8s}")
    print('-' * 90)
    for row in all_rows:
        print(f"{row['seed']:>4d} | {row['fold']:>4d} | {row['test_bacc']:>6.3f} | "
              f"{fmt_val(row['test_auc'], '>6.3f'):>6s} | "
              f"{row['test_acc']:>6.3f} | {row['test_f1']:>6.3f} | "
              f"{row['test_sens']:>6.3f} | {row['test_spec']:>6.3f} | "
              f"{fmt_val(row['inner_cv_val_bacc'], '>8.3f'):>8s} | "
              f"{fmt_val(row['eeg_backbone_val_loss'], '>8.4f'):>8s} | "
              f"{fmt_val(row['aud_backbone_val_loss'], '>8.4f'):>8s}")

    # Per-seed aggregate
    print('\n\n=== Per-seed summary ===')
    seeds = sorted(set(r['seed'] for r in all_rows))
    seed_summaries = []
    for s in seeds:
        s_rows = [r for r in all_rows if r['seed'] == s]
        baccs = [r['test_bacc'] for r in s_rows]
        aucs = [r['test_auc'] for r in s_rows if isinstance(r['test_auc'], (int, float))]
        print(f"  seed {s:>4d}: bacc={np.mean(baccs):.3f}±{np.std(baccs):.3f}  "
              f"auc={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
              f"n_folds={len(s_rows)}")
        seed_summaries.append({
            'seed': s,
            'bacc_mean': float(np.mean(baccs)),
            'bacc_std': float(np.std(baccs)),
            'auc_mean': float(np.mean(aucs)) if aucs else '',
            'n_folds': len(s_rows),
        })

    # Global aggregate
    all_baccs = [r['test_bacc'] for r in all_rows]
    all_aucs = [r['test_auc'] for r in all_rows if isinstance(r['test_auc'], (int, float))]
    print(f'\n  GLOBAL: bacc={np.mean(all_baccs):.3f}±{np.std(all_baccs):.3f}  '
          f'auc={np.mean(all_aucs):.3f}±{np.std(all_aucs):.3f}  '
          f'n_folds={len(all_baccs)}  n_seeds={len(seeds)}')

    # Save CSV
    if args.save:
        out_dir = os.path.join(RESULTS_ROOT, f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'aggregated_results.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
        print(f'\nSaved CSV: {csv_path}')


if __name__ == '__main__':
    main()

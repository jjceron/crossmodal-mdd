"""
Launch the 7-config ablation table for Stage 2 (fusion + self-attention).
Requires run_crossmodal_extract.py to have completed first (cache in cache/crossmodal_features/).

Usage:
  py src/training/run_ablation_grid.py [--cache cache/crossmodal_features] [--sequential]
"""
import sys
import os
import subprocess
import argparse

CONFIGS = [
    # (fusion, n_self_attn_layers, bottleneck_dim, param_control, label_suffix)
    ('concat',    0, None, False, 'baseline_concat'),
    ('gating',    0, None, False, 'gating'),
    ('cross_attn', 0, None, False, 'cross_only'),
    ('cross_attn', 1, None, False, 'cross_self_1L'),
    ('cross_attn', 2, None, False, 'cross_self_2L'),
    ('cross_attn', 0, None, True,  'cross_paramctrl'),
    ('cross_attn', 1, 64,    False, 'cross_self_1L_bn64'),
]

SCRIPT = os.path.join(os.path.dirname(__file__), 'run_crossmodal.py')
DEFAULT_CACHE = 'cache/crossmodal_features'

BASE_ARGS = [
    sys.executable, SCRIPT,
    '--epochs', '50',
    '--lr', '5e-4',
    '--wd', '1e-3',
    '--patience', '15',
    '--bs', '8',
    '--max-windows', '50',
]


def main():
    parser = argparse.ArgumentParser(description='Run 7-config ablation grid')
    parser.add_argument('--cache', type=str, default=DEFAULT_CACHE,
                        help='Path to Stage 1 cache directory')
    parser.add_argument('--sequential', action='store_true',
                        help='Run configs sequentially (default: one per GPU if available)')
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        print(f'ERROR: Cache not found at {args.cache}')
        print('  Run: py src/training/run_crossmodal_extract.py first')
        sys.exit(1)

    print('CrossModalAttention Ablation Grid (7 configs)')
    print(f'Cache: {args.cache}')
    print(f'Sequential: {args.sequential}')
    print(f'{"=" * 60}')

    processes = []
    for fusion, n_self, bn, param_ctrl, label in CONFIGS:
        cmd = BASE_ARGS.copy() + [
            '--fusion', fusion,
            '--n-self-attn-layers', str(n_self),
            '--from-cache', args.cache,
        ]
        if bn is not None:
            cmd += ['--bottleneck-dim', str(bn)]
        if param_ctrl:
            cmd += ['--param-control']

        label_padded = f'{label:>25s}'
        cmd_str = ' '.join(cmd[-8:])
        print(f'  {label_padded} | {cmd_str}')

        if args.sequential:
            print(f'\n  Running {label}...')
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f'  {label} FAILED (rc={result.returncode})')
        else:
            p = subprocess.Popen(cmd)
            processes.append((label, p))

    if not args.sequential:
        print(f'\nWaiting for {len(processes)} processes...')
        for label, p in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAILED (rc={p.returncode})'
            print(f'  {label}: {status}')

    print(f'\n{"=" * 60}')
    print('Done. Results in outputs/results/crossmodal/')
    print('Consolidated: outputs/results/crossmodal/consolidated_results.csv')


if __name__ == '__main__':
    main()

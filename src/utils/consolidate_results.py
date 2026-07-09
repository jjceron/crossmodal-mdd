"""
Consolidate all results into a single CSV: Fase 1 (unimodal) + Fase 2 (crossmodal).

Usage:
  py src/training/consolidate_results.py [--crossmodal-dir outputs/results/crossmodal]
                                         [--backbone-cache cache/crossmodal_features]
                                         [--output outputs/results/crossmodal/consolidated_results.csv]
"""
import sys
import os
import json
import argparse
import glob
import numpy as np

EEG_DIR = 'outputs/results/classical_dl/eeg'
AUDIO_DIR = 'outputs/results/classical_dl/audio'
CROSSMODAL_DIR = 'outputs/results/crossmodal'
BACKBONE_CACHE = 'cache/crossmodal_features'
OUTPUT = 'outputs/results/crossmodal/consolidated_results.csv'


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _parse_fase1(directory, modality):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, '*_curves.json'))):
        continue
    for path in sorted(glob.glob(os.path.join(directory, '*.json'))):
        if '_curves' in path:
            continue
        data = _load_json(path)
        s = data.get('summary', {})
        model_key = data.get('model_key', 'unknown')
        bottleneck = ''
        if 'bn' in model_key or 'bottleneck' in str(data.get('args', {})):
            bn_val = data.get('args', {}).get('bottleneck_dim', '')
            if bn_val:
                bottleneck = f'bn{bn_val}'
        config_name = f'{modality}_{model_key}'
        if bottleneck:
            config_name += f'_{bottleneck}'
        rows.append({
            'config_name': config_name,
            'probe_type': 'unimodal',
            'fusion': '-',
            'n_self_attn': '-',
            'bottleneck_dim': str(data.get('args', {}).get('bottleneck_dim', '-')),
            'max_windows': '-',
            'bacc_mean': s.get('bacc_mean', ''),
            'bacc_std': s.get('bacc_std', ''),
            'auc_mean': s.get('bacc_opt_mean', s.get('auc_mean', '')),
            'auc_std': s.get('bacc_opt_std', s.get('auc_std', '')),
            'n_subjects': data.get('n_subjects', ''),
            'n_mdd': data.get('n_mdd', ''),
            'n_hc': data.get('n_hc', ''),
            'backbone_eeg_val_bacc': '',
            'backbone_audio_val_bacc': '',
        })
    return rows


def _parse_fase2(crossmodal_dir, backbone_cache):
    rows = []
    if not os.path.exists(crossmodal_dir):
        return rows

    # Load backbone metrics per fold
    bb_metrics = {}
    bb_path = os.path.join(backbone_cache, 'fold_metrics.json')
    if os.path.exists(bb_path):
        bb_metrics = _load_json(bb_path)

    eeg_bb_baccs = [v.get('eeg_val_bacc', '') for v in bb_metrics.values() if v.get('eeg_val_bacc') is not None]
    aud_bb_baccs = [v.get('audio_val_bacc', '') for v in bb_metrics.values() if v.get('audio_val_bacc') is not None]
    eeg_bb_mean = f'{float(np.mean(eeg_bb_baccs)):.4f}' if eeg_bb_baccs else ''
    aud_bb_mean = f'{float(np.mean(aud_bb_baccs)):.4f}' if aud_bb_baccs else ''

    # Walk through crossmodal_dir and its subdirectories for results.json
    for cfg_dir in sorted(glob.glob(os.path.join(crossmodal_dir, '**/results.json'), recursive=True)):
        cfg_parent = os.path.dirname(cfg_dir)
        data = _load_json(cfg_dir)
        s = data.get('summary', {})
        args = data.get('args', {})
        pt = data.get('probe_type', 'fusion_probe')
        rows.append({
            'config_name': data.get('config_name', os.path.basename(cfg_parent)),
            'probe_type': pt,
            'fusion': args.get('fusion', ''),
            'n_self_attn': str(args.get('n_self_attn_layers', '')),
            'bottleneck_dim': str(args.get('bottleneck_dim', '-')),
            'max_windows': str(args.get('max_windows', '')),
            'bacc_mean': s.get('bacc_mean', ''),
            'bacc_std': s.get('bacc_std', ''),
            'auc_mean': s.get('auc_mean', ''),
            'auc_std': s.get('auc_std', ''),
            'n_subjects': '',
            'n_mdd': '',
            'n_hc': '',
            'backbone_eeg_val_bacc': eeg_bb_mean if 'fusion_probe' in pt else '',
            'backbone_audio_val_bacc': aud_bb_mean if 'fusion_probe' in pt else '',
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Consolidate results CSV')
    parser.add_argument('--crossmodal-dir', default=CROSSMODAL_DIR)
    parser.add_argument('--backbone-cache', default=BACKBONE_CACHE)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--no-unimodal', action='store_true',
                        help='Skip Fase 1 unimodal rows')
    args = parser.parse_args()

    all_rows = []
    if not args.no_unimodal:
        all_rows += _parse_fase1(EEG_DIR, 'eeg')
        all_rows += _parse_fase1(AUDIO_DIR, 'audio')
    all_rows += _parse_fase2(args.crossmodal_dir, args.backbone_cache)

    if not all_rows:
        print('No results found.')
        sys.exit(1)

    header = ('config_name,probe_type,fusion,n_self_attn,bottleneck_dim,'
              'max_windows,bacc_mean,bacc_std,auc_mean,auc_std,'
              'n_subjects,n_mdd,n_hc,'
              'backbone_eeg_val_bacc,backbone_audio_val_bacc')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(header + '\n')
        for r in all_rows:
            row = ','.join(str(r.get(k, '')) for k in
                          ['config_name', 'probe_type', 'fusion', 'n_self_attn',
                           'bottleneck_dim', 'max_windows', 'bacc_mean', 'bacc_std',
                           'auc_mean', 'auc_std', 'n_subjects', 'n_mdd', 'n_hc',
                           'backbone_eeg_val_bacc', 'backbone_audio_val_bacc'])
            f.write(row + '\n')

    print(f'Saved: {args.output}  ({len(all_rows)} rows)')
    print(f'\n{"=" * 70}')
    print(f'  {"Row":<30s} {"bacc":>8s} {"auc":>8s}')
    print(f'  {"-" * 30} {"-" * 8} {"-" * 8}')
    for r in all_rows:
        name = r['config_name']
        if len(name) > 28:
            name = name[:25] + '...'
        print(f'  {name:<30s} {str(r.get("bacc_mean", "")):>8s} {str(r.get("auc_mean", "")):>8s}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()

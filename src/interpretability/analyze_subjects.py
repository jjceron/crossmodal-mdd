"""
Subject-level prediction analysis: per-subject logits + confusion.

Usage:
  python -m src.interpretability.analyze_subjects --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1
  python -m src.interpretability.analyze_subjects --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --global
  python -m src.interpretability.analyze_subjects --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1 --save
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from interpretability.base import (
    load_eeg_cache, load_audio_cache, load_mapping,
    build_paired_subjects, build_models, load_checkpoint,
    find_checkpoint_dir, extract_all_features,
    device, FIGURES_ROOT
)


def _get_subject_ids(tag, seed):
    """Return list of fold results with test_subject_ids per fold."""
    ckpt_dir = find_checkpoint_dir(tag, seed)
    with open(os.path.join(ckpt_dir, 'results.json')) as f:
        results = json.load(f)
    folds_info = []
    for fd in results['folds']:
        folds_info.append({
            'fold': fd['fold'],
            'test_subjects': fd['test_subjects'],
        })
    return folds_info


def main():
    p = argparse.ArgumentParser(description='Subject-level prediction analysis')
    p.add_argument('--tag', required=True, help='Experiment tag')
    p.add_argument('--seed', type=int, default=42, help='Seed')
    p.add_argument('--fold', type=int, default=1, help='Fold (ignored if --global)')
    p.add_argument('--global_', action='store_true', dest='global_',
                   help='Aggregate across all folds')
    p.add_argument('--save', action='store_true', help='Save figure')
    p.add_argument('--cache-suffix', type=str, default='64ch',
                   help='EEG cache suffix (e.g. 64ch, mddk64). Default: 64ch')
    args = p.parse_args()
    if args.save:
        matplotlib.use('Agg')

    print('Loading data...')
    (eeg_data, eeg_labels, eeg_cods), (aud_data, aud_labels, aud_cods), mapping = \
        load_eeg_cache(suffix=args.cache_suffix), load_audio_cache(), load_mapping()
    pairs, eeg_subjs, aud_subjs = build_paired_subjects(
        eeg_data, eeg_labels, eeg_cods, aud_data, aud_labels, aud_cods, mapping)

    if args.global_:
        folds_to_run = _get_subject_ids(args.tag, args.seed)
    else:
        folds_to_run = [{'fold': args.fold, 'test_subjects': []}]

    all_logits = []
    all_labels = []
    all_correct = []

    for finfo in folds_to_run:
        fi = finfo['fold']
        if args.global_:
            test_subj_ids = finfo['test_subjects']
        else:
            ckpt_dir = find_checkpoint_dir(args.tag, args.seed)
            with open(os.path.join(ckpt_dir, 'results.json')) as f:
                results = json.load(f)
            test_subj_ids = results['folds'][fi - 1]['test_subjects']

        test_indices = [i for i, (eid, _, _) in enumerate(pairs) if eid in test_subj_ids]
        if not test_indices:
            continue
        sub_pairs = [pairs[i] for i in test_indices]
        labels = np.array([p[2] for p in sub_pairs])

        print(f'  Fold {fi}: {len(test_indices)} test subjects')

        ckpt = load_checkpoint(args.tag, args.seed, fi)
        eeg_model, aud_model, fusion_model = build_models(ckpt)

        Z_e, Z_a, masks = extract_all_features(eeg_model, aud_model, sub_pairs, eeg_subjs, aud_subjs)

        t_e = torch.FloatTensor(Z_e).to(device)
        t_a = torch.FloatTensor(Z_a).to(device)
        t_m = torch.FloatTensor(masks).to(device)

        with torch.no_grad():
            logits = fusion_model(t_e, t_a, t_m).cpu().numpy()

        all_logits.extend(logits.tolist())
        all_labels.extend(labels.tolist())
        preds = (logits > 0).astype(int)
        all_correct.extend((preds == labels).tolist())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_correct = np.array(all_correct)

    print(f'  Total subjects: {len(all_labels)}')

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ['#3498db', '#e74c3c']

    # Panel 1: Per-subject logits
    ax = axes[0]
    x = np.arange(len(all_labels))

    for i in range(len(all_labels)):
        color = colors[all_labels[i]]
        marker = 'o' if all_correct[i] else 'x'
        ax.scatter(i, all_logits[i], c=color, marker=marker, s=80,
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.vlines(i, 0, all_logits[i], color=color, alpha=0.2, linewidth=0.8)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(len(all_labels))], fontsize=6, rotation=45)
    ax.set_ylabel('Logit', fontsize=10)
    ax.set_xlabel('Test subject', fontsize=10)
    ax.set_title('Per-subject prediction logits', fontsize=11)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=8, label='HC correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=8, label='MDD correct'),
        Line2D([0], [0], marker='x', color='black', markersize=8, label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    acc = all_correct.mean()
    ax.text(0.02, 0.95, f'Acc: {acc:.0%} ({int(all_correct.sum())}/{len(all_correct)})',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: Confusion matrix
    ax = axes[1]
    preds_all = (all_logits > 0).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(all_labels, preds_all):
        cm[t, p] += 1

    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['HC', 'MDD'], fontsize=9)
    ax.set_yticklabels(['HC', 'MDD'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    bacc = (cm[0, 0] / max(cm[0].sum(), 1) + cm[1, 1] / max(cm[1].sum(), 1)) / 2
    ax.set_title(f'Confusion matrix  (BACC={bacc:.3f})', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    mode = 'global' if args.global_ else f'fold{args.fold}'
    plt.suptitle(f'Subject-level predictions | seed={args.seed} {mode}', fontsize=12)
    plt.tight_layout()

    if args.save:
        out_dir = os.path.join(FIGURES_ROOT, 'subjects', f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'seed{args.seed}_{mode}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

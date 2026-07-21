"""
Per-window prediction importance analysis: box plots + class density.

Usage:
  python -m src.interpretability.analyze_windows --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1
  python -m src.interpretability.analyze_windows --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --global
  python -m src.interpretability.analyze_windows --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1 --save
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
    find_checkpoint_dir, encode_eeg, encode_audio,
    device, FIGURES_ROOT
)


def _get_fold_subjects(tag, seed):
    ckpt_dir = find_checkpoint_dir(tag, seed)
    with open(os.path.join(ckpt_dir, 'results.json')) as f:
        results = json.load(f)
    return [(fd['fold'], fd['test_subjects']) for fd in results['folds']]


def _get_window_logits(eeg_model, aud_model, fusion_model, sub_pairs, eeg_subjs, aud_subjs):
    all_logits = []
    all_labels = []
    for eid, aid, label in sub_pairs:
        we = eeg_subjs[eid]['windows'].copy()
        wa = aud_subjs[aid]['windows'].copy()
        K = min(len(we), len(wa))
        we, wa = we[:K], wa[:K]
        we = (we - we.mean()) / (we.std() + 1e-8)
        wa = (wa - wa.mean()) / (wa.std() + 1e-8)

        ze = encode_eeg(eeg_model, we)
        za = encode_audio(aud_model, wa)

        t_e = torch.FloatTensor(ze).unsqueeze(0).to(device)
        t_a = torch.FloatTensor(za).unsqueeze(0).to(device)

        with torch.no_grad():
            win_logits = fusion_model.forward_per_window(t_e, t_a)

        logits = win_logits.squeeze(0).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(label)
    return all_logits, all_labels


def _plot(all_logits, all_labels, title_suffix):
    colors = ['#3498db', '#e74c3c']
    n_subj = len(all_logits)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Box plot per subject
    ax = axes[0]
    positions = np.arange(n_subj)
    bp = ax.boxplot(all_logits, positions=positions, widths=0.5, patch_artist=True,
                     showfliers=False)

    for patch, label in zip(bp['boxes'], all_labels):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.7)

    for i in range(n_subj):
        ax.axvspan(i - 0.5, i + 0.5, facecolor=colors[all_labels[i]], alpha=0.08)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'S{i+1}' for i in range(n_subj)], fontsize=7, rotation=45)
    ax.set_xlabel('Subject', fontsize=10)
    ax.set_ylabel('Window logit', fontsize=10)
    ax.set_title('Per-subject window logit distribution', fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], alpha=0.7, label='HC'),
                       Patch(facecolor=colors[1], alpha=0.7, label='MDD')]
    ax.legend(handles=legend_elements, fontsize=9)

    # Panel 2: Density by class
    ax = axes[1]
    mdd_logits = np.concatenate([logits for logits, lbl in zip(all_logits, all_labels) if lbl == 1])
    hc_logits = np.concatenate([logits for logits, lbl in zip(all_logits, all_labels) if lbl == 0])

    bins = np.linspace(-3, 3, 40)
    if len(hc_logits) > 0:
        ax.hist(hc_logits, bins=bins, density=True, alpha=0.5, color=colors[0],
                label=f'HC (n={len(hc_logits)} wins)', edgecolor='white', linewidth=0.3)
    if len(mdd_logits) > 0:
        ax.hist(mdd_logits, bins=bins, density=True, alpha=0.5, color=colors[1],
                label=f'MDD (n={len(mdd_logits)} wins)', edgecolor='white', linewidth=0.3)

    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Logit', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Window logit distribution by class', fontsize=11)
    ax.legend(fontsize=9)

    plt.suptitle(f'Per-window analysis | {title_suffix}', fontsize=12)
    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description='Per-window importance analysis')
    p.add_argument('--tag', required=True, help='Experiment tag')
    p.add_argument('--seed', type=int, default=42, help='Seed')
    p.add_argument('--fold', type=int, default=1, help='Fold (ignored if --global)')
    p.add_argument('--global_', action='store_true', dest='global_',
                   help='Aggregate across all folds')
    p.add_argument('--save', action='store_true', help='Save figure')
    args = p.parse_args()
    if args.save:
        matplotlib.use('Agg')

    print('Loading data...')
    (eeg_data, eeg_labels, eeg_cods), (aud_data, aud_labels, aud_cods), mapping = \
        load_eeg_cache(), load_audio_cache(), load_mapping()
    pairs, eeg_subjs, aud_subjs = build_paired_subjects(
        eeg_data, eeg_labels, eeg_cods, aud_data, aud_labels, aud_cods, mapping)

    all_logits = []
    all_labels = []

    if args.global_:
        fold_subjs = _get_fold_subjects(args.tag, args.seed)
        for fi, test_subj_ids in fold_subjs:
            test_indices = [i for i, (eid, _, _) in enumerate(pairs) if eid in test_subj_ids]
            if not test_indices:
                continue
            sub_pairs = [pairs[i] for i in test_indices]
            labels = [p[2] for p in sub_pairs]
            print(f'  Fold {fi}: {len(test_indices)} subjects')

            ckpt = load_checkpoint(args.tag, args.seed, fi)
            eeg_model, aud_model, fusion_model = build_models(ckpt)

            logits, _ = _get_window_logits(eeg_model, aud_model, fusion_model,
                                           sub_pairs, eeg_subjs, aud_subjs)
            all_logits.extend(logits)
            all_labels.extend(labels)

        title_suffix = f'seed={args.seed} global ({len(all_logits)} subj)'
    else:
        fi = args.fold
        ckpt_dir = find_checkpoint_dir(args.tag, args.seed)
        with open(os.path.join(ckpt_dir, 'results.json')) as f:
            results = json.load(f)
        test_subj_ids = results['folds'][fi - 1]['test_subjects']

        test_indices = [i for i, (eid, _, _) in enumerate(pairs) if eid in test_subj_ids]
        sub_pairs = [pairs[i] for i in test_indices]
        labels = [p[2] for p in sub_pairs]
        print(f'  Fold {fi}: {len(test_indices)} subjects')

        ckpt = load_checkpoint(args.tag, args.seed, fi)
        eeg_model, aud_model, fusion_model = build_models(ckpt)

        all_logits, all_labels = _get_window_logits(eeg_model, aud_model, fusion_model,
                                                     sub_pairs, eeg_subjs, aud_subjs)
        title_suffix = f'seed={args.seed} fold={fi}'

    fig = _plot(all_logits, all_labels, title_suffix)

    if args.save:
        out_dir = os.path.join(FIGURES_ROOT, 'windows', f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        mode = 'global' if args.global_ else f'fold{args.fold}'
        path = os.path.join(out_dir, f'seed{args.seed}_{mode}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

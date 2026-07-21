"""
Per-window prediction importance analysis: box plots + class density.

Usage:
  python -m src.interpretability.analyze_windows --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1
  python -m src.interpretability.analyze_windows --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1 --save
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from interpretability.base import (
    load_eeg_cache, load_audio_cache, load_mapping,
    build_paired_subjects, build_models, load_checkpoint,
    find_checkpoint_dir, encode_eeg, encode_audio,
    device, FIGURES_ROOT, parse_shared_args
)


def main():
    args = parse_shared_args('Per-window importance analysis')
    if args.save:
        matplotlib.use('Agg')

    # Load data
    print('Loading data...')
    (eeg_data, eeg_labels, eeg_cods), (aud_data, aud_labels, aud_cods), mapping = \
        load_eeg_cache(), load_audio_cache(), load_mapping()
    pairs, eeg_subjs, aud_subjs = build_paired_subjects(
        eeg_data, eeg_labels, eeg_cods, aud_data, aud_labels, aud_cods, mapping)

    print(f'Loading checkpoint seed={args.seed} fold={args.fold}...')
    ckpt = load_checkpoint(args.tag, args.seed, args.fold)
    eeg_model, aud_model, fusion_model = build_models(ckpt)

    ckpt_dir = find_checkpoint_dir(args.tag, args.seed)
    with open(os.path.join(ckpt_dir, 'results.json')) as f:
        results = json.load(f)
    fold_data = results['folds'][args.fold - 1]
    test_subj_ids = fold_data['test_subjects']

    test_indices = [i for i, (eid, _, _) in enumerate(pairs) if eid in test_subj_ids]
    sub_pairs = [pairs[i] for i in test_indices]
    print(f'  Test subjects: {len(test_indices)}')

    # Per-window logits for each test subject
    all_logits = []
    all_labels = []
    subj_names = []

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
        subj_names.append(eid)

    # ── Figure: boxplots + density ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ['#3498db', '#e74c3c']

    # Panel 1: Box plot per subject
    ax = axes[0]
    n_subj = len(subj_names)
    positions = np.arange(n_subj)
    bp = ax.boxplot(all_logits, positions=positions, widths=0.5, patch_artist=True,
                     showfliers=False)

    for patch, label in zip(bp['boxes'], all_labels):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.7)

    # Color the background by class
    for i in range(n_subj):
        ax.axvspan(i - 0.5, i + 0.5, facecolor=colors[all_labels[i]], alpha=0.08)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'S{i+1}' for i in range(n_subj)], fontsize=8)
    ax.set_xlabel('Subject', fontsize=10)
    ax.set_ylabel('Window logit', fontsize=10)
    ax.set_title('Per-subject window logit distribution', fontsize=11)

    # Add legend patches for class colors
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

    plt.suptitle(f'Per-window analysis | seed={args.seed} fold={args.fold}', fontsize=12)
    plt.tight_layout()

    if args.save:
        out_dir = os.path.join(FIGURES_ROOT, 'windows', f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'seed{args.seed}_fold{args.fold}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

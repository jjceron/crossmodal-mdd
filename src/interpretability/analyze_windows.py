"""
Per-window prediction importance analysis.

Uses forward_per_window() to get logits per window, identifies which
windows drive the final subject-level prediction most.

Usage:
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

    # Load checkpoint
    print(f'Loading checkpoint seed={args.seed} fold={args.fold}...')
    ckpt = load_checkpoint(args.tag, args.seed, args.fold)
    eeg_model, aud_model, fusion_model = build_models(ckpt)

    # Find test subjects
    ckpt_dir = find_checkpoint_dir(args.tag, args.seed)
    with open(os.path.join(ckpt_dir, 'results.json')) as f:
        results = json.load(f)
    fold_data = results['folds'][args.fold - 1]
    test_subj_ids = fold_data['test_subjects']

    test_indices = [i for i, (eid, _, _) in enumerate(pairs) if eid in test_subj_ids]
    sub_pairs = [pairs[i] for i in test_indices]
    print(f'  Test subjects: {len(test_indices)}')

    # For each test subject, get per-window logits
    all_win_logits = []
    n_windows_list = []

    for eid, aid, label in sub_pairs:
        we = eeg_subjs[eid]['windows'].copy()
        wa = aud_subjs[aid]['windows'].copy()

        K = min(len(we), len(wa))
        we, wa = we[:K], wa[:K]

        # Z-score per subject
        we = (we - we.mean()) / (we.std() + 1e-8)
        wa = (wa - wa.mean()) / (wa.std() + 1e-8)

        # Encode to features
        ze = encode_eeg(eeg_model, we)
        za = encode_audio(aud_model, wa)

        # Add batch dim and convert to tensors
        t_e = torch.FloatTensor(ze).unsqueeze(0).to(device)
        t_a = torch.FloatTensor(za).unsqueeze(0).to(device)

        with torch.no_grad():
            win_logits = fusion_model.forward_per_window(t_e, t_a)  # [1, K]
        all_win_logits.append(win_logits.squeeze(0).cpu().numpy())
        n_windows_list.append(K)

    # Plot: per-subject window importance distributions
    n_subj = len(sub_pairs)
    fig, axes = plt.subplots(1, n_subj, figsize=(5 * n_subj, 4))
    if n_subj == 1:
        axes = [axes]

    for idx in range(n_subj):
        eid, aid, label = sub_pairs[idx]
        scores = all_win_logits[idx]
        K = n_windows_list[idx]

        ax = axes[idx]
        ax.bar(range(K), scores, color='#3498db' if label == 0 else '#e74c3c', alpha=0.7)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_title(f'{eid} ({"MDD" if label else "HC"})', fontsize=10)
        ax.set_xlabel('Window index')
        ax.set_ylabel('Logit')
        ax.set_xlim(-0.5, K - 0.5)

        # Top-3 most MDD-like windows
        top_mdd = np.argsort(scores)[-3:][::-1]
        top_hc = np.argsort(scores)[:3]
        ax.scatter(top_mdd, scores[top_mdd], color='darkred', s=30, zorder=5, label='Top MDD')
        ax.scatter(top_hc, scores[top_hc], color='darkblue', s=30, zorder=5, label='Top HC')

        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle(f'Per-window logits | seed={args.seed} fold={args.fold}', fontsize=14)
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

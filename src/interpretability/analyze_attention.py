"""
Visualize cross-attention weights per subject.

Usage:
  python -m src.interpretability.analyze_attention --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1 --save
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
    find_checkpoint_dir, extract_all_features,
    device, FIGURES_ROOT, parse_shared_args
)


def main():
    args = parse_shared_args('Cross-attention weight visualization')
    if args.save:
        matplotlib.use('Agg')

    # Load data
    print('Loading data...')
    (eeg_data, eeg_labels, eeg_cods), (aud_data, aud_labels, aud_cods), mapping = \
        load_eeg_cache(), load_audio_cache(), load_mapping()
    pairs, eeg_subjs, aud_subjs = build_paired_subjects(
        eeg_data, eeg_labels, eeg_cods, aud_data, aud_labels, aud_cods, mapping)
    print(f'  Paired subjects: {len(pairs)}')

    # Load checkpoint
    print(f'Loading checkpoint seed={args.seed} fold={args.fold}...')
    ckpt = load_checkpoint(args.tag, args.seed, args.fold)
    eeg_model, aud_model, fusion_model = build_models(ckpt)

    # Load results.json to find test subjects
    ckpt_dir = find_checkpoint_dir(args.tag, args.seed)
    with open(os.path.join(ckpt_dir, 'results.json')) as f:
        results = json.load(f)
    fold_data = results['folds'][args.fold - 1]
    test_subj_ids = fold_data['test_subjects']
    print(f'  Test subjects: {test_subj_ids}')

    # Get test subject indices in pairs list
    test_indices = [i for i, (eid, _, _) in enumerate(pairs) if eid in test_subj_ids]
    if not test_indices:
        print('  No test subjects found — using all paired subjects')
        test_indices = list(range(len(pairs)))

    # Extract features for selected subjects
    print('Extracting features...')
    sub_pairs = [pairs[i] for i in test_indices]
    Z_e, Z_a, masks = extract_all_features(eeg_model, aud_model, sub_pairs, eeg_subjs, aud_subjs)

    # Forward pass to get attention weights
    print('Running forward pass...')
    t_e = torch.FloatTensor(Z_e).to(device)
    t_a = torch.FloatTensor(Z_a).to(device)
    t_m = torch.FloatTensor(masks).to(device)

    with torch.no_grad():
        fusion_model(t_e, t_a, t_m)
        e_attn, a_attn = fusion_model._attn_weights  # [B, n_heads, K, K]

    B, n_heads, K, _ = e_attn.shape

    # Average over heads and over key dim -> per-window importance
    e_imp = e_attn.mean(dim=1)  # [B, K, K] — each row is how much EEG attends to each Audio window
    a_imp = a_attn.mean(dim=1)  # [B, K, K] — each row is how much Audio attends to each EEG window

    # For each subject, show both matrices
    n_subj = len(test_indices)
    fig, axes = plt.subplots(n_subj, 2, figsize=(10, 4 * n_subj))
    if n_subj == 1:
        axes = axes[np.newaxis, :]

    for idx in range(n_subj):
        eeg_id, aud_id, label = sub_pairs[idx]
        actual_K = int(masks[idx].sum())

        # EEG→Audio attention: average over source windows to get per-target importance
        ax = axes[idx, 0]
        im = ax.imshow(e_imp[idx, :actual_K, :actual_K].cpu().numpy(), aspect='auto', cmap='viridis')
        ax.set_title(f'{eeg_id} (MDD' if label == 1 else f'{eeg_id} (HC')
        ax.set_xlabel('Audio window (attends to)')
        ax.set_ylabel('EEG window (query)')
        plt.colorbar(im, ax=ax)

        # Audio→EEG attention
        ax = axes[idx, 1]
        im = ax.imshow(a_imp[idx, :actual_K, :actual_K].cpu().numpy(), aspect='auto', cmap='viridis')
        ax.set_title(f'{aud_id} (MDD' if label == 1 else f'{aud_id} (HC')
        ax.set_xlabel('EEG window (attends to)')
        ax.set_ylabel('Audio window (query)')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Cross-attention weights | seed={args.seed} fold={args.fold}', fontsize=14)
    plt.tight_layout()

    if args.save:
        out_dir = os.path.join(FIGURES_ROOT, 'attention', f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'seed{args.seed}_fold{args.fold}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

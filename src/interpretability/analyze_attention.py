"""
Cross-attention weight analysis: aggregate matrices + per-subject asymmetry.

Usage:
  python -m src.interpretability.analyze_attention --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1
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
        load_eeg_cache(suffix=args.cache_suffix), load_audio_cache(), load_mapping()
    pairs, eeg_subjs, aud_subjs = build_paired_subjects(
        eeg_data, eeg_labels, eeg_cods, aud_data, aud_labels, aud_cods, mapping)

    ckpt = load_checkpoint(args.tag, args.seed, args.fold)
    eeg_model, aud_model, fusion_model = build_models(ckpt)

    ckpt_dir = find_checkpoint_dir(args.tag, args.seed)
    with open(os.path.join(ckpt_dir, 'results.json')) as f:
        results = json.load(f)
    fold_data = results['folds'][args.fold - 1]
    test_subj_ids = fold_data['test_subjects']

    test_indices = [i for i, (eid, _, _) in enumerate(pairs) if eid in test_subj_ids]
    sub_pairs = [pairs[i] for i in test_indices]
    test_labels = np.array([p[2] for p in sub_pairs])

    print(f'  Test subjects: {len(test_indices)}')

    Z_e, Z_a, masks = extract_all_features(eeg_model, aud_model, sub_pairs, eeg_subjs, aud_subjs)

    print('Running forward pass...')
    t_e = torch.FloatTensor(Z_e).to(device)
    t_a = torch.FloatTensor(Z_a).to(device)
    t_m = torch.FloatTensor(masks).to(device)

    with torch.no_grad():
        fusion_model(t_e, t_a, t_m)
        e_attn, a_attn = fusion_model._attn_weights

    B, n_heads, K, _ = e_attn.shape

    # Average over heads, keep mask
    e_attn_mean = e_attn.mean(dim=1).cpu().numpy()
    a_attn_mean = a_attn.mean(dim=1).cpu().numpy()
    masks_np = masks

    # ── FIGURE 1: Aggregate attention matrices (average across subjects) ──
    # Compute subject-level average over valid windows only
    e_mats, a_mats = [], []
    for idx in range(B):
        actual_K = int(masks_np[idx].sum())
        if actual_K < 2:
            continue
        e_mats.append(e_attn_mean[idx, :actual_K, :actual_K])
        a_mats.append(a_attn_mean[idx, :actual_K, :actual_K])

    if e_mats:
        # Resize all to same K for averaging (use min K across subjects)
        min_K = min(m.shape[0] for m in e_mats)
        e_stack = np.stack([m[:min_K, :min_K] for m in e_mats])
        a_stack = np.stack([m[:min_K, :min_K] for m in a_mats])
        e_avg = e_stack.mean(axis=0)
        a_avg = a_stack.mean(axis=0)
        asym = e_avg - a_avg
        asym_val = float(asym.mean())
    else:
        e_avg = a_avg = asym = np.zeros((2, 2))
        asym_val = 0.0

    fig1, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    im0 = axes[0].imshow(e_avg, aspect='auto', cmap='Blues', vmin=0, vmax=e_avg.max() if e_avg.max() > 0 else 1)
    axes[0].set_title('EEG attends to Audio (mean)', fontsize=11)
    axes[0].set_xlabel('Audio window', fontsize=9)
    axes[0].set_ylabel('EEG window', fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(a_avg, aspect='auto', cmap='Oranges', vmin=0, vmax=a_avg.max() if a_avg.max() > 0 else 1)
    axes[1].set_title('Audio attends to EEG (mean)', fontsize=11)
    axes[1].set_xlabel('EEG window', fontsize=9)
    axes[1].set_ylabel('Audio window', fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig1.suptitle(f'Aggregate attention | seed={args.seed} fold={args.fold}  (asymmetry={asym_val:+.3f})',
                  fontsize=12)
    plt.tight_layout()

    # ── FIGURE 2: Per-subject attention asymmetry ──
    fig2, ax = plt.subplots(figsize=(max(4, B * 0.6), 4))

    x = np.arange(B)
    w = 0.35

    eeg_att_by_subj = []
    aud_att_by_subj = []
    for idx in range(B):
        actual_K = int(masks_np[idx].sum())
        if actual_K < 2:
            eeg_att_by_subj.append(0.5)
            aud_att_by_subj.append(0.5)
        else:
            eeg_att_by_subj.append(float(e_attn_mean[idx, :actual_K, :actual_K].mean()))
            aud_att_by_subj.append(float(a_attn_mean[idx, :actual_K, :actual_K].mean()))

    colors_bg = ['#ffd8d8' if lbl == 1 else '#d8e8ff' for lbl in test_labels]
    for i in range(B):
        ax.axvspan(i - 0.5, i + 0.5, facecolor=colors_bg[i], alpha=0.4)

    ax.bar(x - w / 2, eeg_att_by_subj, w, label='EEG→Audio', color='#3498db', alpha=0.85)
    ax.bar(x + w / 2, aud_att_by_subj, w, label='Audio→EEG', color='#e67e22', alpha=0.85)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(B)], fontsize=8)
    ax.set_ylabel('Mean attention weight', fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(f'Attention asymmetry per subject | seed={args.seed} fold={args.fold}', fontsize=11)

    plt.tight_layout()

    if args.save:
        out_dir = os.path.join(FIGURES_ROOT, 'attention', f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        path1 = os.path.join(out_dir, f'seed{args.seed}_fold{args.fold}_aggregate.png')
        path2 = os.path.join(out_dir, f'seed{args.seed}_fold{args.fold}_asymmetry.png')
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f'Saved: {path1}')
        print(f'Saved: {path2}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

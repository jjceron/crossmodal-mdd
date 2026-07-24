"""
t-SNE visualization of backbone features (EEG, Audio, Fusion).

Usage:
  python -m src.interpretability.analyze_features --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1 --save
"""
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from interpretability.base import (
    load_eeg_cache, load_audio_cache, load_mapping,
    build_paired_subjects, build_models, load_checkpoint,
    extract_all_features,
    FIGURES_ROOT, parse_shared_args
)


def compute_silhouette(feats_2d, labels):
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(silhouette_score(feats_2d, labels))


def main():
    args = parse_shared_args('t-SNE feature visualization')
    if args.save:
        matplotlib.use('Agg')

    # Load data
    print('Loading data...')
    (eeg_data, eeg_labels, eeg_cods), (aud_data, aud_labels, aud_cods), mapping = \
        load_eeg_cache(suffix=args.cache_suffix), load_audio_cache(), load_mapping()
    pairs, eeg_subjs, aud_subjs = build_paired_subjects(
        eeg_data, eeg_labels, eeg_cods, aud_data, aud_labels, aud_cods, mapping)
    print(f'  Paired subjects: {len(pairs)}')

    # Load ensemble checkpoints
    print(f'Loading ensemble seed={args.seed} fold={args.fold}...')
    inner_ckpts = load_inner_checkpoints(args.tag, args.seed, args.fold)
    eeg_models, aud_models, fusion_models = build_ensemble_models(inner_ckpts)

    # Extract features for all paired subjects (average across ensemble)
    print('Extracting features (ensemble of %d)...' % len(inner_ckpts))
    all_feats_eeg, all_feats_aud = [], []
    for em, am in zip(eeg_models, aud_models):
        Z_e, Z_a, masks = extract_all_features(em, am, pairs, eeg_subjs, aud_subjs)
        masks_3d = masks[..., np.newaxis]
        feats_eeg = (Z_e * masks_3d).sum(axis=1) / masks.sum(axis=1, keepdims=True).clip(min=1)
        feats_aud = (Z_a * masks_3d).sum(axis=1) / masks.sum(axis=1, keepdims=True).clip(min=1)
        all_feats_eeg.append(feats_eeg)
        all_feats_aud.append(feats_aud)

    feats_eeg = np.mean(all_feats_eeg, axis=0)
    feats_aud = np.mean(all_feats_aud, axis=0)
    feats_cat = np.concatenate([feats_eeg, feats_aud], axis=1)

    labels = np.array([p[2] for p in pairs])

    # t-SNE on each modality
    print('Running t-SNE...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(pairs) - 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    titles = ['EEG backbone features', 'Audio backbone features', 'EEG + Audio']
    feats_list = [feats_eeg, feats_aud, feats_cat]
    colors = ['#e74c3c', '#3498db']
    markers = ['o', 's']

    for ax, title, feats in zip(axes, titles, feats_list):
        feats_2d = tsne.fit_transform(feats)
        sil = compute_silhouette(feats_2d, labels)

        for cls in [0, 1]:
            mask = labels == cls
            ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                       c=colors[cls], marker=markers[cls],
                       label=f'{"MDD" if cls == 1 else "HC"} (n={mask.sum()})',
                       alpha=0.8, edgecolors='black', linewidth=0.3, s=60)
        ax.set_title(f'{title}\nSilhouette: {sil:.3f}', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f't-SNE | seed={args.seed} fold={args.fold}', fontsize=14)
    plt.tight_layout()

    if args.save:
        out_dir = os.path.join(FIGURES_ROOT, 'features', f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'seed{args.seed}_fold{args.fold}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

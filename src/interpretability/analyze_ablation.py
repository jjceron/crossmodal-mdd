"""
Ablation: compare fusion vs EEG-only vs Audio-only on test subjects.

Usage:
  python -m src.interpretability.analyze_ablation --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1
  python -m src.interpretability.analyze_ablation --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --global
  python -m src.interpretability.analyze_ablation --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1 --save
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score


@torch.no_grad()
def evaluate(fusion_model, Z_e, Z_a, masks, labels):
    t_e = torch.FloatTensor(Z_e).to(device)
    t_a = torch.FloatTensor(Z_a).to(device)
    t_m = torch.FloatTensor(masks).to(device)
    logits = fusion_model(t_e, t_a, t_m).cpu().numpy()
    preds = (logits > 0).astype(int)
    bacc = balanced_accuracy_score(labels, preds)
    auc = roc_auc_score(labels, logits) if len(np.unique(labels)) > 1 else 0.5
    f1 = f1_score(labels, preds, zero_division=0)
    return bacc, auc, f1


def _run_ablation(tag, seed, fi, pairs, eeg_subjs, aud_subjs, test_subj_ids):
    test_indices = [i for i, (eid, _, _) in enumerate(pairs) if eid in test_subj_ids]
    sub_pairs = [pairs[i] for i in test_indices]
    test_labels = np.array([p[2] for p in sub_pairs])

    ckpt = load_checkpoint(tag, seed, fi)
    eeg_model, aud_model, fusion_model = build_models(ckpt)

    Z_e, Z_a, masks = extract_all_features(eeg_model, aud_model, sub_pairs, eeg_subjs, aud_subjs)

    bacc_f, auc_f, f1_f = evaluate(fusion_model, Z_e, Z_a, masks, test_labels)
    bacc_e, auc_e, f1_e = evaluate(fusion_model, Z_e, np.zeros_like(Z_a), masks, test_labels)
    bacc_a, auc_a, f1_a = evaluate(fusion_model, np.zeros_like(Z_e), Z_a, masks, test_labels)

    return {
        'fold': fi,
        'fusion': (bacc_f, auc_f, f1_f),
        'eeg': (bacc_e, auc_e, f1_e),
        'audio': (bacc_a, auc_a, f1_a),
    }


def main():
    p = argparse.ArgumentParser(description='Modality ablation study')
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

    ckpt_dir = find_checkpoint_dir(args.tag, args.seed)
    with open(os.path.join(ckpt_dir, 'results.json')) as f:
        results = json.load(f)

    if args.global_:
        all_results = []
        for fd in results['folds']:
            fi = fd['fold']
            print(f'  Fold {fi}...')
            r = _run_ablation(args.tag, args.seed, fi, pairs, eeg_subjs, aud_subjs,
                              fd['test_subjects'])
            all_results.append(r)

        # Aggregate with mean ± std
        metrics = ['BACC', 'AUC', 'F1']
        fusion_vals = np.array([r['fusion'] for r in all_results])
        eeg_vals = np.array([r['eeg'] for r in all_results])
        aud_vals = np.array([r['audio'] for r in all_results])

        fusion_mean = fusion_vals.mean(axis=0)
        fusion_std = fusion_vals.std(axis=0)
        eeg_mean = eeg_vals.mean(axis=0)
        eeg_std = eeg_vals.std(axis=0)
        aud_mean = aud_vals.mean(axis=0)
        aud_std = aud_vals.std(axis=0)

        print(f'\n  Global ablation ({len(all_results)} folds):')
        print(f'    Fusion:  BACC={fusion_mean[0]:.3f}±{fusion_std[0]:.3f}  AUC={fusion_mean[1]:.3f}±{fusion_std[1]:.3f}  F1={fusion_mean[2]:.3f}±{fusion_std[2]:.3f}')
        print(f'    EEG:     BACC={eeg_mean[0]:.3f}±{eeg_std[0]:.3f}  AUC={eeg_mean[1]:.3f}±{eeg_std[1]:.3f}  F1={eeg_mean[2]:.3f}±{eeg_std[2]:.3f}')
        print(f'    Audio:   BACC={aud_mean[0]:.3f}±{aud_std[0]:.3f}  AUC={aud_mean[1]:.3f}±{aud_std[1]:.3f}  F1={aud_mean[2]:.3f}±{aud_std[2]:.3f}')

        # Plot with error bars
        x = np.arange(len(metrics))
        w = 0.25

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w, eeg_mean, w, yerr=eeg_std, capsize=3,
               label='EEG only', color='#3498db', alpha=0.85)
        ax.bar(x, aud_mean, w, yerr=aud_std, capsize=3,
               label='Audio only', color='#e67e22', alpha=0.85)
        ax.bar(x + w, fusion_mean, w, yerr=fusion_std, capsize=3,
               label='Fusion', color='#2ecc71', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title(f'Modality Ablation (global, {len(all_results)} folds) | seed={args.seed}')
        ax.legend(fontsize=10)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        plt.tight_layout()
    else:
        fi = args.fold
        test_subj_ids = results['folds'][fi - 1]['test_subjects']
        r = _run_ablation(args.tag, args.seed, fi, pairs, eeg_subjs, aud_subjs, test_subj_ids)

        print(f'  Fusion:  BACC={r["fusion"][0]:.3f}  AUC={r["fusion"][1]:.3f}  F1={r["fusion"][2]:.3f}')
        print(f'  EEG:     BACC={r["eeg"][0]:.3f}  AUC={r["eeg"][1]:.3f}  F1={r["eeg"][2]:.3f}')
        print(f'  Audio:   BACC={r["audio"][0]:.3f}  AUC={r["audio"][1]:.3f}  F1={r["audio"][2]:.3f}')

        metrics = ['BACC', 'AUC', 'F1']
        fusion_vals = list(r['fusion'])
        eeg_vals = list(r['eeg'])
        aud_vals = list(r['audio'])

        x = np.arange(len(metrics))
        w = 0.25

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w, eeg_vals, w, label='EEG only', color='#3498db', alpha=0.85)
        ax.bar(x, aud_vals, w, label='Audio only', color='#e67e22', alpha=0.85)
        ax.bar(x + w, fusion_vals, w, label='Fusion', color='#2ecc71', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title(f'Modality Ablation | seed={args.seed} fold={fi}')
        ax.legend(fontsize=10)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

        for i, (f, e, a) in enumerate(zip(fusion_vals, eeg_vals, aud_vals)):
            ax.text(i - w, e + 0.02, f'{e:.2f}', ha='center', va='bottom', fontsize=8, color='#3498db')
            ax.text(i, a + 0.02, f'{a:.2f}', ha='center', va='bottom', fontsize=8, color='#e67e22')
            ax.text(i + w, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8, color='#2ecc71')

        plt.tight_layout()

    if args.save:
        subdir = 'ablation_global' if args.global_ else 'ablation'
        out_dir = os.path.join(FIGURES_ROOT, subdir, f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        mode = 'global' if args.global_ else f'fold{args.fold}'
        path = os.path.join(out_dir, f'seed{args.seed}_{mode}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

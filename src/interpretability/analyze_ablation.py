"""
Ablation: compare fusion vs EEG-only vs Audio-only on test subjects.

Usage:
  python -m src.interpretability.analyze_ablation --tag bbvalfix_d07_lr5e4_6seeds --seed 42 --fold 1 --save
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from interpretability.base import (
    load_eeg_cache, load_audio_cache, load_mapping,
    build_paired_subjects, build_models, load_checkpoint,
    find_checkpoint_dir, extract_all_features,
    device, RESULTS_ROOT, FIGURES_ROOT, parse_shared_args
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


def main():
    args = parse_shared_args('Modality ablation study')

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
    test_labels = np.array([p[2] for p in sub_pairs])

    print(f'  Test subjects: {len(test_indices)}')

    # Extract features
    print('Extracting features...')
    Z_e, Z_a, masks = extract_all_features(eeg_model, aud_model, sub_pairs, eeg_subjs, aud_subjs)

    # Evaluate all modalities
    print('Evaluating...')

    # 1) Fusion
    bacc_f, auc_f, f1_f = evaluate(fusion_model, Z_e, Z_a, masks, test_labels)
    print(f'  Fusion:  bacc={bacc_f:.3f}  auc={auc_f:.3f}  f1={f1_f:.3f}')

    # 2) EEG only — zero out audio features
    Z_a_zero = np.zeros_like(Z_a)
    bacc_e, auc_e, f1_e = evaluate(fusion_model, Z_e, Z_a_zero, masks, test_labels)
    print(f'  EEG only: bacc={bacc_e:.3f}  auc={auc_e:.3f}  f1={f1_e:.3f}')

    # 3) Audio only — zero out EEG features
    Z_e_zero = np.zeros_like(Z_e)
    bacc_a, auc_a, f1_a = evaluate(fusion_model, Z_e_zero, Z_a, masks, test_labels)
    print(f'  Audio only: bacc={bacc_a:.3f}  auc={auc_a:.3f}  f1={f1_a:.3f}')

    # Plot
    metrics = ['BACC', 'AUC', 'F1']
    fusion_vals = [bacc_f, auc_f, f1_f]
    eeg_vals = [bacc_e, auc_e, f1_e]
    aud_vals = [bacc_a, auc_a, f1_a]

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
    ax.set_title(f'Modality Ablation | seed={args.seed} fold={args.fold}')
    ax.legend(fontsize=10)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # Annotate bars
    for i, (f, e, a) in enumerate(zip(fusion_vals, eeg_vals, aud_vals)):
        ax.text(i - w, e + 0.02, f'{e:.2f}', ha='center', va='bottom', fontsize=8, color='#3498db')
        ax.text(i, a + 0.02, f'{a:.2f}', ha='center', va='bottom', fontsize=8, color='#e67e22')
        ax.text(i + w, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8, color='#2ecc71')

    plt.tight_layout()

    if args.save:
        out_dir = os.path.join(FIGURES_ROOT, 'ablation', f'{args.tag}')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'seed{args.seed}_fold{args.fold}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

"""
Nested cross-validation: backbones trained from scratch per outer fold.
Outer LOOCV or K-fold on CAMPNet pairs — no Fase 1 initialization leakage.

Usage:
  py src/training/run_crossmodal_nested.py --fusion cross_attn --n-self-attn-layers 1
  py src/training/run_crossmodal_nested.py --fusion concat --outer-folds 5
"""
import sys, os, json, argparse, copy, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.training.run_crossmodal_e2e import (
    EEGBackbone, AudioBackbone, E2EModel, WindowClassifier,
    SubjectWindowDataset, e2e_collate, evaluate,
    _load_cache, _load_multimodal_pairs,
    N_EEG_CH, N_AUDIO_MELS, N_AUDIO_FRAMES, RANDOM_STATE,
)
from src.utils.training_logger import ClassificationLogger

EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/crossmodal/nested_loocv'
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def _loocv_split(groups, labels, n_folds):
    indices = np.arange(len(groups))
    if n_folds == 0:
        for group in sorted(set(groups.tolist())):
            mask = groups == group
            tei = indices[mask]
            tvi = indices[~mask]
            yield tvi, tei
    else:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        yield from skf.split(np.zeros(len(groups)), labels, groups=groups)


def main():
    parser = argparse.ArgumentParser(description='Nested LOOCV crossmodal training')
    parser.add_argument('--fusion', choices=['concat', 'gating', 'cross_attn'],
                        default='cross_attn')
    parser.add_argument('--n-self-attn-layers', type=int, default=1)
    parser.add_argument('--self-attn-heads', type=int, default=4)
    parser.add_argument('--self-attn-dropout', type=float, default=0.1)
    parser.add_argument('--bottleneck-dim', type=int, default=None)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=2)
    parser.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max-windows', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (both backbones and fusion)')
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--bs', type=int, default=4, help='Subjects per batch')
    parser.add_argument('--window-aux', action='store_true', default=True,
                        help='Auxiliary window-level classifier (multi-task)')
    parser.add_argument('--window-aux-weight', type=float, default=0.3)
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Mixup alpha (0 = disable)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save best model checkpoint per fold')
    parser.add_argument('--param-control', action='store_true',
                        help='Add MLP with matching param count when n_self_attn=0')
    parser.add_argument('--outer-folds', type=int, default=0,
                        help='0=LOOCV, 1..N=K-fold (default 0=LOOCV, 38 folds)')
    args = parser.parse_args()

    is_loocv = args.outer_folds == 0
    outer_label = 'loocv' if is_loocv else f'k{args.outer_folds}'
    cfg_name = f'{args.fusion}'
    if args.n_self_attn_layers > 0:
        cfg_name += f'_self{args.n_self_attn_layers}L'
    if args.bottleneck_dim is not None:
        cfg_name += f'_bn{args.bottleneck_dim}'
    if args.param_control:
        cfg_name += '_paramctrl'
    cfg_name += f'_nested_{outer_label}_w{args.max_windows}'
    out_dir = os.path.join(OUTPUT_DIR, cfg_name)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'Nested CrossModalAttention — {cfg_name}')
    print(f'  LR={args.lr}  Window aux={args.window_aux} (weight={args.window_aux_weight})')
    print(f'  Mixup alpha={args.mixup_alpha}  Epochs={args.epochs}  Patience={args.patience}')
    n_folds_info = 'LOOCV' if is_loocv else f'{args.outer_folds}-fold'
    print(f'  Outer: {n_folds_info}')

    # Load data
    eeg_subjs, _, _ = _load_cache(EEG_CACHE)
    aud_subjs, _, _ = _load_cache(AUDIO_CACHE)
    pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
    labels = np.array([p[2] for p in pairs])
    group_ids = np.array([f'p{i}' for i in range(len(pairs))])
    print(f'  Multimodal pairs: {len(pairs)} ({int(labels.sum())} MDD, '
          f'{len(pairs) - int(labels.sum())} HC)')

    fold_results = []
    for fi, (tvi, tei) in enumerate(_loocv_split(group_ids, labels, args.outer_folds)):
        n_folds_total = len(pairs) if is_loocv else args.outer_folds
        print(f'\n─── Fold {fi + 1}/{n_folds_total} ───')
        try:
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                         random_state=RANDOM_STATE + fi)
            tr_i, vl_i = next(inner.split(np.zeros(len(tvi)),
                                          labels[tvi], groups=group_ids[tvi]))
            tr_idx, vl_idx = tvi[tr_i], tvi[vl_i]

            tr_ds = SubjectWindowDataset(eeg_subjs, aud_subjs, pairs, tr_idx, args.max_windows)
            vl_ds = SubjectWindowDataset(eeg_subjs, aud_subjs, pairs, vl_idx, args.max_windows)
            tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                                   collate_fn=e2e_collate, num_workers=0)
            vl_loader = DataLoader(vl_ds, batch_size=args.bs, shuffle=False,
                                   collate_fn=e2e_collate, num_workers=0)

            model = E2EModel(args).to(device)
            # NO Fase 1 initialization — training from scratch

            total_p = sum(p.numel() for p in model.parameters())
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if fi == 0:
                print(f'    Model params: {total_p:,} total, {trainable_p:,} trainable')
                print(f'    Train subjects: {len(tr_idx)}  Val: {len(vl_idx)}  Test: {len(tei)}')

            # Override LR in optimizer (same LR for all params when training from scratch)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='max', factor=0.5, patience=5)
            crit = nn.BCEWithLogitsLoss()

            best_vb, best_st, pat = 0.0, None, 0
            logger = ClassificationLogger()
            logger.log_header()

            for ep in range(1, args.epochs + 1):
                model.train()
                tr_loss, tr_n = 0.0, 0
                tr_logits, tr_labels = [], []

                for X_e, X_a, mask, yb in tr_loader:
                    X_e, X_a = X_e.to(device), X_a.to(device)
                    mask, yb = mask.to(device), yb.to(device)
                    opt.zero_grad()
                    logits, win_logits = model(X_e, X_a, mask, return_window=args.window_aux)
                    y_smooth = yb * 0.95 + 0.025
                    loss = crit(logits, y_smooth)
                    if win_logits is not None:
                        B, K = X_e.shape[0], X_e.shape[1]
                        y_win = yb.unsqueeze(1).expand(-1, K).reshape(-1)
                        mask_flat = mask.reshape(-1)
                        win_loss = crit(win_logits, y_win * 0.95 + 0.025)
                        win_loss = (win_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                        loss = loss + args.window_aux_weight * win_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tr_loss += loss.item() * yb.size(0)
                    tr_n += yb.size(0)
                    tr_logits.append(logits.detach())
                    tr_labels.append(yb)

                tr_loss /= tr_n
                tr_pred = (torch.sigmoid(torch.cat(tr_logits)).cpu().numpy() >= 0.5).astype(int)
                tr_true = torch.cat(tr_labels).cpu().numpy()
                tr_m = logger.metrics(tr_true, tr_pred)

                model.eval()
                vl_logits, vl_labels = [], []
                with torch.no_grad():
                    for X_e, X_a, mask, yb in vl_loader:
                        logits, _ = model(X_e.to(device), X_a.to(device), mask.to(device))
                        vl_logits.append(logits.cpu())
                        vl_labels.append(yb)
                vl_logits = torch.cat(vl_logits)
                vl_labels = torch.cat(vl_labels)
                vl_loss = crit(vl_logits, vl_labels).item()
                vl_pred = (torch.sigmoid(vl_logits).numpy() >= 0.5).astype(int)
                vl_m = logger.metrics(vl_labels.numpy(), vl_pred)
                sched.step(vl_m['bacc'])

                if vl_m['bacc'] > best_vb:
                    best_vb = vl_m['bacc']
                    best_st = copy.deepcopy(model.state_dict())
                    pat = 0
                else:
                    pat += 1
                if ep == 1 or pat == 0 or ep % 10 == 0:
                    logger.log_epoch(ep, tr_loss, vl_loss, tr_m, vl_m, pat)
                if pat >= args.patience:
                    break

            model.load_state_dict(best_st)
            print(f'    Best val bacc = {best_vb:.4f}')

            if args.save_model:
                ckpt_path = os.path.join(ckpt_dir, f'fold_{fi+1}.pt')
                torch.save({
                    'fold': fi + 1,
                    'best_val_bacc': best_vb,
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                }, ckpt_path)
                print(f'    Saved: {ckpt_path}')

            te_ds = SubjectWindowDataset(eeg_subjs, aud_subjs, pairs, tei, args.max_windows)
            te_loader = DataLoader(te_ds, batch_size=len(tei), shuffle=False,
                                   collate_fn=e2e_collate, num_workers=0)
            X_e_te, X_a_te, mask_te, y_te = next(iter(te_loader))
            y_true_s, y_pred_s, y_prob_s = evaluate(model, X_e_te, X_a_te, mask_te, y_te)

            cm = confusion_matrix(y_true_s, y_pred_s).tolist()
            roc_auc = float(roc_auc_score(y_true_s, y_prob_s))
            bacc = balanced_accuracy_score(y_true_s, y_pred_s)
            logger = ClassificationLogger()
            fm = logger.log_fold_test(y_true_s, y_pred_s)

            fold_results.append({
                'fold': fi + 1,
                'best_val_bacc': float(best_vb),
                'test_metrics': fm,
                'test_bacc': float(bacc),
                'test_auc': roc_auc,
                'test_cm': cm,
                'test_roc': {'y_true': y_true_s.tolist(), 'y_prob': y_prob_s.tolist()},
            })
            print(f'  Fold {fi + 1}: bacc={bacc:.3f} AUC={roc_auc:.3f}')

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  Fold {fi + 1} FAILED: {e}')
            import traceback; traceback.print_exc()

    if fold_results:
        baccs = [r['test_bacc'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        summary = {
            'bacc_mean': float(np.mean(baccs)), 'bacc_std': float(np.std(baccs)),
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
        }
        print(f'\n{"=" * 55}')
        print(f'  {cfg_name}')
        print(f'  bacc = {summary["bacc_mean"]:.3f} ± {summary["bacc_std"]:.3f}')
        print(f'  auc  = {summary["auc_mean"]:.3f} ± {summary["auc_std"]:.3f}')
        print(f'{"=" * 55}')

        out_results = {
            'config_name': cfg_name,
            'probe_type': 'nested_loocv',
            'args': vars(args),
            'folds': fold_results,
            'summary': summary,
        }
        out_path = os.path.join(out_dir, 'results.json')
        with open(out_path, 'w') as f:
            json.dump(out_results, f, indent=2)
        print(f'Saved: {out_path}')

        csv_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'consolidated_results.csv')
        header = 'config_name,probe_type,fusion,n_self_attn,bottleneck_dim,' \
                 f'max_windows,bacc_mean,bacc_std,auc_mean,auc_std\n'
        row = f'{cfg_name},nested_loocv,{args.fusion},{args.n_self_attn_layers},' \
              f'{args.bottleneck_dim},{args.max_windows},' \
              f'{summary["bacc_mean"]:.4f},{summary["bacc_std"]:.4f},' \
              f'{summary["auc_mean"]:.4f},{summary["auc_std"]:.4f}\n'
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(header)
                f.write(row)
        else:
            with open(csv_path, 'a') as f:
                f.write(row)
        print(f'Appended to: {csv_path}')


if __name__ == '__main__':
    main()

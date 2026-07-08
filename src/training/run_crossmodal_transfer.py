"""
Clean transfer learning: backbones trained from scratch per outer fold
EXCLUDING CAMPNet test subjects — zero leakage.

For each outer fold (5-fold):
  1. Hold out ~8 CAMPNet test subjects
  2. Train EEG backbone on ~45 of 53 EEG subjects (excluding test)
  3. Train Audio backbone on ~44 of 52 audio subjects (excluding test)
  4. Fine-tune fusion on ~30 CAMPNet train pairs
  5. Test on ~8 held-out subjects

Usage:
  py src/training/run_crossmodal_transfer.py --fusion cross_attn --n-self-attn-layers 1
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

from src.training.dl_eeg_benchmark import load_cached_eeg, train_eeg_backbone
from src.training.dl_audio_benchmark import load_cached_audio, train_audio_backbone
from src.training.run_crossmodal_e2e import (
    E2EModel, SubjectWindowDataset, e2e_collate, evaluate,
    _load_multimodal_pairs, N_EEG_CH, N_AUDIO_MELS, N_AUDIO_FRAMES, RANDOM_STATE,
)
from src.utils.training_logger import ClassificationLogger

AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/crossmodal/transfer_clean'
N_OUTER = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def main():
    parser = argparse.ArgumentParser(description='Clean transfer crossmodal training')
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
    parser.add_argument('--bb-epochs', type=int, default=100,
                        help='Epochs for backbone training (Stage 1)')
    parser.add_argument('--bb-lr', type=float, default=5e-4,
                        help='Learning rate for backbone training')
    parser.add_argument('--bb-patience', type=int, default=20,
                        help='Patience for backbone training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs for fusion fine-tuning (Stage 2)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for fusion fine-tuning')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--bs', type=int, default=32,
                        help='Batch size for backbone training')
    parser.add_argument('--bs-ft', type=int, default=4,
                        help='Subjects per batch for fusion fine-tuning')
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--window-aux', action='store_true', default=True)
    parser.add_argument('--window-aux-weight', type=float, default=0.3)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--param-control', action='store_true')
    args = parser.parse_args()

    cfg_name = f'{args.fusion}'
    if args.n_self_attn_layers > 0:
        cfg_name += f'_self{args.n_self_attn_layers}L'
    if args.bottleneck_dim is not None:
        cfg_name += f'_bn{args.bottleneck_dim}'
    if args.param_control:
        cfg_name += '_paramctrl'
    cfg_name += f'_transfer_w{args.max_windows}'
    out_dir = os.path.join(OUTPUT_DIR, cfg_name)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'Clean Transfer CrossModalAttention - {cfg_name}')
    print(f'  Backbone: bb_epochs={args.bb_epochs} bb_lr={args.bb_lr}')
    print(f'  Fusion:   epochs={args.epochs} lr={args.lr}')
    print(f'  Window aux={args.window_aux} weight={args.window_aux_weight}')
    print(f'  Mixup alpha={args.mixup_alpha}')

    # 1. Load all unimodal data
    print('\n--- Loading EEG data ---')
    eeg_data, eeg_labels, eeg_ids, n_samples = load_cached_eeg(64)
    print(f'\n--- Loading Audio data ---')
    aud_data, aud_labels, aud_ids = load_cached_audio()
    print()

    # 2. Load CAMPNet pairs
    eeg_subjs = {sid: {'windows': eeg_data[i], 'label': int(eeg_labels[i])}
                 for i, sid in enumerate(eeg_ids)}
    aud_subjs = {sid: {'windows': aud_data[i], 'label': int(aud_labels[i])}
                 for i, sid in enumerate(aud_ids)}
    pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
    pair_labels = np.array([p[2] for p in pairs])
    pair_groups = np.array([f'p{i}' for i in range(len(pairs))])

    print(f'  CAMPNet pairs: {len(pairs)} ({int(pair_labels.sum())} MDD, '
          f'{len(pairs) - int(pair_labels.sum())} HC)')

    # 3. Backbone training args
    bb_args = argparse.Namespace(
        lr=args.bb_lr, wd=args.wd, epochs=args.bb_epochs, patience=args.bb_patience,
        bs=args.bs, max_windows=args.max_windows, bottleneck_dim=args.bottleneck_dim,
    )

    skf = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(pairs)), pair_labels, groups=pair_groups)):
        print(f'\n{"=" * 60}')
        print(f'  OUTER FOLD {fi + 1}/{N_OUTER}')
        print(f'  Train pairs: {len(tvi)}  Test pairs: {len(tei)}')
        print(f'{"=" * 60}')

        try:
            # ── Identify test subject IDs (to exclude from backbone pre-training) ──
            test_eeg_ids = set()
            test_aud_ids = set()
            for pidx in tei:
                eid, aid, _ = pairs[pidx]
                test_eeg_ids.add(eid)
                test_aud_ids.add(aid)

            # ── EEG backbone: exclude test subjects, train on rest ──
            eeg_train_mask = np.array([sid not in test_eeg_ids for sid in eeg_ids])
            eeg_bb_idx = np.where(eeg_train_mask)[0]
            eeg_n_train = len(eeg_bb_idx)
            print(f'  EEG backbone: {eeg_n_train} subjects (removed {len(test_eeg_ids)} test)')

            inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                          random_state=RANDOM_STATE + fi)
            eeg_tr_i, eeg_vl_i = next(inner.split(
                np.zeros(eeg_n_train), eeg_labels[eeg_bb_idx], groups=eeg_bb_idx))
            eeg_tr_idx = eeg_bb_idx[eeg_tr_i]
            eeg_vl_idx = eeg_bb_idx[eeg_vl_i]
            print(f'    Inner: {len(eeg_tr_idx)} train, {len(eeg_vl_idx)} val')

            eeg_model, eeg_vb = train_eeg_backbone(
                eeg_tr_idx, eeg_vl_idx, eeg_data, eeg_labels, eeg_ids,
                N_EEG_CH, n_samples, bb_args, model_key='deepconvnet')
            print(f'    EEG backbone val bacc = {eeg_vb:.4f}')

            # ── Audio backbone: exclude test subjects, train on rest ──
            aud_train_mask = np.array([sid not in test_aud_ids for sid in aud_ids])
            aud_bb_idx = np.where(aud_train_mask)[0]
            aud_n_train = len(aud_bb_idx)
            print(f'  Audio backbone: {aud_n_train} subjects (removed {len(test_aud_ids)} test)')

            inner2 = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                           random_state=RANDOM_STATE + fi + 10)
            aud_tr_i, aud_vl_i = next(inner2.split(
                np.zeros(aud_n_train), aud_labels[aud_bb_idx], groups=aud_bb_idx))
            aud_tr_idx = aud_bb_idx[aud_tr_i]
            aud_vl_idx = aud_bb_idx[aud_vl_i]
            print(f'    Inner: {len(aud_tr_idx)} train, {len(aud_vl_idx)} val')

            aud_model, aud_vb = train_audio_backbone(
                aud_tr_idx, aud_vl_idx, aud_data, aud_labels, aud_ids,
                bb_args, model_key='shallowconvnet')
            print(f'    Audio backbone val bacc = {aud_vb:.4f}')

            # ── Build E2EModel and load trained backbone weights ──
            model = E2EModel(args).to(device)
            eeg_sd = eeg_model.state_dict()
            aud_sd = aud_model.state_dict()

            eeg_own = model.eeg_bb.state_dict()
            eeg_map = {k: v for k, v in eeg_sd.items()
                       if k in eeg_own and 'classifier' not in k}
            eeg_own.update(eeg_map)
            model.eeg_bb.load_state_dict(eeg_own)

            aud_own = model.aud_bb.state_dict()
            aud_map = {k: v for k, v in aud_sd.items()
                       if k in aud_own and 'classifier' not in k}
            aud_own.update(aud_map)
            model.aud_bb.load_state_dict(aud_own)

            total_p = sum(p.numel() for p in model.parameters())
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if fi == 0:
                print(f'\n  Model params: {total_p:,} total, {trainable_p:,} trainable')

            # ── Fine-tune on CAMPNet train pairs ──
            inner_camp = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                               random_state=RANDOM_STATE + fi + 20)
            c_tr_i, c_vl_i = next(inner_camp.split(
                np.zeros(len(tvi)), pair_labels[tvi], groups=pair_groups[tvi]))
            ft_tr_idx = tvi[c_tr_i]
            ft_vl_idx = tvi[c_vl_i]
            print(f'  Fusion fine-tune: {len(ft_tr_idx)} train, {len(ft_vl_idx)} val, '
                  f'{len(tei)} test')

            ft_tr_ds = SubjectWindowDataset(eeg_subjs, aud_subjs, pairs, ft_tr_idx,
                                             args.max_windows)
            ft_vl_ds = SubjectWindowDataset(eeg_subjs, aud_subjs, pairs, ft_vl_idx,
                                             args.max_windows)
            ft_tr_loader = DataLoader(ft_tr_ds, batch_size=args.bs_ft, shuffle=True,
                                       collate_fn=e2e_collate, num_workers=0)
            ft_vl_loader = DataLoader(ft_vl_ds, batch_size=args.bs_ft, shuffle=False,
                                       collate_fn=e2e_collate, num_workers=0)

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
                for X_e, X_a, mask, yb in ft_tr_loader:
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
                    for X_e, X_a, mask, yb in ft_vl_loader:
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
                    'backbone_eeg_val_bacc': eeg_vb,
                    'backbone_audio_val_bacc': aud_vb,
                }, ckpt_path)
                print(f'    Saved: {ckpt_path}')

            te_ds = SubjectWindowDataset(eeg_subjs, aud_subjs, pairs, tei,
                                          args.max_windows)
            te_loader = DataLoader(te_ds, batch_size=len(tei), shuffle=False,
                                    collate_fn=e2e_collate, num_workers=0)
            X_e_te, X_a_te, mask_te, y_te = next(iter(te_loader))
            y_true_s, y_pred_s, y_prob_s = evaluate(model, X_e_te, X_a_te, mask_te, y_te)
            cm = confusion_matrix(y_true_s, y_pred_s).tolist()
            roc_auc = float(roc_auc_score(y_true_s, y_prob_s))
            bacc = balanced_accuracy_score(y_true_s, y_pred_s)
            lf = ClassificationLogger()
            fm = lf.log_fold_test(y_true_s, y_pred_s)
            fold_results.append({
                'fold': fi + 1,
                'best_val_bacc': float(best_vb),
                'test_metrics': fm,
                'test_bacc': float(bacc),
                'test_auc': roc_auc,
                'test_cm': cm,
                'test_roc': {'y_true': y_true_s.tolist(), 'y_prob': y_prob_s.tolist()},
                'backbone_eeg_val_bacc': float(eeg_vb),
                'backbone_audio_val_bacc': float(aud_vb),
            })
            print(f'  Fold {fi + 1}: bacc={bacc:.3f} AUC={roc_auc:.3f}')
            del model, eeg_model, aud_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  Fold {fi + 1} FAILED: {e}')
            import traceback; traceback.print_exc()

    if fold_results:
        baccs = [r['test_bacc'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        eeg_bbs = [r['backbone_eeg_val_bacc'] for r in fold_results]
        aud_bbs = [r['backbone_audio_val_bacc'] for r in fold_results]
        summary = {
            'bacc_mean': float(np.mean(baccs)), 'bacc_std': float(np.std(baccs)),
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            'backbone_eeg_val_bacc_mean': float(np.mean(eeg_bbs)),
            'backbone_audio_val_bacc_mean': float(np.mean(aud_bbs)),
        }
        print(f'\n{"=" * 55}')
        print(f'  {cfg_name}')
        print(f'  bacc = {summary["bacc_mean"]:.3f} +/- {summary["bacc_std"]:.3f}')
        print(f'  auc  = {summary["auc_mean"]:.3f} +/- {summary["auc_std"]:.3f}')
        print(f'  EEG backbone val bacc (mean) = {summary["backbone_eeg_val_bacc_mean"]:.4f}')
        print(f'  Audio backbone val bacc (mean) = {summary["backbone_audio_val_bacc_mean"]:.4f}')
        print(f'{"=" * 55}')

        out_results = {
            'config_name': cfg_name,
            'probe_type': 'transfer_clean',
            'args': vars(args),
            'folds': fold_results,
            'summary': summary,
        }
        out_path = os.path.join(out_dir, 'results.json')
        with open(out_path, 'w') as f:
            json.dump(out_results, f, indent=2)
        print(f'Saved: {out_path}')

        csv_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'consolidated_results.csv')
        header = ('config_name,probe_type,fusion,n_self_attn,bottleneck_dim,'
                  'max_windows,bacc_mean,bacc_std,auc_mean,auc_std,'
                  'backbone_eeg_val_bacc,backbone_audio_val_bacc\n')
        row = (f'{cfg_name},transfer_clean,{args.fusion},{args.n_self_attn_layers},'
               f'{args.bottleneck_dim},{args.max_windows},'
               f'{summary["bacc_mean"]:.4f},{summary["bacc_std"]:.4f},'
               f'{summary["auc_mean"]:.4f},{summary["auc_std"]:.4f},'
               f'{summary["backbone_eeg_val_bacc_mean"]:.4f},{summary["backbone_audio_val_bacc_mean"]:.4f}\n')
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

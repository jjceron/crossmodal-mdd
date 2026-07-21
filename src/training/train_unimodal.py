"""
Strict nested unimodal benchmark (SNGKF protocol).

Trains a single backbone (EEG or Audio) under the same SNGKF protocol as
train_crossmodal_sngkf.py:
  1. For each outer fold's training subjects, perform inner CV:
     - Re-train backbone excluding inner_val subjects
     - Evaluate on inner_val
  2. Average best_epoch across inner folds
  3. Train final backbone on ALL outer training subjects
  4. Evaluate on outer test subjects

Usage:
  py -m src.training.train_unimodal --model deepconvnet --cache-suffix ftsm8 --tag eeg_ftsm8 --seed 42 1825 410
  py -m src.training.train_unimodal --model shallowconvnet --tag aud_baseline --seed 42 1825 410
"""
import sys
import os
import json
import argparse
import copy
import warnings
import platform
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from datetime import datetime
import subprocess

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4 if platform.system() != 'Windows' else 0
sys.path.insert(0, '.')

from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.utils.training_logger import ClassificationLogger
from src.utils.get_seed import set_seed, parse_seeds

# ── Constants ──

EEG_CACHE_TPL = 'data/processed/eeg_preprocessed_{}.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/unimodal'
RANDOM_STATE = 42

# ── Model wrappers ──

class DeepConvNetWrapper(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()
        self.m = DeepConvNet(n_channels, 1, n_samples, 0.5)
    def forward(self, x):
        return self.m(x).squeeze(-1)

class ShallowConvNetWrapper(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()
        self.m = ShallowConvNet(n_channels, 1, n_samples, 0.5)
    def forward(self, x):
        return self.m(x).squeeze(-1)

# ── Data loading ──

def load_eeg(cache_suffix):
    path = EEG_CACHE_TPL.format(cache_suffix)
    c = np.load(path, allow_pickle=True)
    data = list(c['windows'])
    labels = c['labels'].astype(int)
    cods = list(c['subject_ids'])
    n_samples = data[0].shape[2]
    n_ch = data[0].shape[1]
    print(f'  EEG ({cache_suffix}): {len(cods)} subj ({int(labels.sum())} MDD, {int((1-labels).sum())} HC), '
          f'windows: {n_ch}ch x {n_samples}')
    return data, labels, cods, n_samples, n_ch

def load_audio():
    c = np.load(AUDIO_CACHE, allow_pickle=True)
    data = list(c['windows'])
    labels = c['labels'].astype(int)
    cods = [str(s) for s in c['subject_ids']]
    n_samples = data[0].shape[2]
    n_ch = data[0].shape[1]
    print(f'  Audio: {len(cods)} subj ({int(labels.sum())} MDD, {int((1-labels).sum())} HC), '
          f'windows: {n_ch}ch x {n_samples}')
    return data, labels, cods, n_samples, n_ch

def select_windows_deterministic(windows, max_windows):
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    indices = np.linspace(0, n - 1, max_windows, dtype=int)
    return windows[indices]

class WindowDataset(Dataset):
    def __init__(self, windows_list, labels_list, subj_names, indices, max_windows=None, seed=None):
        self._windows = windows_list
        self._subj_names = subj_names
        self._labels = labels_list
        self._index = []
        for idx in indices:
            wins = windows_list[idx]
            subj_mean = wins.mean()
            subj_std = wins.std() + 1e-8
            n = wins.shape[0]
            if max_windows is not None and n > max_windows:
                if seed is not None:
                    rng = np.random.RandomState(seed + idx)
                    keep = rng.choice(n, max_windows, replace=False)
                else:
                    keep = np.linspace(0, n - 1, max_windows, dtype=int)
                for k in keep:
                    self._index.append((idx, int(k), float(labels_list[idx]), subj_mean, subj_std))
            else:
                for w in range(n):
                    self._index.append((idx, w, float(labels_list[idx]), subj_mean, subj_std))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        idx, w_idx, label, mu, sig = self._index[i]
        w = self._windows[idx][w_idx].copy()
        w = (w - mu) / sig
        return torch.from_numpy(w).float(), torch.tensor(label, dtype=torch.float), self._subj_names[idx]

# ── Training ──

def train_backbone(model, train_loader, val_loader, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss()
    logger = ClassificationLogger()
    logger.log_header()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        tr_logits, tr_labels = [], []
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device).float()
            opt.zero_grad()
            logits = model(X)
            y_smooth = y * 0.95 + 0.025
            loss = crit(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * X.size(0)
            tr_n += X.size(0)
            tr_logits.append(logits.detach())
            tr_labels.append(y)
        tr_loss /= tr_n
        tr_pred = (torch.sigmoid(torch.cat(tr_logits)).cpu().numpy() >= 0.5).astype(int)
        tr_true = torch.cat(tr_labels).cpu().numpy()
        tr_m = logger.metrics(tr_true, tr_pred)

        model.eval()
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device).float()
                logits = model(X)
                vl_logits.append(logits.cpu())
                vl_labels.append(y.cpu())
        vl_loss = crit(torch.cat(vl_logits).to(device), torch.cat(vl_labels).to(device)).item()
        vl_pred = (torch.sigmoid(torch.cat(vl_logits)).numpy() >= 0.5).astype(int)
        vl_m = logger.metrics(torch.cat(vl_labels).numpy(), vl_pred)
        sched.step()

        if ep == 1 or ep % 10 == 0:
            logger.log_epoch(ep, tr_loss, vl_loss, tr_m, vl_m, 0)

    final_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    final_vl = float(vl_loss)
    return final_st, final_vl, args.epochs

# ── Evaluation ──

def evaluate(model, loader):
    model.eval()
    all_logits, all_labels, all_subjs = [], [], []
    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device)
            logits = model(X)
            all_logits.append(logits.cpu())
            all_labels.append(y)
            all_subjs.extend(s)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    return labels, preds, probs.squeeze(), all_subjs

def subject_majority_vote(subjs, probs, labels):
    unique = sorted(set(subjs))
    y_true, y_pred, y_prob = [], [], []
    for s in unique:
        mask = [i for i, ss in enumerate(subjs) if ss == s]
        s_probs = np.array([probs[i] for i in mask])
        s_labels = labels[mask[0]]
        vote = int((s_probs >= 0.5).mean() >= 0.5)
        y_true.append(s_labels)
        y_pred.append(vote)
        y_prob.append(float(s_probs.mean()))
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Strict nested unimodal benchmark (SNGKF)')
    parser.add_argument('--model', choices=['deepconvnet', 'shallowconvnet'], required=True)
    parser.add_argument('--cache-suffix', type=str, default='64ch', help='EEG cache suffix (ignored for audio)')
    parser.add_argument('--max-windows', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=5)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--seed', type=int, nargs='+', default=[42])
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()

    is_eeg = args.model == 'deepconvnet'

    for seed in parse_seeds(args.seed):
        run_seed(seed, args, is_eeg)

def run_seed(seed, args, is_eeg):
    global RANDOM_STATE
    RANDOM_STATE = seed
    set_seed(RANDOM_STATE)

    git_commit = ''
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_commit = 'unknown'

    modal = 'eeg' if is_eeg else 'aud'
    cache_label = args.cache_suffix if is_eeg else '64mel'
    tag = args.tag or f'{args.model}_{cache_label}'
    cfg_name = f'unimodal_sngkf_{modal}_{args.model}_{cache_label}_seed{seed}'
    tag_suffix = f'_tag{tag}' if args.tag and args.tag != tag else ''
    cfg_name += tag_suffix
    out_dir = os.path.join(OUTPUT_DIR, cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Seed={seed}  Model={args.model}  Cache={cache_label}')
    print(f'  {cfg_name}')
    print(f'{"="*60}')
    print(f'Device: {device}')

    # Load data
    if is_eeg:
        data, labels, cods, n_samples, n_ch = load_eeg(args.cache_suffix)
    else:
        data, labels, cods, n_samples, n_ch = load_audio()

    print(f'  Outer CV: {args.outer_folds}-fold  Inner: {args.inner_folds}-fold')

    # Build subject dict
    subj_dict = {cods[i]: {'windows': data[i], 'label': int(labels[i])} for i in range(len(cods))}
    unique_ids = list(subj_dict.keys())
    unique_labels = np.array([subj_dict[sid]['label'] for sid in unique_ids])

    splitter = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fi, (tvi, tei) in enumerate(splitter.split(np.zeros(len(unique_ids)), unique_labels, groups=unique_ids)):
        print(f'\n{"="*50}')
        print(f'  Outer Fold {fi + 1}')
        print(f'{"="*50}')

        tr_ids = [unique_ids[i] for i in tvi]
        te_ids = [unique_ids[i] for i in tei]
        y_tr = np.array([subj_dict[sid]['label'] for sid in tr_ids])
        y_te = np.array([subj_dict[sid]['label'] for sid in te_ids])

        print(f'  Train: {len(tr_ids)} subj  Test: {len(te_ids)} subj')

        # Inner CV
        inner_splitter = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True, random_state=seed + fi)
        inner_best_eps = []

        for inner_fi, (inner_tr_i, inner_vl_i) in enumerate(
                inner_splitter.split(np.zeros(len(tr_ids)), y_tr, groups=tr_ids)):
            print(f'\n    --- Inner fold {inner_fi + 1}/{args.inner_folds} ---')

            inner_vl_ids = [tr_ids[i] for i in inner_vl_i]
            inner_tr_ids = [tr_ids[i] for i in inner_tr_i]

            # Build backbone training data (exclude inner_vl)
            bb_data = [subj_dict[sid]['windows'] for sid in inner_tr_ids]
            bb_labels = np.array([subj_dict[sid]['label'] for sid in inner_tr_ids])
            bb_cods = inner_tr_ids

            # Train backbone
            n_bb = len(bb_labels)
            bb_vl_size = 4
            sss = StratifiedShuffleSplit(n_splits=1, test_size=bb_vl_size, random_state=seed + fi * 10 + inner_fi + 999)
            bb_tr_i, bb_vl_i = next(sss.split(np.zeros(n_bb), bb_labels))
            # Reconstruct full data lists (indices into bb_data)
            tr_eeg_list = [bb_data[i] for i in bb_tr_i]
            tr_labels_list = bb_labels[bb_tr_i]
            tr_cods_list = [bb_cods[i] for i in bb_tr_i]
            vl_eeg_list = [bb_data[i] for i in bb_vl_i]
            vl_labels_list = bb_labels[bb_vl_i]
            vl_cods_list = [bb_cods[i] for i in bb_vl_i]

            train_ds = WindowDataset(tr_eeg_list, tr_labels_list, tr_cods_list,
                                     list(range(len(tr_eeg_list))),
                                     max_windows=args.max_windows, seed=seed + fi * 10 + inner_fi)
            val_ds = WindowDataset(vl_eeg_list, vl_labels_list, vl_cods_list,
                                   list(range(len(vl_eeg_list))),
                                   max_windows=args.max_windows, seed=seed + fi * 10 + inner_fi)
            train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, pin_memory=NUM_WORKERS > 0)
            val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=NUM_WORKERS > 0)

            if is_eeg:
                model = DeepConvNetWrapper(n_ch, n_samples).to(device)
            else:
                model = ShallowConvNetWrapper(n_ch, n_samples).to(device)

            if fi == 0 and inner_fi == 0:
                print(f'    Params: {sum(p.numel() for p in model.parameters()):,}')
            print('    Training backbone...')
            best_st, best_loss, _ = train_backbone(model, train_ldr, val_ldr, args)
            model.load_state_dict(best_st)
            model.eval()

            # Evaluate on inner_vl
            vl_data_list = [subj_dict[sid]['windows'] for sid in inner_vl_ids]
            vl_labels_list = np.array([subj_dict[sid]['label'] for sid in inner_vl_ids])
            vl_cods_list = inner_vl_ids
            vl_ds = WindowDataset(vl_data_list, vl_labels_list, vl_cods_list,
                                  list(range(len(vl_data_list))),
                                  max_windows=args.max_windows, seed=seed + fi * 10 + inner_fi)
            vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=NUM_WORKERS > 0)

            yt_vl, yp_vl, pr_vl, subjs_vl = evaluate(model, vl_ldr)
            # Subject-level majority vote
            yt_vl_s, yp_vl_s, _ = subject_majority_vote(subjs_vl, pr_vl, yt_vl)
            inner_bacc = balanced_accuracy_score(yt_vl_s, yp_vl_s)
            print(f'    >>> Inner fold {inner_fi + 1} val_bacc={inner_bacc:.3f}')

        # ── Train FINAL backbone on ALL training subjects ──
        print('\n    --- Training FINAL backbone on ALL training subjects ---')
        bb_data = [subj_dict[sid]['windows'] for sid in tr_ids]
        bb_labels = np.array([subj_dict[sid]['label'] for sid in tr_ids])
        bb_cods = tr_ids

        n_bb = len(bb_labels)
        bb_vl_size = 4
        sss = StratifiedShuffleSplit(n_splits=1, test_size=bb_vl_size, random_state=seed + fi + 999)
        bb_tr_i, bb_vl_i = next(sss.split(np.zeros(n_bb), bb_labels))
        tr_eeg_list = [bb_data[i] for i in bb_tr_i]
        tr_labels_list = bb_labels[bb_tr_i]
        tr_cods_list = [bb_cods[i] for i in bb_tr_i]
        vl_eeg_list = [bb_data[i] for i in bb_vl_i]
        vl_labels_list = bb_labels[bb_vl_i]
        vl_cods_list = [bb_cods[i] for i in bb_vl_i]

        train_ds = WindowDataset(tr_eeg_list, tr_labels_list, tr_cods_list,
                                 list(range(len(tr_eeg_list))),
                                 max_windows=args.max_windows, seed=seed + fi)
        val_ds = WindowDataset(vl_eeg_list, vl_labels_list, vl_cods_list,
                               list(range(len(vl_eeg_list))),
                               max_windows=args.max_windows, seed=seed + fi)
        train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, pin_memory=NUM_WORKERS > 0)
        val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=NUM_WORKERS > 0)

        model = DeepConvNetWrapper(n_ch, n_samples).to(device) if is_eeg else ShallowConvNetWrapper(n_ch, n_samples).to(device)
        best_st, best_loss, _ = train_backbone(model, train_ldr, val_ldr, args)
        model.load_state_dict(best_st)
        model.eval()

        # ── Evaluate on test subjects ──
        te_data_list = [subj_dict[sid]['windows'] for sid in te_ids]
        te_labels_arr = np.array([subj_dict[sid]['label'] for sid in te_ids])
        te_cods_list = te_ids
        te_ds = WindowDataset(te_data_list, te_labels_arr, te_cods_list,
                              list(range(len(te_data_list))),
                              max_windows=args.max_windows, seed=seed + fi)
        te_ldr = DataLoader(te_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=NUM_WORKERS > 0)

        y_true, y_pred, y_prob, subjs_test = evaluate(model, te_ldr)
        y_true_s, y_pred_s, y_prob_s = subject_majority_vote(subjs_test, y_prob, y_true)

        bacc = balanced_accuracy_score(y_true_s, y_pred_s)
        roc_auc = float(roc_auc_score(y_true_s, y_prob_s if len(np.unique(y_true_s)) > 1 else y_true_s))
        cm = confusion_matrix(y_true_s, y_pred_s).tolist()

        logger = ClassificationLogger()
        fm = logger.log_fold_test(y_true_s, y_pred_s)
        print(f'\n  >>> Fold {fi + 1}: test BACC={bacc:.3f}  AUC={roc_auc:.3f}')

        fold_results.append({
            'fold': fi + 1,
            'test_metrics': fm,
            'test_bacc': float(bacc),
            'test_auc': roc_auc,
            'test_cm': cm,
            'n_train': len(tr_ids),
            'n_test': len(te_ids),
            'test_subjects': te_ids,
            'model': args.model,
            'modal': modal,
            'cache_suffix': cache_label,
            'eeg_backbone_val_loss': float(best_loss),
        })

        # Save checkpoint
        if args.save_model:
            ckpt_dir = os.path.join(out_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'fold': fi + 1,
                'model_state_dict': best_st,
                'args': vars(args),
                'test_bacc': float(bacc),
                'test_auc': roc_auc,
            }, os.path.join(ckpt_dir, f'fold_{fi+1}.pt'))

        # Partial save
        partial_baccs = [r['test_bacc'] for r in fold_results]
        partial_aucs = [r['test_auc'] for r in fold_results]
        partial_out = {
            'experiment': {
                'name': 'unimodal_sngkf', 'model': args.model, 'modal': modal,
                'cache_suffix': cache_label, 'tag': args.tag,
                'timestamp': datetime.now().isoformat(), 'git_commit': git_commit, 'seed': seed,
            },
            'config': {
                'model': args.model, 'modal': modal, 'cache_suffix': cache_label,
                'lr': args.lr, 'wd': args.wd, 'epochs': args.epochs,
                'max_windows': args.max_windows,
                'outer_folds': args.outer_folds, 'inner_folds': args.inner_folds,
            },
            'test': {
                'bacc_mean': float(np.mean(partial_baccs)),
                'bacc_std': float(np.std(partial_baccs)),
                'auc_mean': float(np.mean(partial_aucs)),
                'auc_std': float(np.std(partial_aucs)),
            },
            'folds': fold_results,
            'partial': True,
        }
        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(partial_out, f, indent=2)

    # ── Summary ──
    if fold_results:
        baccs = [r['test_bacc'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        test = {
            'bacc_mean': float(np.mean(baccs)),
            'bacc_std': float(np.std(baccs)),
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
        }
        summary = {
            'bacc_mean': float(np.mean(baccs)),
            'bacc_std': float(np.std(baccs)),
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
        }
        print(f'\n{"="*50}')
        print(f'  {cfg_name}')
        print(f'  BACC = {summary["bacc_mean"]:.3f} ± {summary["bacc_std"]:.3f}')
        print(f'  AUC  = {summary["auc_mean"]:.3f} ± {summary["auc_std"]:.3f}')
        print(f'{"="*50}')

        out_results = {
            'experiment': {
                'name': 'unimodal_sngkf', 'model': args.model, 'modal': modal,
                'cache_suffix': cache_label, 'tag': args.tag,
                'timestamp': datetime.now().isoformat(), 'git_commit': git_commit, 'seed': seed,
            },
            'config': {
                'model': args.model, 'modal': modal, 'cache_suffix': cache_label,
                'lr': args.lr, 'wd': args.wd, 'epochs': args.epochs,
                'max_windows': args.max_windows,
                'outer_folds': args.outer_folds, 'inner_folds': args.inner_folds,
            },
            'test': test,
            'folds': fold_results,
            'summary': summary,
        }
        out_path = os.path.join(out_dir, 'results.json')
        with open(out_path, 'w') as f:
            json.dump(out_results, f, indent=2)
        print(f'Saved: {out_path}')
    else:
        print('\nAll folds failed — no results saved')

if __name__ == '__main__':
    main()

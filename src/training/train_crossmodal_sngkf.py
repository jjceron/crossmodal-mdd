"""
Strict nested cross-modal training — fixes fusion validation leakage.

Problem in original train_crossmodal_strict.py:
  Backbones are trained on ALL tr_paired subjects (~31), then fusion validation
  split comes from the same tr_paired — val subjects were already seen by backbone.

Fix:
  For each outer fold's tr_paired (~31):
    - Inner 3-fold CV over tr_paired only
    - For each inner fold:
      1. Exclude inner_val subjects from backbone training (re-train backbones
         on ~60-2=58 subjects instead of ~60)
      2. Extract features with those clean backbones
      3. Train fusion on inner_train, validate on inner_val
    - Average val_bacc and best_epoch across inner folds
    - Finally: train backbones on ALL tr_paired + unpaired (~60),
      train fusion for avg_best_epoch (from inner CV) epochs

Outer test subjects remain completely unseen throughout.

Usage:
  py src/training/train_crossmodal_strict_nested.py --fusion cross_attn --dropout 0.7 --max-windows 200
  py src/training/train_crossmodal_strict_nested.py --fusion cross_attn --dropout 0.7 --max-windows 200 --bottleneck-dim 32
"""
import sys
import os
import json
import argparse
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from datetime import datetime
import subprocess

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.models.crossmodal_attn import CrossModalAttention  # noqa: E402
from src.models.deepconvnet import DeepConvNet  # noqa: E402
from src.models.shallowconvnet import ShallowConvNet  # noqa: E402
from src.utils.training_logger import ClassificationLogger  # noqa: E402
from src.utils.get_seed import set_seed, parse_seeds  # noqa: E402

# ── Backbone wrappers ──

class DeepConvNetWrapper(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()
        self.m = DeepConvNet(n_channels, 1, n_samples, 0.7)
    def forward(self, x): return self.m(x).squeeze(-1)
    def forward_features(self, x): return self.m.forward_features(x)

class ShallowConvNetWrapper(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()
        self.m = ShallowConvNet(n_channels, 1, n_samples, 0.7)
    def forward(self, x): return self.m(x).squeeze(-1)
    def forward_features(self, x): return self.m.forward_features(x)

# ── Data / constants ──

EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/crossmodal_nested'
RANDOM_STATE = 42
N_FOLDS = 5
INNER_FUSION_FOLDS = 3
N_MELS = 64
N_AUDIO_SAMPLES = 200
N_EEG_SAMPLES = 500
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(RANDOM_STATE)

# ── Data loading (same as original) ──

def _load_eeg_cache(path=EEG_CACHE):
    c = np.load(path, allow_pickle=True)
    data = list(c['windows'])
    labels = c['labels'].astype(int)
    cods = list(c['subject_ids'])
    n_samples = data[0].shape[2]
    n_ch = data[0].shape[1]
    print(f'  EEG: {len(cods)} subj ({int(labels.sum())} MDD, {int((1-labels).sum())} HC), '
          f'windows: {n_ch}ch x {n_samples}')
    return data, labels, cods

def _load_audio_cache(path=AUDIO_CACHE):
    c = np.load(path, allow_pickle=True)
    data = list(c['windows'])
    labels = c['labels'].astype(int)
    cods = [str(s) for s in c['subject_ids']]
    print(f'  Audio: {len(cods)} subj ({int(labels.sum())} MDD, {int((1-labels).sum())} HC)')
    return data, labels, cods

def _load_mapping(path=MAPPING_PATH):
    with open(path) as f:
        return json.load(f)['orig_to_bids']

def _select_windows_deterministic(windows, max_windows):
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    indices = np.linspace(0, n - 1, max_windows, dtype=int)
    return windows[indices]

# ── Augmentation ──

class EEGAugment:
    def __init__(self, noise_std=0.05, time_mask_prob=0.3, time_mask_max=20,
                 channel_drop_prob=0.3, channel_drop_ratio=0.15):
        self.noise_std = noise_std
        self.time_mask_prob = time_mask_prob
        self.time_mask_max = time_mask_max
        self.channel_drop_prob = channel_drop_prob
        self.channel_drop_ratio = channel_drop_ratio

    def __call__(self, w):
        w = w.copy()
        if self.noise_std > 0:
            w += np.random.randn(*w.shape).astype(np.float32) * self.noise_std
        if self.time_mask_prob > 0 and np.random.random() < self.time_mask_prob:
            t = w.shape[1]
            mask_len = np.random.randint(5, self.time_mask_max + 1)
            start = np.random.randint(0, max(1, t - mask_len))
            w[:, start:start+mask_len] = 0.0
        if self.channel_drop_prob > 0 and np.random.random() < self.channel_drop_prob:
            n_ch = w.shape[0]
            n_drop = max(1, int(n_ch * self.channel_drop_ratio))
            drop_idx = np.random.choice(n_ch, n_drop, replace=False)
            w[drop_idx] = 0.0
        return w

class AudioAugment:
    def __init__(self, noise_std=0.05, time_mask_prob=0.3, time_mask_max=10,
                 freq_mask_prob=0.3, freq_mask_max=8):
        self.noise_std = noise_std
        self.time_mask_prob = time_mask_prob
        self.time_mask_max = time_mask_max
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_max = freq_mask_max

    def __call__(self, w):
        w = w.copy()
        if self.noise_std > 0:
            w += np.random.randn(*w.shape).astype(np.float32) * self.noise_std
        if self.time_mask_prob > 0 and np.random.random() < self.time_mask_prob:
            t = w.shape[1]
            mask_len = np.random.randint(3, self.time_mask_max + 1)
            start = np.random.randint(0, max(1, t - mask_len))
            w[:, start:start+mask_len] = 0.0
        if self.freq_mask_prob > 0 and np.random.random() < self.freq_mask_prob:
            f = w.shape[0]
            mask_len = np.random.randint(2, self.freq_mask_max + 1)
            start = np.random.randint(0, max(1, f - mask_len))
            w[start:start+mask_len, :] = 0.0
        return w

# ── Window dataset ──

class WindowDataset(Dataset):
    def __init__(self, windows_list, labels_list, subj_names, indices, max_windows=None, augmenter=None):
        self._windows = windows_list
        self._subj_names = subj_names
        self._labels = labels_list
        self._augmenter = augmenter
        self._index = []
        self._subj_mean = {}
        self._subj_std = {}
        for idx in indices:
            wins = windows_list[idx]
            self._subj_mean[idx] = wins.mean()
            self._subj_std[idx] = wins.std() + 1e-8
            n = wins.shape[0]
            if max_windows is not None and n > max_windows:
                keep = np.linspace(0, n - 1, max_windows, dtype=int)
                for k in keep:
                    self._index.append((idx, int(k), float(labels_list[idx])))
            else:
                for w in range(n):
                    self._index.append((idx, w, float(labels_list[idx])))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        idx, w_idx, label = self._index[i]
        w = self._windows[idx][w_idx].copy()
        w = (w - self._subj_mean[idx]) / self._subj_std[idx]
        if self._augmenter is not None:
            w = self._augmenter(w)
        return torch.from_numpy(w).float(), torch.tensor(label, dtype=torch.float), self._subj_names[idx]

# ── Backbone training ──

def _logits_to_binary(logits):
    return (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)

def _compute_epoch_metrics(model, loader, crit):
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device).float()
            logits = model(X)
            total_loss += crit(logits, y).item() * X.size(0)
            n += X.size(0)
            all_logits.append(logits)
            all_labels.append(y)
    loss = total_loss / n
    preds = _logits_to_binary(torch.cat(all_logits))
    trues = torch.cat(all_labels).cpu().numpy()
    return loss, ClassificationLogger().metrics(trues, preds)

def train_backbone(model, train_loader, val_loader, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=False)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    crit = nn.BCEWithLogitsLoss()
    best_vb, best_st, pat, best_ep = -1.0, None, 0, 0
    logger = ClassificationLogger()
    logger.log_header()
    history = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc',
                                'val_bacc', 'val_f1', 'val_sens', 'val_spec')}
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        tr_logits, tr_labels = [], []
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device).float()
            opt.zero_grad()
            logits = model(X)
            if torch.isnan(logits).any():
                raise RuntimeError('NaN in logits — training diverged')
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
        tr_pred = _logits_to_binary(torch.cat(tr_logits))
        tr_true = torch.cat(tr_labels).cpu().numpy()
        tr_m = logger.metrics(tr_true, tr_pred)
        vl_loss, vl_m = _compute_epoch_metrics(model, val_loader, crit)
        sched.step(vl_m['bacc'])

        history['train_loss'].append(float(tr_loss))
        history['val_loss'].append(float(vl_loss))
        history['train_acc'].append(tr_m['acc'])
        history['val_acc'].append(vl_m['acc'])
        for k in ('bacc', 'f1', 'sens', 'spec'):
            history[f'val_{k}'].append(vl_m[k])

        if vl_m['bacc'] > best_vb:
            best_vb = vl_m['bacc']
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            pat = 0
        else:
            pat += 1
        if ep == 1 or pat == 0 or ep % 10 == 0:
            logger.log_epoch(ep, tr_loss, vl_loss, tr_m, vl_m, pat)
        if pat >= args.patience:
            break
    if best_st is None:
        best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_st, best_vb, best_ep, history

# ── Feature extraction ──

def _encode_eeg(model, windows, device):
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(device)
        with torch.no_grad():
            feats.append(model.forward_features(batch).cpu())
    return torch.cat(feats, dim=0).numpy()

def _encode_audio(model, windows, device):
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(device)
        with torch.no_grad():
            feats.append(model.forward_features(batch).cpu())
    return torch.cat(feats, dim=0).numpy()

def extract_subject_features(eeg_model, aud_model, eeg_wins, aud_wins,
                              eeg_augment=None, aud_augment=None):
    we = eeg_wins.copy()
    wa = aud_wins.copy()
    K = min(len(we), len(wa))
    we, wa = we[:K], wa[:K]
    eeg_mu, eeg_sig = we.mean(), we.std() + 1e-8
    aud_mu, aud_sig = wa.mean(), wa.std() + 1e-8
    we = (we - eeg_mu) / eeg_sig
    wa = (wa - aud_mu) / aud_sig
    if eeg_augment is not None:
        we = np.array([eeg_augment(we[i]) for i in range(K)])
    if aud_augment is not None:
        wa = np.array([aud_augment(wa[i]) for i in range(K)])
    ze = _encode_eeg(eeg_model, we, device)
    za = _encode_audio(aud_model, wa, device)
    return ze, za

def extract_all_features(eeg_model, aud_model, subj_pairs, eeg_subjs, aud_subjs,
                          max_windows, eeg_augment=None, aud_augment=None):
    all_ze, all_za, all_masks = [], [], []
    for eid, aid, _ in subj_pairs:
        we = _select_windows_deterministic(eeg_subjs[eid]['windows'], max_windows)
        wa = _select_windows_deterministic(aud_subjs[aid]['windows'], max_windows)
        ze, za = extract_subject_features(eeg_model, aud_model, we, wa,
                                           eeg_augment, aud_augment)
        K = len(ze)
        all_ze.append(ze)
        all_za.append(za)
        all_masks.append(np.ones(K, dtype=np.float32))

    max_K = max(m.shape[0] for m in all_masks)
    eeg_dim = all_ze[0].shape[1]
    aud_dim = all_za[0].shape[1]
    Z_e = np.zeros((len(subj_pairs), max_K, eeg_dim), dtype=np.float32)
    Z_a = np.zeros((len(subj_pairs), max_K, aud_dim), dtype=np.float32)
    masks = np.zeros((len(subj_pairs), max_K), dtype=np.float32)
    for i in range(len(subj_pairs)):
        k = len(all_ze[i])
        Z_e[i, :k] = all_ze[i]
        Z_a[i, :k] = all_za[i]
        masks[i, :k] = all_masks[i]
    return Z_e, Z_a, masks

# ── Fusion head training ──

def train_fusion_head(model, Z_e_tr, Z_a_tr, mask_tr, y_tr,
                      Z_e_vl, Z_a_vl, mask_vl, y_vl, args):
    ds_tr = torch.utils.data.TensorDataset(
        torch.FloatTensor(Z_e_tr), torch.FloatTensor(Z_a_tr),
        torch.FloatTensor(mask_tr), torch.FloatTensor(y_tr))
    ds_vl = torch.utils.data.TensorDataset(
        torch.FloatTensor(Z_e_vl), torch.FloatTensor(Z_a_vl),
        torch.FloatTensor(mask_vl), torch.FloatTensor(y_vl))
    tr_ldr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True)
    vl_ldr = DataLoader(ds_vl, batch_size=args.bs, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_fusion,
                            weight_decay=args.wd_fusion, foreach=False)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    n_mdd = y_tr.sum()
    n_hc = len(y_tr) - n_mdd
    pos_weight = torch.tensor([n_hc / max(n_mdd, 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_vb, best_st, pat, best_ep = -1.0, None, 0, 0
    logger = ClassificationLogger()
    logger.log_header()
    history = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc',
                                'val_bacc', 'val_f1', 'val_sens', 'val_spec')}

    for ep in range(1, args.fusion_epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        tr_logits, tr_labels = [], []
        for ze, za, m, yb in tr_ldr:
            ze, za, m, yb = ze.to(device), za.to(device), m.to(device), yb.to(device)
            opt.zero_grad()
            tr_labels.append(yb)
            if args.mixup_alpha > 0:
                ze, za, yb, _ = mixup_features(ze, za, yb, args.mixup_alpha)
            return_window = args.window_aux and model.training
            out = model(ze, za, mask=m, return_window=return_window)
            if return_window:
                logits, win_logits = out
            else:
                logits = out
                win_logits = None
            y_smooth = yb * 0.95 + 0.025
            loss = crit(logits, y_smooth)
            if win_logits is not None:
                K = ze.shape[1]
                y_win = yb.unsqueeze(1).expand(-1, K).reshape(-1)
                mask_flat = m.reshape(-1)
                win_loss = crit(win_logits, y_win * 0.95 + 0.025)
                win_loss = (win_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                loss = loss + args.window_aux_weight * win_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * yb.size(0)
            tr_n += yb.size(0)
            tr_logits.append(logits.detach())
        tr_loss /= tr_n
        tr_pred = (torch.sigmoid(torch.cat(tr_logits)).cpu().numpy() >= 0.5).astype(int)
        tr_true = torch.cat(tr_labels).cpu().numpy()
        tr_m = logger.metrics(tr_true, tr_pred)

        model.eval()
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for ze, za, m, yb in vl_ldr:
                logits = model(ze.to(device), za.to(device), mask=m.to(device))
                vl_logits.append(logits.cpu())
                vl_labels.append(yb)
        vl_logits = torch.cat(vl_logits)
        vl_labels = torch.cat(vl_labels)
        vl_loss = crit(vl_logits.to(device), vl_labels.to(device)).item()
        vl_pred = (torch.sigmoid(vl_logits).numpy() >= 0.5).astype(int)
        vl_m = logger.metrics(vl_labels.numpy(), vl_pred)
        sched.step(vl_m['bacc'])

        history['train_loss'].append(float(tr_loss))
        history['val_loss'].append(float(vl_loss))
        history['train_acc'].append(tr_m['acc'])
        history['val_acc'].append(vl_m['acc'])
        for k in ('bacc', 'f1', 'sens', 'spec'):
            history[f'val_{k}'].append(vl_m[k])

        if vl_m['bacc'] > best_vb:
            best_vb = vl_m['bacc']
            best_st = copy.deepcopy(model.state_dict())
            best_ep = ep
            pat = 0
        else:
            pat += 1

        if ep == 1 or pat == 0 or ep % 10 == 0:
            logger.log_epoch(ep, tr_loss, vl_loss, tr_m, vl_m, pat)

        if pat >= args.fusion_patience:
            break

    if best_st is not None:
        model.load_state_dict(best_st)
    return model, best_vb, best_ep, history

# ── Mixup ──

def mixup_features(z_e, z_a, y, alpha=0.2):
    if alpha <= 0:
        return z_e, z_a, y, None
    B = z_e.shape[0]
    lam = np.random.beta(alpha, alpha, size=B).astype(np.float32)
    lam = torch.from_numpy(lam).to(z_e.device)
    perm = torch.randperm(B, device=z_e.device)
    z_e_mix = lam.view(-1, 1, 1) * z_e + (1 - lam).view(-1, 1, 1) * z_e[perm]
    z_a_mix = lam.view(-1, 1, 1) * z_a + (1 - lam).view(-1, 1, 1) * z_a[perm]
    y_mix = lam * y + (1 - lam) * y[perm]
    return z_e_mix, z_a_mix, y_mix, perm

# ── Evaluation ──

def evaluate_fusion(model, Z_e, Z_a, mask, y_true):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(Z_e).to(device),
                       torch.FloatTensor(Z_a).to(device),
                       mask=torch.FloatTensor(mask).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return y_true, preds, probs

# ── Build subject index helpers ──

def build_subject_dict(data_list, labels_arr, cods_list):
    return {cods_list[i]: {'windows': data_list[i], 'label': int(labels_arr[i])}
            for i in range(len(cods_list))}

def build_backbone_dataset(eeg_dict, aud_dict, mapping, train_paired_ids):
    paired_eeg_ids = [p[0] for p in train_paired_ids]
    paired_aud_ids = [p[1] for p in train_paired_ids]
    mapped_eeg = set(mapping.values())
    eeg_keys = [k for k in eeg_dict.keys()]
    eeg_data_list, eeg_labels_list, eeg_cods_list = [], [], []
    for sid in eeg_keys:
        if sid in paired_eeg_ids or sid not in mapped_eeg:
            eeg_data_list.append(eeg_dict[sid]['windows'])
            eeg_labels_list.append(eeg_dict[sid]['label'])
            eeg_cods_list.append(sid)
    mapped_aud = set(mapping.keys())
    aud_keys = [k for k in aud_dict.keys()]
    aud_data_list, aud_labels_list, aud_cods_list = [], [], []
    for sid in aud_keys:
        if sid in paired_aud_ids or sid not in mapped_aud:
            aud_data_list.append(aud_dict[sid]['windows'])
            aud_labels_list.append(aud_dict[sid]['label'])
            aud_cods_list.append(sid)
    return (eeg_data_list, np.array(eeg_labels_list, dtype=int), eeg_cods_list,
            aud_data_list, np.array(aud_labels_list, dtype=int), aud_cods_list)


def main():
    parser = argparse.ArgumentParser(description='Strict nested cross-modal training (no fusion validation leakage)')
    parser.add_argument('--fusion', choices=['concat', 'gating', 'cross_attn'], default='cross_attn')
    parser.add_argument('--n-self-attn-layers', type=int, default=1)
    parser.add_argument('--self-attn-heads', type=int, default=4)
    parser.add_argument('--self-attn-dropout', type=float, default=0.1)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=1)
    parser.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--bottleneck-dim', type=int, default=None)
    parser.add_argument('--max-windows', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr-fusion', type=float, default=5e-4)
    parser.add_argument('--wd-fusion', type=float, default=1e-3)
    parser.add_argument('--fusion-epochs', type=int, default=200)
    parser.add_argument('--fusion-patience', type=int, default=30)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--augment-backbone', action='store_true',
                        help='Apply augmentation during backbone training (WindowDataset)')
    parser.add_argument('--noise-std', type=float, default=0.05)
    parser.add_argument('--time-mask-max', type=int, default=20)
    parser.add_argument('--channel-drop-ratio', type=float, default=0.15)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--adapter-dim', type=int, default=None)
    parser.add_argument('--window-aux', action='store_true')
    parser.add_argument('--window-aux-weight', type=float, default=0.3)
    parser.add_argument('--mixup-alpha', type=float, default=0.0)
    parser.add_argument('--feat-dropout', type=float, default=0.0)
    parser.add_argument('--loocv', action='store_true')
    parser.add_argument('--inner-folds', type=int, default=INNER_FUSION_FOLDS,
                        help='Number of inner CV folds for fusion validation (default: 3)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Custom suffix appended to config name for experiment tracking')
    parser.add_argument('--seed', type=int, nargs='+', default=[42],
                        help='Seed(s). Single: --seed 42. Multiple sequential: --seed 42 54 100')
    parser.add_argument('--init-seed', type=int, nargs='+', default=None,
                        help='Seed(s) for subject CV partition. When set, --seed is the fixed init seed.')
    args = parser.parse_args()
    if args.init_seed is not None:
        init_seed = parse_seeds(args.seed)[0]
        for cv_seed in parse_seeds(args.init_seed):
            run_experiment(init_seed, args, cv_seed=cv_seed)
    else:
        for seed in parse_seeds(args.seed):
            run_experiment(seed, args)


def run_experiment(seed, args, cv_seed=None):
    global RANDOM_STATE
    RANDOM_STATE = seed
    set_seed(RANDOM_STATE)
    if cv_seed is None:
        cv_seed = seed

    if len(parse_seeds(args.seed)) > 1 or (args.init_seed is not None and len(args.init_seed) > 1):
        print("\n" + "#" * 60)
        print(f"  Seed run: init={seed}  partition={cv_seed}")
        print("#" * 60)


    git_commit = ''
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_commit = 'unknown'

    cfg_name = f'mhcmattention_seed{RANDOM_STATE}_part{cv_seed}_{args.fusion}'
    cfg_name += f'_h{args.hidden}'
    if args.n_self_attn_layers > 0:
        cfg_name += f'_self{args.n_self_attn_layers}L'
    if args.n_heads != 1:
        cfg_name += f'_hd{args.n_heads}'
    if args.bottleneck_dim is not None:
        cfg_name += f'_bn{args.bottleneck_dim}'
    if args.pooling == 'cls':
        cfg_name += '_cls'
    if args.augment:
        cfg_name += '_aug'
    if args.augment_backbone:
        cfg_name += '_bba'  # backbone augmentation
    if args.adapter_dim is not None:
        cfg_name += f'_ad{args.adapter_dim}'
    if args.window_aux:
        cfg_name += '_wax'
    if args.mixup_alpha > 0:
        cfg_name += f'_mix{args.mixup_alpha}'
    if args.feat_dropout > 0:
        cfg_name += f'_fd{args.feat_dropout}'
    if args.loocv:
        cfg_name += '_loocv'
    if args.inner_folds != 3:
        cfg_name += f'_inf{args.inner_folds}'
    cfg_name += f'_w{args.max_windows}'
    if args.tag is not None:
        cfg_name += f'_{args.tag}'

    out_dir = os.path.join(OUTPUT_DIR, cfg_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Device: {device}')
    print(f'Strict Nested CrossModal — {cfg_name}')
    print(f'  Fusion={args.fusion}  Self-attn={args.n_self_attn_layers}L')
    print(f'  Augment={args.augment}  AugmentBackbone={args.augment_backbone}  Max windows={args.max_windows}')
    print(f'  Adapter dim={args.adapter_dim}  Window aux={args.window_aux}  Mixup alpha={args.mixup_alpha}')
    print(f'  Feat dropout={args.feat_dropout}  LOOCV={args.loocv}')
    print(f'  Inner fusion folds={args.inner_folds}  (backbones re-trained per inner fold)')
    print(f'  Backbone: lr={args.lr} wd={args.wd} epochs={args.epochs}')
    print(f'  Fusion:   lr={args.lr_fusion} wd={args.wd_fusion} epochs={args.fusion_epochs}')

    # Load data
    eeg_data, eeg_labels, eeg_cods = _load_eeg_cache()
    aud_data, aud_labels, aud_cods = _load_audio_cache()
    mapping = _load_mapping()
    eeg_dict = build_subject_dict(eeg_data, eeg_labels, eeg_cods)
    aud_dict = build_subject_dict(aud_data, aud_labels, aud_cods)

    # ── Backbone augmentation (if enabled) ──
    bb_eeg_aug = EEGAugment(noise_std=args.noise_std,
                           time_mask_max=args.time_mask_max,
                           channel_drop_ratio=args.channel_drop_ratio) if args.augment_backbone else None
    bb_aud_aug = AudioAugment(noise_std=args.noise_std) if args.augment_backbone else None

    # Build paired list
    pairs = []
    for aud_id, eeg_id in mapping.items():
        if eeg_id in eeg_dict and aud_id in aud_dict:
            pairs.append((eeg_id, aud_id, eeg_dict[eeg_id]['label']))
    labels = np.array([p[2] for p in pairs])
    group_ids = np.array([f'p{i}' for i in range(len(pairs))])
    print(f'  Paired subjects: {len(pairs)} ({int(labels.sum())} MDD, {len(pairs)-int(labels.sum())} HC)')
    print(f'  Outer CV: {"LOOCV (38 folds)" if args.loocv else "5-fold"}')

    # Outer CV
    if args.loocv:
        splitter = LeaveOneGroupOut()
    else:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
    fold_results = []

    for fi, (tvi, tei) in enumerate(splitter.split(np.zeros(len(pairs)), labels, groups=group_ids)):
        print(f'\n{"="*60}')
        print(f'  Outer Fold {fi + 1}')
        print(f'{"="*60}')
        try:
            tr_paired = [pairs[i] for i in tvi]
            te_paired = [pairs[i] for i in tei]

            # Build backbone dataset from ALL tr_paired + unpaired
            eeg_bb_data, eeg_bb_labels, eeg_bb_cods, \
                aud_bb_data, aud_bb_labels, aud_bb_cods = \
                build_backbone_dataset(eeg_dict, aud_dict, mapping, tr_paired)

            print('  Backbone pool (all):')
            print(f'    EEG: {len(eeg_bb_cods)} subj ({int(eeg_bb_labels.sum())} MDD, '
                  f'{int((1-eeg_bb_labels).sum())} HC)')
            print(f'    Audio: {len(aud_bb_cods)} subj ({int(aud_bb_labels.sum())} MDD, '
                  f'{int((1-aud_bb_labels).sum())} HC)')
            print(f'    Paired train: {len(tr_paired)}  Test: {len(te_paired)}')

            # ── Inner CV for fusion validation ──
            y_tr = np.array([p[2] for p in tr_paired], dtype=np.float32)
            inner_splitter = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True,
                                                  random_state=cv_seed + fi)

            inner_best_vbs = []  # list of val_bacc per inner fold
            inner_best_eps = []  # list of best_epoch per inner fold
            inner_folds_info = []  # list of dicts with subject IDs per inner fold

            for inner_fi, (fuse_tr_i, fuse_vl_i) in enumerate(
                    inner_splitter.split(np.zeros(len(tr_paired)), y_tr,
                                         groups=[f'p{i}' for i in range(len(tr_paired))])):
                print(f'\n  --- Inner fold {inner_fi + 1}/{args.inner_folds} ---')

                # Identify which paired subjects are in inner validation
                inner_vl_paired = [tr_paired[i] for i in fuse_vl_i]
                inner_tr_paired = [tr_paired[i] for i in fuse_tr_i]

                # Build backbone training set EXCLUDING inner_vl subjects
                eeg_bb_tr_data, eeg_bb_tr_labels, eeg_bb_tr_cods, \
                    aud_bb_tr_data, aud_bb_tr_labels, aud_bb_tr_cods = \
                    build_backbone_dataset(eeg_dict, aud_dict, mapping, inner_tr_paired)

                # ── Train clean EEG backbone ──
                print('    Training clean EEG backbone (inner_vl excluded)...')
                inner_seed = cv_seed + fi * 10 + inner_fi
                inner_bb_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=inner_seed)
                eeg_bb_tr_i, eeg_bb_vl_i = next(inner_bb_split.split(
                    np.zeros(len(eeg_bb_tr_labels)), eeg_bb_tr_labels, groups=eeg_bb_tr_cods))
                tr_ds = WindowDataset(eeg_bb_tr_data, eeg_bb_tr_labels, eeg_bb_tr_cods,
                                      [eeg_bb_tr_i[i] for i in range(len(eeg_bb_tr_i))],
                                      max_windows=args.max_windows, augmenter=bb_eeg_aug)
                vl_ds = WindowDataset(eeg_bb_tr_data, eeg_bb_tr_labels, eeg_bb_tr_cods,
                                      [eeg_bb_vl_i[i] for i in range(len(eeg_bb_vl_i))],
                                      max_windows=args.max_windows)
                tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
                vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False)
                eeg_model = DeepConvNetWrapper(64, N_EEG_SAMPLES).to(device)
                if fi == 0 and inner_fi == 0:
                    print(f'    EEG params: {sum(p.numel() for p in eeg_model.parameters()):,}')
                eeg_best_st, eeg_best_vb, _, _ = train_backbone(eeg_model, tr_ldr, vl_ldr, args)
                eeg_model.load_state_dict(eeg_best_st)
                eeg_model.eval()
                del tr_ds, vl_ds, tr_ldr, vl_ldr

                # ── Train clean audio backbone ──
                print('    --- Training clean Audio backbone...')
                aud_bb_tr_i, aud_bb_vl_i = next(inner_bb_split.split(
                    np.zeros(len(aud_bb_tr_labels)), aud_bb_tr_labels, groups=aud_bb_tr_cods))
                tr_ds = WindowDataset(aud_bb_tr_data, aud_bb_tr_labels, aud_bb_tr_cods,
                                       [aud_bb_tr_i[i] for i in range(len(aud_bb_tr_i))],
                                       max_windows=args.max_windows, augmenter=bb_aud_aug)
                vl_ds = WindowDataset(aud_bb_tr_data, aud_bb_tr_labels, aud_bb_tr_cods,
                                      [aud_bb_vl_i[i] for i in range(len(aud_bb_vl_i))],
                                      max_windows=args.max_windows)
                tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
                vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False)
                aud_model = ShallowConvNetWrapper(N_MELS, N_AUDIO_SAMPLES).to(device)
                if fi == 0 and inner_fi == 0:
                    print(f'    Audio params: {sum(p.numel() for p in aud_model.parameters()):,}')
                aud_best_st, aud_best_vb, _, _ = train_backbone(aud_model, tr_ldr, vl_ldr, args)
                aud_model.load_state_dict(aud_best_st)
                aud_model.eval()
                del tr_ds, vl_ds, tr_ldr, vl_ldr

                # ── Extract features with CLEAN backbones ──
                eeg_aug = EEGAugment(noise_std=args.noise_std,
                                   time_mask_max=args.time_mask_max,
                                   channel_drop_ratio=args.channel_drop_ratio) if args.augment else None
                aud_aug = AudioAugment(noise_std=args.noise_std) if args.augment else None

                Z_e_tr, Z_a_tr, mask_tr = extract_all_features(
                    eeg_model, aud_model, inner_tr_paired, eeg_dict, aud_dict,
                    args.max_windows, eeg_augment=eeg_aug, aud_augment=aud_aug)
                Z_e_vl, Z_a_vl, mask_vl = extract_all_features(
                    eeg_model, aud_model, inner_vl_paired, eeg_dict, aud_dict,
                    args.max_windows)

                y_inner_tr = np.array([p[2] for p in inner_tr_paired], dtype=np.float32)
                y_inner_vl = np.array([p[2] for p in inner_vl_paired], dtype=np.float32)

                eeg_dim = Z_e_tr.shape[2]
                aud_dim = Z_a_tr.shape[2]

                # ── Train fusion head ──
                fusion_model = CrossModalAttention(
                    eeg_dim=eeg_dim, aud_dim=aud_dim,
                    hidden=args.hidden, n_heads=args.n_heads,
                    bottleneck_dim=args.bottleneck_dim,
                    n_self_attn_layers=args.n_self_attn_layers,
                    self_attn_heads=args.self_attn_heads,
                    self_attn_dropout=args.self_attn_dropout,
                    fusion=args.fusion, pooling=args.pooling, dropout=args.dropout,
                    adapter_dim=args.adapter_dim, window_aux=args.window_aux,
                    feat_dropout=args.feat_dropout,
                ).to(device)

                fusion_model, _, fusion_best_ep, _ = train_fusion_head(
                    fusion_model,
                    Z_e_tr, Z_a_tr, mask_tr, y_inner_tr,
                    Z_e_vl, Z_a_vl, mask_vl, y_inner_vl,
                    args)

                # Evaluate on inner validation
                yt, yp, ypr = [], [], []
                for si in range(len(inner_vl_paired)):
                    yt_s, yp_s, ypr_s = evaluate_fusion(
                        fusion_model,
                        Z_e_vl[si:si+1], Z_a_vl[si:si+1],
                        mask_vl[si:si+1], y_inner_vl[si:si+1])
                    yt.append(yt_s[0])
                    yp.append(yp_s[0])
                    ypr.append(ypr_s[0])
                inner_bacc = balanced_accuracy_score(np.array(yt), np.array(yp))
                print(f'    >>> Inner fold {inner_fi + 1} val_bacc={inner_bacc:.3f}  best_ep={fusion_best_ep}')
                inner_best_vbs.append(inner_bacc)
                inner_best_eps.append(fusion_best_ep)
                inner_folds_info.append({
                    'inner_fold': inner_fi + 1,
                    'inner_val_bacc': float(inner_bacc),
                    'inner_train_subjects': [p[0] for p in inner_tr_paired],
                    'inner_val_subjects': [p[0] for p in inner_vl_paired],
                    'eeg_backbone_subjects': eeg_bb_tr_cods,
                    'aud_backbone_subjects': aud_bb_tr_cods,
                })

                # Cleanup
                del eeg_model, aud_model, fusion_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ── Aggregate inner CV results ──
            avg_inner_val = float(np.mean(inner_best_vbs))
            avg_best_ep = int(round(np.mean(inner_best_eps)))
            final_fusion_epochs = args.fusion_epochs
            print(f'\n  *** Inner CV avg val_bacc={avg_inner_val:.3f}  avg_best_ep={avg_best_ep}  '
                  f'final_fusion_epochs={final_fusion_epochs} (cosine annealing) ***')

            # ── Now train FINAL model on ALL tr_paired (no more validation needed) ──
            print('\n  --- Training FINAL backbones on ALL paired subjects ---')

            # Train EEG backbone on ALL tr_paired + unpaired
            inner_seed = cv_seed + fi
            final_bb_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=inner_seed)
            eeg_f_tr_i, eeg_f_vl_i = next(final_bb_split.split(
                np.zeros(len(eeg_bb_labels)), eeg_bb_labels, groups=eeg_bb_cods))
            tr_ds = WindowDataset(eeg_bb_data, eeg_bb_labels, eeg_bb_cods,
                                  [eeg_f_tr_i[i] for i in range(len(eeg_f_tr_i))],
                                  max_windows=args.max_windows, augmenter=bb_eeg_aug)
            vl_ds = WindowDataset(eeg_bb_data, eeg_bb_labels, eeg_bb_cods,
                                   [eeg_f_vl_i[i] for i in range(len(eeg_f_vl_i))],
                                   max_windows=args.max_windows)
            tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
            vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False)
            eeg_model = DeepConvNetWrapper(64, N_EEG_SAMPLES).to(device)
            eeg_best_st, eeg_best_vb, _, eeg_history = train_backbone(eeg_model, tr_ldr, vl_ldr, args)
            eeg_model.load_state_dict(eeg_best_st)
            eeg_model.eval()
            del tr_ds, vl_ds, tr_ldr, vl_ldr
            print(f'    EEG backbone best val bacc: {eeg_best_vb:.3f}')

            # Train audio backbone on ALL tr_paired + unpaired
            aud_f_tr_i, aud_f_vl_i = next(final_bb_split.split(
                np.zeros(len(aud_bb_labels)), aud_bb_labels, groups=aud_bb_cods))
            tr_ds = WindowDataset(aud_bb_data, aud_bb_labels, aud_bb_cods,
                                  [aud_f_tr_i[i] for i in range(len(aud_f_tr_i))],
                                  max_windows=args.max_windows, augmenter=bb_aud_aug)
            vl_ds = WindowDataset(aud_bb_data, aud_bb_labels, aud_bb_cods,
                                   [aud_f_vl_i[i] for i in range(len(aud_f_vl_i))],
                                   max_windows=args.max_windows)
            tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
            vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False)
            aud_model = ShallowConvNetWrapper(N_MELS, N_AUDIO_SAMPLES).to(device)
            aud_best_st, aud_best_vb, _, aud_history = train_backbone(aud_model, tr_ldr, vl_ldr, args)
            aud_model.load_state_dict(aud_best_st)
            aud_model.eval()
            del tr_ds, vl_ds, tr_ldr, vl_ldr
            print(f'    Audio backbone best val bacc: {aud_best_vb:.3f}')

            # Extract features from ALL tr_paired (final)
            eeg_aug = None
            aud_aug = None
            Z_e_tr, Z_a_tr, mask_tr = extract_all_features(
                eeg_model, aud_model, tr_paired, eeg_dict, aud_dict,
                args.max_windows)

            # Extract test features
            Z_e_te, Z_a_te, mask_te = extract_all_features(
                eeg_model, aud_model, te_paired, eeg_dict, aud_dict,
                args.max_windows)

            y_tr = np.array([p[2] for p in tr_paired], dtype=np.float32)
            y_te = np.array([p[2] for p in te_paired], dtype=np.float32)

            eeg_dim = Z_e_tr.shape[2]
            aud_dim = Z_a_tr.shape[2]
            print(f'\n    EEG feat dim={eeg_dim}  Audio feat dim={aud_dim}')
            print(f'    Train subjects={len(tr_paired)}  Test subjects={len(te_paired)}')

            # ── Train final fusion head ──
            # Train on ALL tr_paired with all epochs (no early stopping from contaminated val)
            # Use fusion_epochs as max; no early stopping needed since we train on all data
            print('\n  --- Training final fusion head on ALL paired ---')
            fusion_model = CrossModalAttention(
                eeg_dim=eeg_dim, aud_dim=aud_dim,
                hidden=args.hidden, n_heads=args.n_heads,
                bottleneck_dim=args.bottleneck_dim,
                n_self_attn_layers=args.n_self_attn_layers,
                self_attn_heads=args.self_attn_heads,
                self_attn_dropout=args.self_attn_dropout,
                fusion=args.fusion, pooling=args.pooling, dropout=args.dropout,
                adapter_dim=args.adapter_dim, window_aux=args.window_aux,
                feat_dropout=args.feat_dropout,
            ).to(device)

            # Train on ALL training data (no validation split)
            ds_all = torch.utils.data.TensorDataset(
                torch.FloatTensor(Z_e_tr), torch.FloatTensor(Z_a_tr),
                torch.FloatTensor(mask_tr), torch.FloatTensor(y_tr))
            ld_all = DataLoader(ds_all, batch_size=args.bs, shuffle=True)

            opt = torch.optim.AdamW(fusion_model.parameters(), lr=args.lr_fusion,
                                    weight_decay=args.wd_fusion, foreach=False)
            n_mdd = y_tr.sum()
            n_hc = len(y_tr) - n_mdd
            pos_weight = torch.tensor([n_hc / max(n_mdd, 1)]).to(device)
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            scheduler = CosineAnnealingLR(opt, T_max=final_fusion_epochs)
            fusion_hist = {'train_loss': [], 'train_acc': [], 'train_bacc': [], 'lr': []}
            for ep in range(1, final_fusion_epochs + 1):
                fusion_model.train()
                tr_loss, tr_n = 0.0, 0
                tr_logits, tr_labels = [], []
                for ze, za, m, yb in ld_all:
                    ze, za, m, yb = ze.to(device), za.to(device), m.to(device), yb.to(device)
                    opt.zero_grad()
                    tr_labels.append(yb)
                    if args.mixup_alpha > 0:
                        ze, za, yb, _ = mixup_features(ze, za, yb, args.mixup_alpha)
                    logits = fusion_model(ze, za, mask=m)
                    y_smooth = yb * 0.95 + 0.025
                    loss = crit(logits, y_smooth)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
                    opt.step()
                    tr_loss += loss.item() * yb.size(0)
                    tr_n += yb.size(0)
                    tr_logits.append(logits.detach())
                scheduler.step()
                tr_loss /= tr_n
                tr_pred = (torch.sigmoid(torch.cat(tr_logits)).cpu().numpy() >= 0.5).astype(int)
                tr_true = torch.cat(tr_labels).cpu().numpy()
                tr_bacc = balanced_accuracy_score(tr_true, tr_pred)
                tr_acc = (tr_pred == tr_true).mean()
                current_lr = scheduler.get_last_lr()[0]
                fusion_hist['train_loss'].append(float(tr_loss))
                fusion_hist['train_acc'].append(float(tr_acc))
                fusion_hist['train_bacc'].append(float(tr_bacc))
                fusion_hist['lr'].append(float(current_lr))
                if ep == 1 or ep % 10 == 0:
                    print(f'    Epoch {ep}: train_loss={tr_loss:.4f}  train_bacc={tr_bacc:.3f}  lr={current_lr:.2e}')

            # ── Evaluate on test subjects ──
            print('\n  --- Evaluating ---')
            y_true_list, y_pred_list, y_prob_list = [], [], []
            for si in range(len(te_paired)):
                yt, yp, ypr = evaluate_fusion(
                    fusion_model,
                    Z_e_te[si:si+1], Z_a_te[si:si+1],
                    mask_te[si:si+1], y_te[si:si+1])
                y_true_list.append(yt[0])
                y_pred_list.append(yp[0])
                y_prob_list.append(ypr[0])

            y_true_s = np.array(y_true_list)
            y_pred_s = np.array(y_pred_list)
            y_prob_s = np.array(y_prob_list)

            cm = confusion_matrix(y_true_s, y_pred_s).tolist()
            roc_auc = float(roc_auc_score(y_true_s, y_prob_s))
            bacc = balanced_accuracy_score(y_true_s, y_pred_s)
            logger = ClassificationLogger()
            fm = logger.log_fold_test(y_true_s, y_pred_s)

            fold_results.append({
                'fold': fi + 1,
                'inner_cv_val_bacc': avg_inner_val,
                'inner_cv_best_epoch_mean': avg_best_ep,
                'inner_cv_best_epoch_std': float(np.std(inner_best_eps)),
                'final_fusion_epochs': final_fusion_epochs,
                'inner_folds_val_baccs': [float(v) for v in inner_best_vbs],
                'inner_folds_best_epochs': [int(e) for e in inner_best_eps],
                'eeg_backbone_val_bacc': float(eeg_best_vb),
                'aud_backbone_val_bacc': float(aud_best_vb),
                'test_metrics': fm,
                'test_bacc': float(bacc),
                'test_acc': float(fm['acc']),
                'test_f1': float(fm['f1']),
                'test_sens': float(fm['sens']),
                'test_spec': float(fm['spec']),
                'test_auc': roc_auc,
                'test_cm': cm,
                'test_roc': {'y_true': y_true_s.tolist(), 'y_prob': y_prob_s.tolist()},
                'n_backbone_eeg': len(eeg_bb_cods),
                'n_backbone_aud': len(aud_bb_cods),
                'n_train_paired': len(tr_paired),
                'n_test': len(te_paired),
                'test_subjects': [p[0] for p in te_paired],
                'eeg_history': eeg_history,
                'aud_history': aud_history,
                'fusion_history': fusion_hist,
                # Subject tracking
                'train_subjects': [p[0] for p in tr_paired],
                'eeg_backbone_subjects': eeg_bb_cods,
                'aud_backbone_subjects': aud_bb_cods,
                'inner_folds': inner_folds_info,
            })
            print(f'  Fold {fi + 1}: inner_cv_val={avg_inner_val:.3f}  '
                  f'test_bacc={bacc:.3f}  test_auc={roc_auc:.3f}')

            # Save checkpoint
            if args.save_model:
                ckpt_dir = os.path.join(out_dir, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    'fold': fi + 1,
                    'fusion_state_dict': fusion_model.state_dict(),
                    'eeg_backbone_state': eeg_best_st,
                    'aud_backbone_state': aud_best_st,
                    'args': vars(args),
                    'test_bacc': float(bacc),
                    'test_auc': roc_auc,
                }, os.path.join(ckpt_dir, f'fold_{fi+1}.pt'))
                print(f'    Saved: fold_{fi+1}.pt')

            # Partial save after each fold
            partial_baccs = [r['test_bacc'] for r in fold_results]
            partial_aucs = [r['test_auc'] for r in fold_results]
            partial_test = {
                'bacc_mean': float(np.mean(partial_baccs)),
                'bacc_std': float(np.std(partial_baccs)),
                'auc_mean': float(np.mean(partial_aucs)),
                'auc_std': float(np.std(partial_aucs)),
            }
            partial_results = {
                'experiment': {
                    'name': 'crossmodal_nested',
                    'fusion': args.fusion,
                    'script': 'train_crossmodal_sngkf.py',
                    'timestamp': datetime.now().isoformat(),
                    'git_commit': git_commit,
                    'seed': RANDOM_STATE,
                },
                'data': {
                    'n_eeg': len(eeg_dict),
                    'n_audio': len(aud_dict),
                    'n_paired': len(pairs),
                    'n_mdd_paired': int(labels.sum()),
                    'n_hc_paired': len(pairs) - int(labels.sum()),
                    'n_folds': len(fold_results),
                    'inner_fusion_folds': args.inner_folds,
                },
                'config': {
                    'fusion': args.fusion,
                    'hidden': args.hidden,
                    'n_heads': args.n_heads,
                    'dropout': args.dropout,
                    'max_windows': args.max_windows,
                    'n_self_attn_layers': args.n_self_attn_layers,
                    'self_attn_heads': args.self_attn_heads,
                    'self_attn_dropout': args.self_attn_dropout,
                    'pooling': args.pooling,
                    'bottleneck_dim': args.bottleneck_dim,
                    'backbone_lr': args.lr,
                    'backbone_wd': args.wd,
                    'backbone_epochs': args.epochs,
                    'backbone_patience': args.patience,
                    'fusion_lr': args.lr_fusion,
                    'fusion_wd': args.wd_fusion,
                    'fusion_epochs': args.fusion_epochs,
                    'fusion_patience': args.fusion_patience,
                    'batch_size': args.bs,
                    'backbone_eeg': 'DeepConvNet',
                    'backbone_aud': 'ShallowConvNet',
                    'augment': args.augment,
                    'augment_backbone': args.augment_backbone,
                    'adapter_dim': args.adapter_dim,
                    'window_aux': args.window_aux,
                    'window_aux_weight': args.window_aux_weight,
                    'mixup_alpha': args.mixup_alpha,
                    'feat_dropout': args.feat_dropout,
                    'loocv': args.loocv,
                    'inner_fusion_folds': args.inner_folds,
                },
                'test': partial_test,
                'folds': fold_results,
                'partial': True,
            }
            partial_path = os.path.join(out_dir, 'results.json')
            with open(partial_path, 'w') as f:
                json.dump(partial_results, f, indent=2)

            del eeg_model, aud_model, fusion_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  Fold {fi + 1} FAILED: {e}')
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Summary ──
    if fold_results:
        baccs = [r['test_bacc'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        inner_vbs = [r['inner_cv_val_bacc'] for r in fold_results]

        test_accs = [r['test_metrics']['acc'] for r in fold_results]
        test_f1s = [r['test_metrics']['f1'] for r in fold_results]
        test_sens = [r['test_metrics']['sens'] for r in fold_results]
        test_spec = [r['test_metrics']['spec'] for r in fold_results]

        test = {
            'bacc_mean': float(np.mean(baccs)),
            'bacc_std': float(np.std(baccs)),
            'acc_mean': float(np.mean(test_accs)),
            'acc_std': float(np.std(test_accs)),
            'f1_mean': float(np.mean(test_f1s)),
            'f1_std': float(np.std(test_f1s)),
            'sens_mean': float(np.mean(test_sens)),
            'sens_std': float(np.std(test_sens)),
            'spec_mean': float(np.mean(test_spec)),
            'spec_std': float(np.std(test_spec)),
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
        }

        summary = {
            'bacc_mean': float(np.mean(baccs)),
            'bacc_std': float(np.std(baccs)),
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
            'inner_cv_val_mean': float(np.mean(inner_vbs)),
            'inner_cv_val_std': float(np.std(inner_vbs)),
        }

        print(f'\n{"="*55}')
        print(f'  {cfg_name}')
        print(f'  bacc = {summary["bacc_mean"]:.3f} ± {summary["bacc_std"]:.3f}')
        print(f'  auc  = {summary["auc_mean"]:.3f} ± {summary["auc_std"]:.3f}')
        print(f'  inner_cv_val = {summary["inner_cv_val_mean"]:.3f} ± {summary["inner_cv_val_std"]:.3f}')
        print(f'{"="*55}')

        out_results = {
            'experiment': {
                'name': 'crossmodal_nested',
                'fusion': args.fusion,
                'script': 'train_crossmodal_sngkf.py',
                'timestamp': datetime.now().isoformat(),
                'git_commit': git_commit,
                'seed': RANDOM_STATE,
            },
            'data': {
                'n_eeg': len(eeg_dict),
                'n_audio': len(aud_dict),
                'n_paired': len(pairs),
                'n_mdd_paired': int(labels.sum()),
                'n_hc_paired': len(pairs) - int(labels.sum()),
                'n_folds': len(fold_results),
                'inner_fusion_folds': args.inner_folds,
            },
            'config': {
                'fusion': args.fusion,
                'hidden': args.hidden,
                'n_heads': args.n_heads,
                'dropout': args.dropout,
                'max_windows': args.max_windows,
                'n_self_attn_layers': args.n_self_attn_layers,
                'self_attn_heads': args.self_attn_heads,
                'self_attn_dropout': args.self_attn_dropout,
                'pooling': args.pooling,
                'bottleneck_dim': args.bottleneck_dim,
                'backbone_lr': args.lr,
                'backbone_wd': args.wd,
                'backbone_epochs': args.epochs,
                'backbone_patience': args.patience,
                'fusion_lr': args.lr_fusion,
                'fusion_wd': args.wd_fusion,
                'fusion_epochs': args.fusion_epochs,
                'fusion_patience': args.fusion_patience,
                'batch_size': args.bs,
                'backbone_eeg': 'DeepConvNet',
                'backbone_aud': 'ShallowConvNet',
                'augment': args.augment,
                'augment_backbone': args.augment_backbone,
                'adapter_dim': args.adapter_dim,
                'window_aux': args.window_aux,
                'window_aux_weight': args.window_aux_weight,
                'mixup_alpha': args.mixup_alpha,
                'feat_dropout': args.feat_dropout,
                'loocv': args.loocv,
                'inner_fusion_folds': args.inner_folds,
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
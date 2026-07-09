"""
Strict 0-leakage cross-modal training for MODMA.
For each external fold k (on 38 paired subjects):
  1. Test subjects (~7) frozen until final eval
  2. Backbone set = remaining ~31 paired + all EEG-only + all audio-only (~60 total)
  3. Train EEG+audio backbones on backbone set (inner val for early stopping)
  4. For each fusion epoch: augment raw windows -> extract via frozen backbones -> train head
  5. Evaluate on held-out test subjects

Usage:
  py src/training/train_crossmodal_strict.py --fusion cross_attn
  py src/training/train_crossmodal_strict.py --fusion cross_attn --augment
  py src/training/train_crossmodal_strict.py --fusion gating --augment

Output: outputs/results/crossmodal_strict/{config_name}/results.json
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
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from datetime import datetime
import subprocess
from src.models.crossmodal_attn import CrossModalAttention
from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.utils.training_logger import ClassificationLogger

# ── Backbone wrappers (squeeze output for BCEWithLogitsLoss) ──

class DeepConvNetWrapper(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()
        self.m = DeepConvNet(n_channels, 1, n_samples, 0.5)
    def forward(self, x): return self.m(x).squeeze(-1)

class ShallowConvNetWrapper(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()
        self.m = ShallowConvNet(n_channels, 1, n_samples, 0.5)
    def forward(self, x): return self.m(x).squeeze(-1)

# ── Joint fine-tune model ─────────────────────────────────────────────────

class JointModel(nn.Module):
    """EEG backbone + Audio backbone + Fusion head, trainable end-to-end."""
    def __init__(self, eeg_backbone, aud_backbone, fusion_model):
        super().__init__()
        self.eeg_backbone = eeg_backbone
        self.aud_backbone = aud_backbone
        self.fusion = fusion_model
        self.eeg_dim = fusion_model.eeg_proj.in_features
        self.aud_dim = fusion_model.aud_proj.in_features

    def forward(self, eeg_wins, aud_wins, mask=None, return_window=False):
        B, K = eeg_wins.shape[0], eeg_wins.shape[1]
        # Flatten windows: ensure we collapse to [B*K, C, T]
        eeg_flat = eeg_wins.reshape(B * K, *eeg_wins.shape[2:])
        aud_flat = aud_wins.reshape(B * K, *aud_wins.shape[2:])

        ze_list, za_list = [], []
        for i in range(0, B * K, 32):
            e_batch = eeg_flat[i:i+32].float().to(device)
            a_batch = aud_flat[i:i+32].float().to(device)
            # DeepConvNet.forward checks dim and unsqueezes if 3D
            # ShallowConvNet.forward always unsqueezes(1)
            # Pass 3D [B, C, T] to both; they handle spatial dim internally
            ze_list.append(self.eeg_backbone.m(e_batch))
            za_list.append(self.aud_backbone.m(a_batch))

        ze = torch.cat(ze_list, dim=0)
        za = torch.cat(za_list, dim=0)

        C_e = ze.shape[1]
        C_a = za.shape[1]
        ze = ze.reshape(B, K, C_e)
        za = za.reshape(B, K, C_a)

        if ze.shape[-1] != self.eeg_dim or za.shape[-1] != self.aud_dim:
            pad_e = self.eeg_dim - ze.shape[-1]
            pad_a = self.aud_dim - za.shape[-1]
            if pad_e > 0:
                ze = torch.nn.functional.pad(ze, (0, pad_e))
            if pad_a > 0:
                za = torch.nn.functional.pad(za, (0, pad_a))
            ze = ze[..., :self.eeg_dim]
            za = za[..., :self.aud_dim]

        return self.fusion(ze, za, mask=mask, return_window=return_window)

    def forward_features(self, eeg_wins, aud_wins):
        B, K = eeg_wins.shape[0], eeg_wins.shape[1]
        eeg_wins_flat = eeg_wins.reshape(B * K, *eeg_wins.shape[2:])
        aud_wins_flat = aud_wins.reshape(B * K, *aud_wins.shape[2:])

        ze_list, za_list = [], []
        for i in range(0, eeg_wins_flat.shape[0], 32):
            e_batch = eeg_wins_flat[i:i+32].float().to(device)
            a_batch = aud_wins_flat[i:i+32].float().to(device)
            if e_batch.dim() == 3:
                e_batch = e_batch.unsqueeze(1)
            if a_batch.dim() == 3:
                a_batch = a_batch.unsqueeze(1)
            with torch.no_grad():
                ze_list.append(self.eeg_backbone.m(e_batch))
                za_list.append(self.aud_backbone.m(a_batch))

        ze = torch.cat(ze_list, dim=0)
        za = torch.cat(za_list, dim=0)
        C_e = ze.shape[1]
        C_a = za.shape[1]
        ze = ze.reshape(B, K, C_e)
        za = za.reshape(B, K, C_a)

        if ze.shape[-1] != self.eeg_dim or za.shape[-1] != self.aud_dim:
            pad_e = self.eeg_dim - ze.shape[-1]
            pad_a = self.aud_dim - za.shape[-1]
            if pad_e > 0:
                ze = torch.nn.functional.pad(ze, (0, pad_e))
            if pad_a > 0:
                za = torch.nn.functional.pad(za, (0, pad_a))
            ze = ze[..., :self.eeg_dim]
            za = za[..., :self.aud_dim]
        return ze, za


class JointWindowDataset(Dataset):
    """Returns raw EEG + audio windows for joint fine-tune.
    Pads subjects with fewer windows to max_windows so batch dims match.
    """
    def __init__(self, eeg_dict, aud_dict, paired_list, max_windows, indices=None):
        self._pairs = []
        if indices is not None:
            paired_list = [paired_list[i] for i in indices]
        for eid, aid, lbl in paired_list:
            we = eeg_dict[eid]['windows']
            wa = aud_dict[aid]['windows']
            K = min(we.shape[0], wa.shape[0], max_windows)
            we = we[:K]
            wa = wa[:K]
            # Pad to max_windows if needed
            if K < max_windows:
                pad = max_windows - K
                we = np.pad(we, ((0, pad), (0, 0), (0, 0)), mode='constant')
                wa = np.pad(wa, ((0, pad), (0, 0), (0, 0)), mode='constant')
            self._pairs.append((we.astype(np.float32), wa.astype(np.float32), lbl, K))

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, i):
        we, wa, lbl, K = self._pairs[i]
        we = np.array([(we[j] - we[j].mean()) / (we[j].std() + 1e-8) for j in range(K)])
        wa = np.array([(wa[j] - wa[j].mean()) / (wa[j].std() + 1e-8) for j in range(K)])
        # Append zero-padding for masked windows
        pad = we.shape[0] - K
        if pad > 0:
            we = np.concatenate([we, np.zeros((pad, *we.shape[1:]), dtype=np.float32)], axis=0)
            wa = np.concatenate([wa, np.zeros((pad, *wa.shape[1:]), dtype=np.float32)], axis=0)
        mask = np.zeros(we.shape[0], dtype=np.float32)
        mask[:K] = 1.0
        return torch.FloatTensor(we), torch.FloatTensor(wa), torch.FloatTensor([lbl]), torch.FloatTensor(mask)


def train_joint(joint_model, tr_loader, vl_loader, args):
    """Joint fine-tune: backbones unfrozen, trained end-to-end with fusion."""
    joint_model.train()

    backbone_params = []
    fusion_params = []
    for name, p in joint_model.named_parameters():
        if 'fusion' in name or 'head' in name or 'cross' in name or 'proj' in name \
           or 'rms' in name or 'self_attn' in name or 'gate' in name or 'concat_proj' in name:
            fusion_params.append(p)
        else:
            backbone_params.append(p)

    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.joint_lr},
        {'params': fusion_params, 'lr': args.joint_lr * 5},
    ], weight_decay=args.wd_fusion, foreach=False)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

    # Compute pos_weight from entire training set
    n_mdd = 0
    for _, _, lbls, _ in tr_loader:
        n_mdd += lbls.sum().item()
    n_hc = len(tr_loader.dataset) - n_mdd
    pos_weight = torch.tensor([n_hc / max(n_mdd, 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_vb, best_st, pat = -1.0, None, 0
    logger = ClassificationLogger()
    logger.log_header()
    history = {k: [] for k in ('train_loss', 'val_loss', 'val_bacc', 'val_acc',
                                'val_f1', 'val_sens', 'val_spec')}

    for ep in range(1, args.joint_epochs + 1):
        joint_model.train()
        tr_loss, tr_n = 0.0, 0
        tr_logits, tr_labels = [], []

        for we, wa, yb, mask_b in tr_loader:
            we, wa, yb, mask_b = (we.to(device), wa.to(device),
                                   yb.to(device).float().squeeze(), mask_b.to(device))
            opt.zero_grad()
            logits = joint_model(we, wa, mask=mask_b)
            y_smooth = yb * 0.95 + 0.025
            loss = crit(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(joint_model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * yb.size(0)
            tr_n += yb.size(0)
            tr_logits.append(logits.detach())
            tr_labels.append(yb)

        tr_loss /= max(tr_n, 1)
        tr_pred = (torch.sigmoid(torch.cat(tr_logits)).cpu().numpy() >= 0.5).astype(int)
        tr_true = torch.cat(tr_labels).cpu().numpy()
        tr_m = logger.metrics(tr_true, tr_pred)

        # Validation
        joint_model.eval()
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for ze, wa, yb, mask_b in vl_loader:
                logits = joint_model(ze.to(device), wa.to(device), mask=mask_b.to(device))
                vl_logits.append(logits.cpu())
                vl_labels.append(yb.squeeze())
        vl_logits = torch.cat(vl_logits)
        vl_labels = torch.cat(vl_labels).float()
        vl_loss = crit(vl_logits.to(device), vl_labels.to(device)).item()
        vl_pred = (torch.sigmoid(vl_logits).numpy() >= 0.5).astype(int)
        vl_m = logger.metrics(vl_labels.numpy(), vl_pred)
        sched.step(vl_m['bacc'])

        history['train_loss'].append(float(tr_loss))
        history['val_loss'].append(float(vl_loss))
        history['val_bacc'].append(vl_m['bacc'])
        history['val_acc'].append(vl_m['acc'])
        for k in ('f1', 'sens', 'spec'):
            history[f'val_{k}'].append(vl_m[k])

        if vl_m['bacc'] > best_vb:
            best_vb = vl_m['bacc']
            best_st = copy.deepcopy(joint_model.state_dict())
            pat = 0
        else:
            pat += 1

        if ep == 1 or pat == 0 or ep % 10 == 0:
            logger.log_epoch(ep, tr_loss, vl_loss, tr_m, vl_m, pat)

        if pat >= args.joint_patience:
            break

    if best_st is not None:
        joint_model.load_state_dict(best_st)
    return joint_model, best_vb, history


# ── Data loading ──────────────────────────────────────────────────────────

EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/crossmodal_strict'
RANDOM_STATE = 42
N_FOLDS = 5
N_MELS = 64
N_AUDIO_SAMPLES = 200
N_EEG_SAMPLES = 500
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Data loading ──────────────────────────────────────────────────────────

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
        return json.load(f)['orig_to_bids']  # aud_id -> eeg_id

def _zscore(w):
    return (w - w.mean()) / (w.std() + 1e-8)

def _select_windows_deterministic(windows, max_windows):
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    indices = np.linspace(0, n - 1, max_windows, dtype=int)
    return windows[indices]

# ── Mixup helper ─────────────────────────────────────────────────────────

def mixup_features(z_e, z_a, y, alpha=0.2):
    """Apply mixup at subject level on pre-extracted features."""
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

# ── Augmentation ──────────────────────────────────────────────────────────

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

# ── Window dataset (for backbone training) ────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, windows_list, labels_list, subj_names, indices, max_windows=None):
        self._windows = windows_list
        self._subj_names = subj_names
        self._labels = labels_list
        self._index = []
        for idx in indices:
            wins = windows_list[idx]
            n = wins.shape[0]
            if max_windows is not None and n > max_windows:
                rng = np.random.RandomState(RANDOM_STATE + idx)
                keep = rng.choice(n, max_windows, replace=False)
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
        w = _zscore(w)
        return torch.from_numpy(w).float(), torch.tensor(label, dtype=torch.float), self._subj_names[idx]

# ── Backbone training ─────────────────────────────────────────────────────

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
    """Train one backbone, return (best_state, best_val_bacc, history)."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=False)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    crit = nn.BCEWithLogitsLoss()
    best_vb, best_st, pat = -1.0, None, 0
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
            pat = 0
        else:
            pat += 1
        if ep == 1 or pat == 0 or ep % 10 == 0:
            logger.log_epoch(ep, tr_loss, vl_loss, tr_m, vl_m, pat)
        if pat >= args.patience:
            break
    if best_st is None:
        best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_st, best_vb, history

# ── Feature extraction (with optional augmentation) ───────────────────────

def _encode_eeg(model, windows, device):
    """Forward EEG windows through DeepConvNet conv blocks, return features [K, 128]."""
    K = windows.shape[0]
    bb = model.m if hasattr(model, 'm') else model
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        with torch.no_grad():
            x = bb.block1(batch)
            x = bb.block2(x)
            x = bb.block3(x)
            x = bb.block4(x)
            feats.append(x.flatten(start_dim=1).cpu())
    return torch.cat(feats, dim=0).numpy()

def _encode_audio(model, windows, device):
    """Forward audio windows through ShallowConvNet conv blocks, return features [K, 576]."""
    K = windows.shape[0]
    bb = model.m if hasattr(model, 'm') else model
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        with torch.no_grad():
            x = bb.temporal_conv(batch)
            x = bb.spatial_conv(x)
            x = bb.bn(x)
            x = torch.square(x)
            x = bb.pool(x)
            x = torch.log(torch.clamp(x, min=1e-7))
            x = bb.dropout(x)
            feats.append(x.flatten(start_dim=1).cpu())
    return torch.cat(feats, dim=0).numpy()

def extract_subject_features(eeg_model, aud_model, eeg_wins, aud_wins,
                              eeg_augment=None, aud_augment=None):
    """Extract frozen backbone features for one subject pair.
    Returns (z_eeg [K, eeg_dim], z_audio [K, aud_dim]).
    """
    we = eeg_wins.copy()
    wa = aud_wins.copy()
    K = min(len(we), len(wa))
    we, wa = we[:K], wa[:K]

    # Normalize
    we = np.array([_zscore(we[i]) for i in range(K)])
    wa = np.array([_zscore(wa[i]) for i in range(K)])

    # Augment if configured (applied per-epoch during training)
    if eeg_augment is not None:
        we = np.array([eeg_augment(we[i]) for i in range(K)])
    if aud_augment is not None:
        wa = np.array([aud_augment(wa[i]) for i in range(K)])

    ze = _encode_eeg(eeg_model, we, device)   # [K, eeg_dim]
    za = _encode_audio(aud_model, wa, device)  # [K, aud_dim]
    return ze, za

def extract_all_features(eeg_model, aud_model, subj_pairs, eeg_subjs, aud_subjs,
                          max_windows, eeg_augment=None, aud_augment=None):
    """Extract features for all subjects. Returns padded arrays."""
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

# ── Fusion head training ──────────────────────────────────────────────────

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
    # Balanced loss: pos_weight = n_HC / n_MDD (0 label = HC, 1 label = MDD)
    n_mdd = y_tr.sum()
    n_hc = len(y_tr) - n_mdd
    pos_weight = torch.tensor([n_hc / max(n_mdd, 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_vb, best_st, pat = -1.0, None, 0
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

            # Mixup (if enabled)
            if args.mixup_alpha > 0:
                ze, za, yb, _ = mixup_features(ze, za, yb, args.mixup_alpha)

            # Forward with optional window-level logits
            return_window = args.window_aux and model.training
            out = model(ze, za, mask=m, return_window=return_window)
            if return_window:
                logits, win_logits = out
            else:
                logits = out
                win_logits = None

            # Subject-level loss (label smoothing)
            y_smooth = yb * 0.95 + 0.025
            loss = crit(logits, y_smooth)

            # Window-level auxiliary loss
            if win_logits is not None:
                B, K = ze.shape[0], ze.shape[1]
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
            tr_labels.append(yb)
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
            pat = 0
        else:
            pat += 1

        if ep == 1 or pat == 0 or ep % 10 == 0:
            logger.log_epoch(ep, tr_loss, vl_loss, tr_m, vl_m, pat)

        if pat >= args.fusion_patience:
            break

    if best_st is not None:
        model.load_state_dict(best_st)
    return model, best_vb, history

# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_fusion(model, Z_e, Z_a, mask, y_true):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(Z_e).to(device),
                       torch.FloatTensor(Z_a).to(device),
                       mask=torch.FloatTensor(mask).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return y_true, preds, probs

# ── Build subject index helpers ───────────────────────────────────────────

def build_subject_dict(data_list, labels_arr, cods_list):
    """Convert parallel arrays to dict of {subject_id: {windows, label}}."""
    return {cods_list[i]: {'windows': data_list[i], 'label': int(labels_arr[i])}
            for i in range(len(cods_list))}

def build_backbone_dataset(eeg_dict, aud_dict, mapping, train_paired_ids):
    """Build subject list for backbone training from paired+eeg-only+audio-only.
    Returns (eeg_subjects, eeg_labels, eeg_cods, aud_subjects, aud_labels, aud_cods)
    as parallel lists (for train_backbone compatibility).
    """
    paired_eeg_ids = [p[0] for p in train_paired_ids]
    paired_aud_ids = [p[1] for p in train_paired_ids]

    # EEG subjects: paired + EEG-only (not in mapping or not in mapping values)
    mapped_eeg = set(mapping.values())
    eeg_keys = [k for k in eeg_dict.keys()]
    eeg_data_list = []
    eeg_labels_list = []
    eeg_cods_list = []
    for sid in eeg_keys:
        if sid in paired_eeg_ids or sid not in mapped_eeg:
            eeg_data_list.append(eeg_dict[sid]['windows'])
            eeg_labels_list.append(eeg_dict[sid]['label'])
            eeg_cods_list.append(sid)

    # Audio subjects: paired + audio-only
    mapped_aud = set(mapping.keys())
    aud_keys = [k for k in aud_dict.keys()]
    aud_data_list = []
    aud_labels_list = []
    aud_cods_list = []
    for sid in aud_keys:
        if sid in paired_aud_ids or sid not in mapped_aud:
            aud_data_list.append(aud_dict[sid]['windows'])
            aud_labels_list.append(aud_dict[sid]['label'])
            aud_cods_list.append(sid)

    return (eeg_data_list, np.array(eeg_labels_list, dtype=int), eeg_cods_list,
            aud_data_list, np.array(aud_labels_list, dtype=int), aud_cods_list)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Strict cross-modal training (0 leakage)')
    parser.add_argument('--fusion', choices=['concat', 'gating', 'cross_attn'],
                        default='cross_attn')
    parser.add_argument('--n-self-attn-layers', type=int, default=1)
    parser.add_argument('--self-attn-heads', type=int, default=4)
    parser.add_argument('--self-attn-dropout', type=float, default=0.1)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=1)
    parser.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--bottleneck-dim', type=int, default=None)
    parser.add_argument('--max-windows', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for backbone training')
    parser.add_argument('--wd', type=float, default=1e-3,
                        help='Weight decay for backbone training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs for backbone training')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for backbone training')
    parser.add_argument('--lr-fusion', type=float, default=5e-4,
                        help='Learning rate for fusion head')
    parser.add_argument('--wd-fusion', type=float, default=1e-3,
                        help='Weight decay for fusion head')
    parser.add_argument('--fusion-epochs', type=int, default=100,
                        help='Max epochs for fusion head')
    parser.add_argument('--fusion-patience', type=int, default=15,
                        help='Patience for fusion head')
    parser.add_argument('--bs', type=int, default=8,
                        help='Batch size for fusion head')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation during fusion training')
    parser.add_argument('--noise-std', type=float, default=0.05,
                        help='Gaussian noise std for augmentation')
    parser.add_argument('--time-mask-max', type=int, default=20,
                        help='Max time mask length for EEG augmentation')
    parser.add_argument('--channel-drop-ratio', type=float, default=0.15,
                        help='Ratio of channels to drop for EEG augmentation')
    parser.add_argument('--save-model', action='store_true',
                        help='Save fusion head checkpoints per fold')
    parser.add_argument('--adapter-dim', type=int, default=None,
                        help='Bottleneck dim for LoRA-style adapter (None = disable)')
    parser.add_argument('--window-aux', action='store_true',
                        help='Auxiliary window-level classifier (multi-task)')
    parser.add_argument('--window-aux-weight', type=float, default=0.3,
                         help='Weight for window-level auxiliary loss')
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                         help='Mixup alpha (0 = disable)')
    parser.add_argument('--feat-dropout', type=float, default=0.0,
                         help='Dropout rate on projected hidden features')
    parser.add_argument('--loocv', action='store_true',
                         help='Use LOOCV (38 folds, one per paired subject) instead of 5-fold')
    parser.add_argument('--joint-ft', action='store_true',
                         help='Joint fine-tune: unfreeze backbones and train end-to-end after frozen phase')
    parser.add_argument('--joint-lr', type=float, default=1e-5,
                         help='Learning rate for joint fine-tune phase')
    parser.add_argument('--joint-epochs', type=int, default=50,
                         help='Max epochs for joint fine-tune phase')
    parser.add_argument('--joint-patience', type=int, default=10,
                         help='Patience for joint fine-tune phase')
    args = parser.parse_args()

    cfg_name = f'{args.fusion}'
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
    if args.joint_ft:
        cfg_name += '_joint'
    cfg_name += f'_w{args.max_windows}'

    out_dir = os.path.join(OUTPUT_DIR, cfg_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Device: {device}')
    print(f'Strict CrossModal — {cfg_name}')
    print(f'  Fusion={args.fusion}  Self-attn={args.n_self_attn_layers}L')
    print(f'  Augment={args.augment}  Max windows={args.max_windows}')
    print(f'  Adapter dim={args.adapter_dim}  Window aux={args.window_aux}  Mixup alpha={args.mixup_alpha}')
    print(f'  Feat dropout={args.feat_dropout}  LOOCV={args.loocv}')
    print(f'  Backbone: lr={args.lr} wd={args.wd} epochs={args.epochs}')
    print(f'  Fusion:   lr={args.lr_fusion} wd={args.wd_fusion} epochs={args.fusion_epochs}')

    # Load data
    eeg_data, eeg_labels, eeg_cods = _load_eeg_cache()
    aud_data, aud_labels, aud_cods = _load_audio_cache()
    mapping = _load_mapping()
    eeg_dict = build_subject_dict(eeg_data, eeg_labels, eeg_cods)
    aud_dict = build_subject_dict(aud_data, aud_labels, aud_cods)

    # Build paired list
    pairs = []
    for aud_id, eeg_id in mapping.items():
        if eeg_id in eeg_dict and aud_id in aud_dict:
            pairs.append((eeg_id, aud_id, eeg_dict[eeg_id]['label']))
    labels = np.array([p[2] for p in pairs])
    group_ids = np.array([f'p{i}' for i in range(len(pairs))])
    print(f'  Paired subjects: {len(pairs)} ({int(labels.sum())} MDD, {len(pairs)-int(labels.sum())} HC)')
    print(f'  CV: {"LOOCV (38 folds)" if args.loocv else "5-fold"}')

    # External CV on paired subjects
    if args.loocv:
        splitter = LeaveOneGroupOut()
    else:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    fold_mapping = {}

    for fi, (tvi, tei) in enumerate(splitter.split(np.zeros(len(pairs)), labels, groups=group_ids)):
        print(f'\n{"="*60}')
        print(f'  Fold {fi + 1}')
        print(f'{"="*60}')
        try:
            # Split paired subjects
            tr_paired = [pairs[i] for i in tvi]  # ~31 paired
            te_paired = [pairs[i] for i in tei]   # ~7 paired
            fold_mapping[f'fold_{fi+1}'] = [p[0] for p in te_paired]

            # Build backbone dataset: ~31 paired + all EEG-only + all audio-only
            eeg_bb_data, eeg_bb_labels, eeg_bb_cods, \
                aud_bb_data, aud_bb_labels, aud_bb_cods = \
                build_backbone_dataset(eeg_dict, aud_dict, mapping, tr_paired)

            print('  Backbone training set:')
            print(f'    EEG: {len(eeg_bb_cods)} subj ({int(eeg_bb_labels.sum())} MDD, '
                  f'{int((1-eeg_bb_labels).sum())} HC)')
            print(f'    Audio: {len(aud_bb_cods)} subj ({int(aud_bb_labels.sum())} MDD, '
                  f'{int((1-aud_bb_labels).sum())} HC)')
            print(f'    Paired train: {len(tr_paired)}  Test: {len(te_paired)}')

            # Inner split for backbone early stopping
            inner_seed = RANDOM_STATE + fi
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=inner_seed)

            # ── Train EEG backbone ──
            print('\n  --- Training EEG backbone ---')
            eeg_tr_i, eeg_vl_i = next(inner.split(
                np.zeros(len(eeg_bb_labels)), eeg_bb_labels, groups=eeg_bb_cods))
            tr_ds = WindowDataset(eeg_bb_data, eeg_bb_labels, eeg_bb_cods,
                                  [eeg_tr_i[i] for i in range(len(eeg_tr_i))],
                                  max_windows=args.max_windows)
            vl_ds = WindowDataset(eeg_bb_data, eeg_bb_labels, eeg_bb_cods,
                                  [eeg_vl_i[i] for i in range(len(eeg_vl_i))],
                                  max_windows=args.max_windows)
            tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
            vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False)

            eeg_model = DeepConvNetWrapper(64, N_EEG_SAMPLES).to(device)
            if fi == 0:
                print(f'    EEG params: {sum(p.numel() for p in eeg_model.parameters()):,}')
            eeg_tr_cods = [eeg_bb_cods[i] for i in eeg_tr_i]
            eeg_vl_cods = [eeg_bb_cods[i] for i in eeg_vl_i]
            eeg_best_st, eeg_best_vb, eeg_history = train_backbone(eeg_model, tr_ldr, vl_ldr, args)
            eeg_model.load_state_dict(eeg_best_st)
            eeg_model.eval()
            print(f'    EEG backbone best val bacc: {eeg_best_vb:.3f}')
            del tr_ds, vl_ds, tr_ldr, vl_ldr

            # ── Train audio backbone ──
            print('\n  --- Training Audio backbone ---')
            aud_tr_i, aud_vl_i = next(inner.split(
                np.zeros(len(aud_bb_labels)), aud_bb_labels, groups=aud_bb_cods))
            tr_ds = WindowDataset(aud_bb_data, aud_bb_labels, aud_bb_cods,
                                  [aud_tr_i[i] for i in range(len(aud_tr_i))],
                                  max_windows=args.max_windows)
            vl_ds = WindowDataset(aud_bb_data, aud_bb_labels, aud_bb_cods,
                                  [aud_vl_i[i] for i in range(len(aud_vl_i))],
                                  max_windows=args.max_windows)
            tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
            vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False)

            aud_model = ShallowConvNetWrapper(N_MELS, N_AUDIO_SAMPLES).to(device)
            if fi == 0:
                print(f'    Audio params: {sum(p.numel() for p in aud_model.parameters()):,}')
            aud_tr_cods = [aud_bb_cods[i] for i in aud_tr_i]
            aud_vl_cods = [aud_bb_cods[i] for i in aud_vl_i]
            aud_best_st, aud_best_vb, aud_history = train_backbone(aud_model, tr_ldr, vl_ldr, args)
            aud_model.load_state_dict(aud_best_st)
            aud_model.eval()
            print(f'    Audio backbone best val bacc: {aud_best_vb:.3f}')
            del tr_ds, vl_ds, tr_ldr, vl_ldr

            # ── Extract features for fusion ──
            print('\n  --- Extracting features ---')

            # Setup augmentation
            eeg_aug = EEGAugment(noise_std=args.noise_std,
                                 time_mask_max=args.time_mask_max,
                                 channel_drop_ratio=args.channel_drop_ratio) if args.augment else None
            aud_aug = AudioAugment(noise_std=args.noise_std) if args.augment else None

            # Extract training features (paired subjects only)
            tr_paired_list = tr_paired
            Z_e_tr, Z_a_tr, mask_tr = extract_all_features(
                eeg_model, aud_model, tr_paired_list, eeg_dict, aud_dict,
                args.max_windows, eeg_augment=eeg_aug, aud_augment=aud_aug)

            # Extract test features (NO augmentation)
            Z_e_te, Z_a_te, mask_te = extract_all_features(
                eeg_model, aud_model, te_paired, eeg_dict, aud_dict,
                args.max_windows)

            y_tr = np.array([p[2] for p in tr_paired_list], dtype=np.float32)
            y_te = np.array([p[2] for p in te_paired], dtype=np.float32)

            eeg_dim = Z_e_tr.shape[2]
            aud_dim = Z_a_tr.shape[2]
            print(f'    EEG feat dim={eeg_dim}  Audio feat dim={aud_dim}')
            print(f'    Train subjects={len(tr_paired_list)}  Test subjects={len(te_paired)}')

            # Inner split for fusion head validation
            inner_fusion = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                                random_state=RANDOM_STATE + fi)
            fuse_tr_i, fuse_vl_i = next(inner_fusion.split(
                np.zeros(len(tr_paired_list)), y_tr,
                groups=[f't{i}' for i in range(len(tr_paired_list))]))
            fuse_tr_cods = [tr_paired_list[i][0] for i in fuse_tr_i]
            fuse_vl_cods = [tr_paired_list[i][0] for i in fuse_vl_i]

            # ── Train fusion head ──
            print('\n  --- Training fusion head ---')
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

            if fi == 0:
                print(f'    Fusion params: {sum(p.numel() for p in fusion_model.parameters()):,}')

            fusion_model, best_val_bacc, fusion_history = train_fusion_head(
                fusion_model,
                Z_e_tr[fuse_tr_i], Z_a_tr[fuse_tr_i], mask_tr[fuse_tr_i], y_tr[fuse_tr_i],
                Z_e_tr[fuse_vl_i], Z_a_tr[fuse_vl_i], mask_tr[fuse_vl_i], y_tr[fuse_vl_i],
                args)

            # ── Joint fine-tune (optional) ──
            joint_history = None
            if args.joint_ft:
                print('\n  --- Joint fine-tune (backbones unfrozen) ---')
                joint_model = JointModel(eeg_model, aud_model, fusion_model).to(device)
                joint_model.load_state_dict(joint_model.state_dict(), strict=False)

                tr_jds = JointWindowDataset(eeg_dict, aud_dict, tr_paired_list,
                                             args.max_windows, indices=fuse_tr_i)
                vl_jds = JointWindowDataset(eeg_dict, aud_dict, tr_paired_list,
                                             args.max_windows, indices=fuse_vl_i)
                tr_jl = DataLoader(tr_jds, batch_size=args.bs, shuffle=True)
                vl_jl = DataLoader(vl_jds, batch_size=args.bs, shuffle=False)

                joint_model, joint_best_vb, joint_history = train_joint(
                    joint_model, tr_jl, vl_jl, args)

                # Update fusion_model with joint-fine-tuned weights
                fusion_model.load_state_dict(
                    {k.replace('fusion.', ''): v for k, v in joint_model.state_dict().items()
                     if k.startswith('fusion.')}, strict=False)
                # Update backbones in-place (they'll be used for re-extraction)
                eeg_model.load_state_dict(
                    {k.replace('eeg_backbone.', ''): v for k, v in joint_model.state_dict().items()
                     if k.startswith('eeg_backbone.')}, strict=False)
                aud_model.load_state_dict(
                    {k.replace('aud_backbone.', ''): v for k, v in joint_model.state_dict().items()
                     if k.startswith('aud_backbone.')}, strict=False)

                print(f'    Joint fine-tune best val bacc: {joint_best_vb:.3f}')

                # Re-extract features with fine-tuned backbones
                print('  --- Re-extracting features with fine-tuned backbones ---')
                tr_paired_list = tr_paired
                Z_e_tr, Z_a_tr, mask_tr = extract_all_features(
                    eeg_model, aud_model, tr_paired_list, eeg_dict, aud_dict,
                    args.max_windows, eeg_augment=eeg_aug, aud_augment=aud_aug)

                Z_e_te, Z_a_te, mask_te = extract_all_features(
                    eeg_model, aud_model, te_paired, eeg_dict, aud_dict,
                    args.max_windows)

                # Update val_bacc tracking
                best_val_bacc = joint_best_vb

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

            # Find best val epoch metrics from fusion_history
            best_val_acc = float(fusion_history['val_acc'][np.argmax(fusion_history['val_bacc'])]) if fusion_history['val_bacc'] else 0.0
            best_val_f1 = float(fusion_history['val_f1'][np.argmax(fusion_history['val_bacc'])]) if fusion_history['val_f1'] else 0.0
            best_val_sens = float(fusion_history['val_sens'][np.argmax(fusion_history['val_bacc'])]) if fusion_history['val_sens'] else 0.0
            best_val_spec = float(fusion_history['val_spec'][np.argmax(fusion_history['val_bacc'])]) if fusion_history['val_spec'] else 0.0
            best_val_auc = float(best_val_bacc)  # placeholder, no AUC computed during fusion val

            fold_results.append({
                'fold': fi + 1,
                'best_val_bacc': float(best_val_bacc),
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1,
                'best_val_sens': best_val_sens,
                'best_val_spec': best_val_spec,
                'best_val_auc': best_val_auc,
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
                'n_train_paired': len(tr_paired_list),
                'n_test': len(te_paired),
                'test_subjects': [p[0] for p in te_paired],
                'train_paired_subjects': [p[0] for p in tr_paired_list],
                'eeg_backbone_train_cods': eeg_tr_cods,
                'eeg_backbone_val_cods': eeg_vl_cods,
                'aud_backbone_train_cods': aud_tr_cods,
                'aud_backbone_val_cods': aud_vl_cods,
                'fusion_train_subjects': fuse_tr_cods,
                'fusion_val_subjects': fuse_vl_cods,
                'eeg_history': eeg_history,
                'aud_history': aud_history,
                'fusion_history': fusion_history,
                'joint_history': joint_history,
            })
            print(f'  Fold {fi + 1}: best_val={best_val_bacc:.3f}  '
                  f'test_bacc={bacc:.3f}  test_auc={roc_auc:.3f}')

            # Save model checkpoints
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

            # Cleanup
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
        val_baccs = [r['best_val_bacc'] for r in fold_results]

        # Per-fold validation metrics (best_val_* from fold data)
        val_accs = []
        val_f1s = []
        val_sens = []
        val_spec = []
        for r in fold_results:
            h = r['fusion_history']
            # find epoch index where val_bacc was best
            if h:
                best_idx = int(np.argmax(h['val_bacc']))
                val_accs.append(h['val_acc'][best_idx])
                val_f1s.append(h['val_f1'][best_idx])
                val_sens.append(h['val_sens'][best_idx])
                val_spec.append(h['val_spec'][best_idx])

        # Test metrics
        test_accs = [r['test_metrics']['acc'] for r in fold_results]
        test_f1s = [r['test_metrics']['f1'] for r in fold_results]
        test_sens = [r['test_metrics']['sens'] for r in fold_results]
        test_spec = [r['test_metrics']['spec'] for r in fold_results]

        def _mean_std(v):
            return float(np.mean(v)), float(np.std(v))

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

        validation = {}
        if val_accs:
            validation = {
                'bacc_mean': float(np.mean(val_baccs)),
                'bacc_std': float(np.std(val_baccs)),
                'acc_mean': float(np.mean(val_accs)),
                'acc_std': float(np.std(val_accs)),
                'f1_mean': float(np.mean(val_f1s)),
                'f1_std': float(np.std(val_f1s)),
                'sens_mean': float(np.mean(val_sens)),
                'sens_std': float(np.std(val_sens)),
                'spec_mean': float(np.mean(val_spec)),
                'spec_std': float(np.std(val_spec)),
            }

        # Git commit
        git_commit = ''
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            git_commit = 'unknown'

        summary = {
            'bacc_mean': float(np.mean(baccs)),
            'bacc_std': float(np.std(baccs)),
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
            'val_bacc_mean': float(np.mean(val_baccs)),
            'val_bacc_std': float(np.std(val_baccs)),
        }

        print(f'\n{"="*55}')
        print(f'  {cfg_name}')
        print(f'  bacc = {summary["bacc_mean"]:.3f} ± {summary["bacc_std"]:.3f}')
        print(f'  auc  = {summary["auc_mean"]:.3f} ± {summary["auc_std"]:.3f}')
        print(f'{"="*55}')

        out_results = {
            'experiment': {
                'name': 'crossmodal_strict',
                'fusion': args.fusion,
                'script': 'train_crossmodal_strict.py',
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
                'adapter_dim': args.adapter_dim,
                'window_aux': args.window_aux,
                'window_aux_weight': args.window_aux_weight,
                'mixup_alpha': args.mixup_alpha,
                'feat_dropout': args.feat_dropout,
                'loocv': args.loocv,
                'joint_ft': args.joint_ft,
                'joint_lr': args.joint_lr,
                'joint_epochs': args.joint_epochs,
                'joint_patience': args.joint_patience,
            },
            'validation': validation,
            'test': test,
            'folds': fold_results,
            'final_model': None,
            'summary': summary,
            'fold_subject_split': fold_mapping,
        }
        out_path = os.path.join(out_dir, 'results.json')
        with open(out_path, 'w') as f:
            json.dump(out_results, f, indent=2)
        print(f'Saved: {out_path}')
    else:
        print('\nAll folds failed — no results saved')


if __name__ == '__main__':
    main()

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.deepconvnet import DeepConvNet
from models.shallowconvnet import ShallowConvNet
from models.crossmodal_attn import CrossModalAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_ROOT = os.path.join('outputs', 'results', 'crossmodal_nested')
FIGURES_ROOT = os.path.join('outputs', 'figures')

EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'

# ── Data loading (standalone, no dep on train_crossmodal_sngkf) ──

def load_eeg_cache(path=EEG_CACHE):
    c = np.load(path, allow_pickle=True)
    data = list(c['windows'])
    labels = c['labels'].astype(int)
    cods = list(c['subject_ids'])
    n_samples = data[0].shape[2]
    n_ch = data[0].shape[1]
    print(f'  EEG: {len(cods)} subj ({int(labels.sum())} MDD, {int((1-labels).sum())} HC), '
          f'windows: {n_ch}ch x {n_samples}')
    return data, labels, cods

def load_audio_cache(path=AUDIO_CACHE):
    c = np.load(path, allow_pickle=True)
    data = list(c['windows'])
    labels = c['labels'].astype(int)
    cods = [str(s) for s in c['subject_ids']]
    print(f'  Audio: {len(cods)} subj ({int(labels.sum())} MDD, {int((1-labels).sum())} HC)')
    return data, labels, cods

def load_mapping(path=MAPPING_PATH):
    with open(path) as f:
        return json.load(f)['orig_to_bids']

def select_windows(windows, max_windows):
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    indices = np.linspace(0, n - 1, max_windows, dtype=int)
    return windows[indices]

# ── Model loading ──

class DeepConvNetWrapper(nn.Module):
    def __init__(self, n_channels=64, n_samples=500):
        super().__init__()
        self.m = DeepConvNet(n_channels, 1, n_samples, dropout=0.0)
    def forward(self, x):
        return self.m(x).squeeze(-1)
    def forward_features(self, x):
        return self.m.forward_features(x)

class ShallowConvNetWrapper(nn.Module):
    def __init__(self, n_channels=64, n_samples=200):
        super().__init__()
        self.m = ShallowConvNet(n_channels, 1, n_samples, dropout=0.0)
    def forward(self, x):
        return self.m(x).squeeze(-1)
    def forward_features(self, x):
        return self.m.forward_features(x)

def encode_eeg(model, windows):
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.FloatTensor(windows[i:i+32]).to(device)
        with torch.no_grad():
            feats.append(model.forward_features(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)

def encode_audio(model, windows):
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.FloatTensor(windows[i:i+32]).to(device)
        with torch.no_grad():
            feats.append(model.forward_features(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)

def extract_subj_features(eeg_model, aud_model, eeg_wins, aud_wins):
    K = min(len(eeg_wins), len(aud_wins))
    eeg_wins, aud_wins = eeg_wins[:K], aud_wins[:K]
    mu_e, sg_e = eeg_wins.mean(), eeg_wins.std() + 1e-8
    mu_a, sg_a = aud_wins.mean(), aud_wins.std() + 1e-8
    eeg_wins = (eeg_wins - mu_e) / sg_e
    aud_wins = (aud_wins - mu_a) / sg_a
    ze = encode_eeg(eeg_model, eeg_wins)
    za = encode_audio(aud_model, aud_wins)
    return ze, za

def extract_all_features(eeg_model, aud_model, pairs, eeg_subjs, aud_subjs, max_windows=200):
    all_ze, all_za, all_masks = [], [], []
    for eid, aid, _ in pairs:
        we = select_windows(eeg_subjs[eid]['windows'], max_windows)
        wa = select_windows(aud_subjs[aid]['windows'], max_windows)
        ze, za = extract_subj_features(eeg_model, aud_model, we, wa)
        K = len(ze)
        all_ze.append(ze)
        all_za.append(za)
        all_masks.append(np.ones(K, dtype=np.float32))
    max_K = max(m.shape[0] for m in all_masks)
    eeg_dim = all_ze[0].shape[1]
    aud_dim = all_za[0].shape[1]
    Z_e = np.zeros((len(pairs), max_K, eeg_dim), dtype=np.float32)
    Z_a = np.zeros((len(pairs), max_K, aud_dim), dtype=np.float32)
    masks = np.zeros((len(pairs), max_K), dtype=np.float32)
    for i in range(len(pairs)):
        k = len(all_ze[i])
        Z_e[i, :k] = all_ze[i]
        Z_a[i, :k] = all_za[i]
        masks[i, :k] = all_masks[i]
    return Z_e, Z_a, masks

# ── Checkpoint / experiment helpers ──

def find_checkpoint_dir(tag, seed):
    pattern = os.path.join(RESULTS_ROOT, f'mhcmattention_sngkf_seed{seed}_iseed_*_outerf5_innerf5_tag{tag}')
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f'No checkpoint dir for seed={seed}, tag={tag}')
    return dirs[-1]

def load_checkpoint(tag, seed, fold):
    ckpt_dir = find_checkpoint_dir(tag, seed)
    ckpt_path = os.path.join(ckpt_dir, 'checkpoints', f'fold_{fold}.pt')
    return torch.load(ckpt_path, map_location='cpu', weights_only=False)

def build_paired_subjects(eeg_data, eeg_labels, eeg_cods,
                          aud_data, aud_labels, aud_cods, mapping):
    rev = {v: k for k, v in mapping.items()}
    pairs, eeg_subjs, aud_subjs = [], {}, {}
    for i, cod in enumerate(eeg_cods):
        eeg_subjs[cod] = {'label': int(eeg_labels[i]), 'windows': eeg_data[i]}
        aid = rev.get(cod)
        if aid in aud_cods:
            pairs.append([cod, aid, int(eeg_labels[i])])
    for i, cod in enumerate(aud_cods):
        aud_subjs[cod] = {'label': int(aud_labels[i]), 'windows': aud_data[i]}
    return pairs, eeg_subjs, aud_subjs

def get_feature_dims():
    eeg_dummy = torch.randn(1, 64, 500).to(device)
    aud_dummy = torch.randn(1, 64, 200).to(device)
    eeg_m = DeepConvNetWrapper(64, 500).to(device)
    aud_m = ShallowConvNetWrapper(64, 200).to(device)
    eeg_m.eval()
    aud_m.eval()
    with torch.no_grad():
        eeg_dim = eeg_m.forward_features(eeg_dummy).shape[1]
        aud_dim = aud_m.forward_features(aud_dummy).shape[1]
    return eeg_dim, aud_dim

def build_models(ckpt):
    eeg_model = DeepConvNetWrapper(64, 500).to(device)
    aud_model = ShallowConvNetWrapper(64, 200).to(device)
    eeg_model.load_state_dict(ckpt['eeg_backbone_state'])
    aud_model.load_state_dict(ckpt['aud_backbone_state'])
    eeg_model.eval()
    aud_model.eval()

    eeg_dim, aud_dim = get_feature_dims()
    args = ckpt.get('args', {})
    fusion_model = CrossModalAttention(
        eeg_dim=eeg_dim,
        aud_dim=aud_dim,
        hidden=args.get('hidden', 64),
        n_heads=args.get('n_heads', 2),
        fusion=args.get('fusion', 'cross_attn'),
        n_self_attn_layers=args.get('n_self_attn_layers', 0),
        self_attn_heads=args.get('self_attn_heads', 4),
        self_attn_dropout=args.get('self_attn_dropout', 0.1),
        pooling=args.get('pooling', 'mean'),
        dropout=args.get('dropout', 0.5),
        bottleneck_dim=args.get('bottleneck_dim', None),
        adapter_dim=args.get('adapter_dim', None),
        window_aux=args.get('window_aux', False),
        feat_dropout=args.get('feat_dropout', 0.0),
    ).to(device)
    fusion_model.load_state_dict(ckpt['fusion_state_dict'])
    fusion_model.eval()
    return eeg_model, aud_model, fusion_model

def parse_shared_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--tag', required=True, help='Experiment tag (e.g. bbvalfix_d07_lr5e4_6seeds)')
    p.add_argument('--seed', type=int, default=42, help='Seed to analyze')
    p.add_argument('--fold', type=int, default=1, help='Fold to analyze')
    p.add_argument('--save', action='store_true', help='Save figure instead of plt.show()')
    return p.parse_args()

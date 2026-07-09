"""
Train final model on 100% of available paired data (no hold-out).

Reads the latest results.json from crossmodal_strict to get the configuration,
then trains backbones on all EEG + all audio + all paired subjects (same as
backbone training set but without test split), extracts features, trains the
fusion head on ALL paired subjects, and saves the final model.

Usage:
  py scripts/final_model.py
  py scripts/final_model.py --path outputs/results/crossmodal_strict/cross_attn_w200/results.json
  py scripts/final_model.py --save-model --output-dir outputs/models

Output:
  - outputs/models/final_model.pt
  - Adds 'final_model' section to the source results.json
"""
import sys
import json
import copy
import argparse
import warnings
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, accuracy_score, f1_score
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.models.crossmodal_attn import CrossModalAttention  # noqa: E402
from src.models.deepconvnet import DeepConvNet  # noqa: E402
from src.models.shallowconvnet import ShallowConvNet  # noqa: E402

# ── Backbone wrappers ──
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

# ── Data paths ──
EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_MODEL_DIR = Path('outputs/models')
RANDOM_STATE = 42
N_MELS = 64
N_AUDIO_SAMPLES = 200
N_EEG_SAMPLES = 500


def _load_eeg_cache(path=EEG_CACHE):
    c = np.load(path, allow_pickle=True)
    return list(c['windows']), c['labels'].astype(int), list(c['subject_ids'])

def _load_audio_cache(path=AUDIO_CACHE):
    c = np.load(path, allow_pickle=True)
    return list(c['windows']), c['labels'].astype(int), [str(s) for s in c['subject_ids']]

def _load_mapping(path=MAPPING_PATH):
    with open(path) as f:
        return json.load(f)['orig_to_bids']

def _zscore(w):
    return (w - w.mean()) / (w.std() + 1e-8)

def _select_windows_deterministic(windows, max_windows):
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    indices = np.linspace(0, n - 1, max_windows, dtype=int)
    return windows[indices]


class WindowDataset(Dataset):
    def __init__(self, windows_list, labels_list, subj_names, indices, max_windows=None):
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
        return torch.from_numpy(w).float(), torch.tensor(label, dtype=torch.float)


def _logits_to_binary(logits):
    return (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)


def train_backbone(model, train_loader, val_loader, epochs=100, lr=5e-4, wd=1e-3, patience=15):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, foreach=False)
    crit = nn.BCEWithLogitsLoss()
    best_vb, best_st, pat = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        tr_logits, tr_labels = [], []
        for X, y in train_loader:
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

        model.eval()
        vl_loss, vl_n = 0.0, 0
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).float()
                logits = model(X)
                vl_loss += crit(logits, y).item() * X.size(0)
                vl_n += X.size(0)
                vl_logits.append(logits)
                vl_labels.append(y)
        vl_loss /= vl_n
        vl_pred = _logits_to_binary(torch.cat(vl_logits))
        vl_true = torch.cat(vl_labels).cpu().numpy()
        from sklearn.metrics import balanced_accuracy_score
        vb = balanced_accuracy_score(vl_true, vl_pred)

        if vb > best_vb:
            best_vb = vb
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break
    if best_st is None:
        best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_st)
    return model


def _encode_eeg(model, windows):
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

def _encode_audio(model, windows):
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


def train_fusion_head(model, Z_e, Z_a, mask, y, epochs=100, lr=5e-4, wd=1e-3, patience=15):
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(Z_e), torch.FloatTensor(Z_a),
        torch.FloatTensor(mask), torch.FloatTensor(y))
    ldr = DataLoader(ds, batch_size=len(y), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, foreach=False)
    crit = nn.BCEWithLogitsLoss()
    best_bacc, best_st, pat = -1.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        for ze, za, m, yb in ldr:
            ze, za, m, yb = ze.to(device), za.to(device), m.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(ze, za, mask=m)
            y_smooth = yb * 0.95 + 0.025
            loss = crit(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            all_logits = model(torch.FloatTensor(Z_e).to(device),
                                torch.FloatTensor(Z_a).to(device),
                                mask=torch.FloatTensor(mask).to(device))
        preds = (torch.sigmoid(all_logits).cpu().numpy() >= 0.5).astype(int)
        bacc = balanced_accuracy_score(y, preds)

        if bacc > best_bacc:
            best_bacc = bacc
            best_st = copy.deepcopy(model.state_dict())
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break

    if best_st is not None:
        model.load_state_dict(best_st)
    return model


def main():
    parser = argparse.ArgumentParser(description='Train final model on 100% data')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to results.json to read config (default: latest)')
    parser.add_argument('--output-dir', type=str, default='outputs/models',
                        help='Directory to save final_model.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    # Find results.json
    if args.path:
        results_path = Path(args.path)
    else:
        results_root = Path('outputs/results')
        candidates = sorted(results_root.rglob('results.json'),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        results_path = candidates[0] if candidates else None

    if results_path is None or not results_path.exists():
        print('No results.json found. Run train_crossmodal_strict.py first.')
        sys.exit(1)

    with open(results_path, 'r') as f:
        data = json.load(f)

    cfg = data.get('config', {})
    if not cfg:
        cfg = data.get('args', {})
    if isinstance(cfg, dict) and 'fusion' not in cfg:
        cfg = vars(argparse.Namespace(**cfg)) if isinstance(cfg, dict) else {}

    max_windows = cfg.get('max_windows', 50)
    fusion_type = cfg.get('fusion', 'cross_attn')
    hidden = cfg.get('hidden', 64)
    n_heads = cfg.get('n_heads', 1)
    dropout = cfg.get('dropout', 0.5)
    bottleneck_dim = cfg.get('bottleneck_dim', None)
    n_self_attn = cfg.get('n_self_attn_layers', 1)
    self_attn_heads = cfg.get('self_attn_heads', 4)
    self_attn_dropout = cfg.get('self_attn_dropout', 0.1)
    pooling = cfg.get('pooling', 'mean')

    print('Training final model on 100% data')
    print(f'  Fusion={fusion_type} hidden={hidden} heads={n_heads} max_windows={max_windows}')

    # Load data
    eeg_data, eeg_labels, eeg_cods = _load_eeg_cache()
    aud_data, aud_labels, aud_cods = _load_audio_cache()
    mapping = _load_mapping()

    eeg_dict = {eeg_cods[i]: {'windows': eeg_data[i], 'label': int(eeg_labels[i])}
                 for i in range(len(eeg_cods))}
    aud_dict = {aud_cods[i]: {'windows': aud_data[i], 'label': int(aud_labels[i])}
                 for i in range(len(aud_cods))}

    pairs = []
    for aud_id, eeg_id in mapping.items():
        if eeg_id in eeg_dict and aud_id in aud_dict:
            pairs.append((eeg_id, aud_id, eeg_dict[eeg_id]['label']))
    labels = np.array([p[2] for p in pairs])
    print(f'  Paired: {len(pairs)} ({int(labels.sum())} MDD, {len(pairs) - int(labels.sum())} HC)')

    # ALL subjects for backbone training
    mapped_eeg = set(mapping.values())
    mapped_aud = set(mapping.keys())

    eeg_indices = [i for i, sid in enumerate(eeg_cods)
                   if sid in [p[0] for p in pairs] or sid not in mapped_eeg]
    aud_indices = [i for i, sid in enumerate(aud_cods)
                   if sid in [p[1] for p in pairs] or sid not in mapped_aud]

    # Train EEG backbone on ALL eligible subjects
    print('\n--- Training EEG backbone (100% subjects) ---')
    tr_ds = WindowDataset(eeg_data, eeg_labels, eeg_cods, eeg_indices, max_windows=max_windows)
    vl_ds = WindowDataset(eeg_data, eeg_labels, eeg_cods, eeg_indices, max_windows=max_windows)
    tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
    vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False)
    eeg_model = DeepConvNetWrapper(64, N_EEG_SAMPLES).to(device)
    train_backbone(eeg_model, tr_ldr, vl_ldr, epochs=args.epochs, patience=args.patience)
    eeg_model.eval()

    # Train audio backbone on ALL eligible subjects
    print('\n--- Training Audio backbone (100% subjects) ---')
    tr_ds = WindowDataset(aud_data, aud_labels, aud_cods, aud_indices, max_windows=max_windows)
    vl_ds = WindowDataset(aud_data, aud_labels, aud_cods, aud_indices, max_windows=max_windows)
    tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)
    vl_ldr = DataLoader(vl_ds, batch_size=32, shuffle=False)
    aud_model = ShallowConvNetWrapper(N_MELS, N_AUDIO_SAMPLES).to(device)
    train_backbone(aud_model, tr_ldr, vl_ldr, epochs=args.epochs, patience=args.patience)
    aud_model.eval()

    # Extract features for ALL paired subjects
    print('\n--- Extracting features ---')
    all_ze, all_za, all_masks = [], [], []
    for eid, aid, _ in pairs:
        we = _select_windows_deterministic(eeg_dict[eid]['windows'], max_windows)
        wa = _select_windows_deterministic(aud_dict[aid]['windows'], max_windows)
        K = min(len(we), len(wa))
        we, wa = we[:K], wa[:K]
        we = np.array([_zscore(we[i]) for i in range(K)])
        wa = np.array([_zscore(wa[i]) for i in range(K)])
        ze = _encode_eeg(eeg_model, we)
        za = _encode_audio(aud_model, wa)
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

    # Train fusion head on ALL paired subjects
    print('\n--- Training Fusion Head (100% subjects) ---')
    fusion_model = CrossModalAttention(
        eeg_dim=eeg_dim, aud_dim=aud_dim,
        hidden=hidden, n_heads=n_heads,
        bottleneck_dim=bottleneck_dim,
        n_self_attn_layers=n_self_attn,
        self_attn_heads=self_attn_heads,
        self_attn_dropout=self_attn_dropout,
        fusion=fusion_type, pooling=pooling, dropout=dropout,
    ).to(device)
    print(f'  Fusion params: {sum(p.numel() for p in fusion_model.parameters()):,}')
    fusion_model = train_fusion_head(fusion_model, Z_e, Z_a, masks, labels,
                                       epochs=args.epochs, patience=args.patience)

    # Evaluate on training data (descriptive only)
    print('\n--- Evaluation (descriptive, on training data) ---')
    fusion_model.eval()
    with torch.no_grad():
        logits = fusion_model(torch.FloatTensor(Z_e).to(device),
                              torch.FloatTensor(Z_a).to(device),
                              mask=torch.FloatTensor(masks).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_true = labels.astype(int)

    bacc = balanced_accuracy_score(y_true, preds)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except Exception:
        roc_auc = 0.5

    print(f'  BACC={bacc:.3f} ACC={acc:.3f} F1={f1:.3f} AUC={roc_auc:.3f}')
    print(f'  Sens={sens:.3f} Spec={spec:.3f}')
    print(f'  CM={cm.tolist()}')

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'final_model.pt'
    torch.save({
        'fusion_state_dict': fusion_model.state_dict(),
        'eeg_backbone_state': eeg_model.state_dict(),
        'aud_backbone_state': aud_model.state_dict(),
        'config': {
            'fusion': fusion_type,
            'hidden': hidden,
            'n_heads': n_heads,
            'dropout': dropout,
            'max_windows': max_windows,
            'bottleneck_dim': bottleneck_dim,
            'n_self_attn_layers': n_self_attn,
            'self_attn_heads': self_attn_heads,
            'self_attn_dropout': self_attn_dropout,
            'pooling': pooling,
            'eeg_dim': eeg_dim,
            'aud_dim': aud_dim,
        },
        'metrics': {
            'bacc': float(bacc),
            'acc': float(acc),
            'f1': float(f1),
            'sens': float(sens),
            'spec': float(spec),
            'auc': float(roc_auc),
        },
        'cm': cm.tolist(),
        'y_true': y_true.tolist(),
        'y_prob': probs.tolist(),
    }, model_path)
    print(f'\nModel saved: {model_path}')

    # Update results.json with final_model
    final_entry = {
        'bacc': float(bacc),
        'acc': float(acc),
        'f1': float(f1),
        'sens': float(sens),
        'spec': float(spec),
        'auc': float(roc_auc),
        'cm': cm.tolist(),
        'roc': {'y_true': y_true.tolist(), 'y_prob': probs.tolist()},
        'model_path': str(model_path),
    }
    data['final_model'] = final_entry
    with open(results_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Updated: {results_path}')


if __name__ == '__main__':
    main()
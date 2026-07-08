"""
End-to-end training: backbone + fusion trained jointly on CAMPNet folds.
Backbones initialized from Fase 1 checkpoints, fine-tuned with low LR.

Usage:
  py src/training/run_crossmodal_e2e.py [--max-windows 50] [--epochs 100] [--save-model]
"""
import sys, os, json, argparse, copy, warnings
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.models.crossmodal_attn import CrossModalAttention, SelfAttentionBlock
from src.utils.training_logger import ClassificationLogger

EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/crossmodal/e2e'
RANDOM_STATE = 42
N_FOLDS = 5
N_EEG_CH = 64
N_AUDIO_MELS = 64
N_AUDIO_FRAMES = 200
EEG_CKPT_DIR = 'outputs/results/classical_dl/trained_eeg/deepconvnet_64ch'
AUDIO_CKPT_DIR = 'outputs/results/classical_dl/trained_audio/shallowconvnet_64mel'
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ── Data loading ──────────────────────────────────────────────────────────

def _load_cache(npz_path):
    c = np.load(npz_path)
    wins = c['windows']
    labels = c['labels']
    ids = [str(s) for s in c['subject_ids']]
    has_mask = 'window_mask' in c
    subjects = {}
    for i, sid in enumerate(ids):
        if has_mask:
            mask = c['window_mask'][i]
            w = wins[i][mask]
        else:
            w = wins[i]
        subjects[sid] = {'windows': w, 'label': int(labels[i])}
    return subjects, ids, np.array(labels)


def _load_multimodal_pairs(eeg_subjs, aud_subjs):
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    pairs = []
    for aud_id, eeg_id in mapping['orig_to_bids'].items():
        if eeg_id in eeg_subjs and aud_id in aud_subjs:
            pairs.append((eeg_id, aud_id, eeg_subjs[eeg_id]['label']))
    return pairs


def _select_windows_deterministic(windows, max_windows):
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    indices = np.linspace(0, n - 1, max_windows, dtype=int)
    return windows[indices]


def _zscore(w):
    return (w - w.mean()) / (w.std() + 1e-8)


# ── Subject-level Dataset (returns all K windows per subject) ─────────────

class SubjectWindowDataset(Dataset):
    """Returns (eeg_windows, aud_windows, label, mask) for one subject.
       eeg_windows: [K, 1, 64, 500], aud_windows: [K, 1, 64, 200]"""
    def __init__(self, eeg_subjs, aud_subjs, pairs, indices, max_windows):
        self.items = []
        for idx in indices:
            eid, aid, lbl = pairs[idx]
            we = eeg_subjs[eid]['windows']
            wa = aud_subjs[aid]['windows']
            we = _select_windows_deterministic(we, max_windows)
            wa = _select_windows_deterministic(wa, max_windows)
            K = min(len(we), len(wa))
            we, wa = we[:K], wa[:K]
            we = np.array([_zscore(we[i]) for i in range(len(we))], dtype=np.float32)
            wa = np.array([_zscore(wa[i]) for i in range(len(wa))], dtype=np.float32)
            self.items.append((we, wa, float(lbl), K))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        we, wa, lbl, K = self.items[i]
        # we: [K, 64, 500], wa: [K, 64, 200]
        # Add channel dim: [K, 1, 64, 500]
        return (torch.from_numpy(we).unsqueeze(1),
                torch.from_numpy(wa).unsqueeze(1),
                torch.tensor(lbl, dtype=torch.float),
                K)


def e2e_collate(batch):
    """Collate a list of subjects into padded batch tensors."""
    Ks = [b[3] for b in batch]
    max_K = max(Ks)
    _, C_e, H_e, W_e = batch[0][0].shape
    _, C_a, H_a, W_a = batch[0][1].shape
    B = len(batch)

    X_e = torch.zeros(B, max_K, C_e, H_e, W_e)
    X_a = torch.zeros(B, max_K, C_a, H_a, W_a)
    masks = torch.zeros(B, max_K, dtype=torch.float)
    labels = torch.zeros(B)

    for i, (we, wa, lbl, k) in enumerate(batch):
        X_e[i, :k] = we
        X_a[i, :k] = wa
        masks[i, :k] = 1.0
        labels[i] = lbl

    return X_e, X_a, masks, labels


# ── Backbone wrappers (return features before classifier) ─────────────────

class EEGBackbone(nn.Module):
    """DeepConvNet conv blocks only → [B, D] features."""
    def __init__(self):
        super().__init__()
        self.net = DeepConvNet(N_EEG_CH, 1, 500, 0.5)
        self.feat_dim = self.net.fc_features

    def forward(self, x):
        # x: [N, 1, 64, 500] (already 4D with channel dim)
        x = self.net.block1(x)
        x = self.net.block2(x)
        x = self.net.block3(x)
        x = self.net.block4(x)
        return x.flatten(start_dim=1)

    def load_fase1_weights(self, ckpt_dir, fold_idx):
        ckpt = os.path.join(ckpt_dir, f'fold_{fold_idx}.pt')
        if not os.path.exists(ckpt):
            ckpt = os.path.join(ckpt_dir, 'fold_1.pt')
        state = torch.load(ckpt, map_location='cpu')['model_state_dict']
        own_state = self.net.state_dict()
        for k, v in state.items():
            if 'classifier' in k:
                continue
            k_clean = k[2:] if k.startswith('m.') else k
            if k_clean in own_state:
                own_state[k_clean].copy_(v)
        print(f'    [EEG] loaded Fase 1 weights from {ckpt}')


class AudioBackbone(nn.Module):
    """ShallowConvNet conv blocks only → [B, 1608] features."""
    def __init__(self):
        super().__init__()
        self.net = ShallowConvNet(N_AUDIO_MELS, 1, N_AUDIO_FRAMES, 0.5)
        # compute feat_dim
        dummy = torch.randn(1, 1, N_AUDIO_MELS, N_AUDIO_FRAMES)
        with torch.no_grad():
            x = self.net.temporal_conv(dummy)
            x = self.net.spatial_conv(x)
            x = self.net.bn(x)
            x = torch.square(x)
            x = self.net.pool(x)
            x = torch.log(torch.clamp(x, min=1e-7))
            self.feat_dim = x.flatten(start_dim=1).shape[1]

    def forward(self, x):
        # x: [N, 1, 64, 200] (already 4D with channel dim)
        x = self.net.temporal_conv(x)
        x = self.net.spatial_conv(x)
        x = self.net.bn(x)
        x = torch.square(x)
        x = self.net.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))
        x = self.net.dropout(x)
        return x.flatten(start_dim=1)

    def load_fase1_weights(self, ckpt_dir, fold_idx):
        ckpt = os.path.join(ckpt_dir, f'fold_{fold_idx}.pt')
        if not os.path.exists(ckpt):
            ckpt = os.path.join(ckpt_dir, 'fold_1.pt')
        state = torch.load(ckpt, map_location='cpu')['model_state_dict']
        own_state = self.net.state_dict()
        for k, v in state.items():
            if 'classifier' in k:
                continue
            k_clean = k[2:] if k.startswith('m.') else k
            if k_clean in own_state:
                own_state[k_clean].copy_(v)
        print(f'    [Audio] loaded Fase 1 weights from {ckpt}')


class WindowClassifier(nn.Module):
    """Auxiliary window-level classifier for multi-task learning."""
    def __init__(self, eeg_dim, aud_dim, hidden=64):
        super().__init__()
        self.fc = nn.Linear(eeg_dim + aud_dim, hidden)
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden, 1)

    def forward(self, z_e, z_a):
        # z_e: [B*K, d], z_a: [B*K, d]
        h = self.relu(self.fc(torch.cat([z_e, z_a], dim=-1)))
        return self.head(h).squeeze(-1)


class E2EModel(nn.Module):
    """Full end-to-end model: backbones + fusion + optional window classifier."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eeg_bb = EEGBackbone()
        self.aud_bb = AudioBackbone()
        self.fusion = CrossModalAttention(
            eeg_dim=self.eeg_bb.feat_dim,
            aud_dim=self.aud_bb.feat_dim,
            hidden=args.hidden,
            n_heads=args.n_heads,
            bottleneck_dim=args.bottleneck_dim,
            n_self_attn_layers=args.n_self_attn_layers,
            self_attn_heads=args.self_attn_heads,
            self_attn_dropout=args.self_attn_dropout,
            fusion=args.fusion,
            pooling=args.pooling,
            dropout=args.dropout,
            param_control=args.param_control,
        )
        if args.window_aux:
            self.win_cls = WindowClassifier(self.eeg_bb.feat_dim, self.aud_bb.feat_dim)

    def forward(self, X_e, X_a, mask, return_window=False):
        """X_e: [B, K, 1, 64, 500], X_a: [B, K, 1, 64, 200]"""
        B, K = X_e.shape[0], X_e.shape[1]

        # Flatten subject+window dims for backbone
        X_e_flat = X_e.view(B * K, 1, N_EEG_CH, -1)  # [B*K, 1, 64, 500]
        X_a_flat = X_a.view(B * K, 1, N_AUDIO_MELS, -1)  # [B*K, 1, 64, 200]

        # Backbone forward (differentiable)
        z_e = self.eeg_bb(X_e_flat)  # [B*K, 256]
        z_a = self.aud_bb(X_a_flat)  # [B*K, 1608]

        # Auxiliary window-level logits
        win_logits = None
        if return_window and hasattr(self, 'win_cls'):
            win_logits = self.win_cls(z_e, z_a)  # [B*K]

        # Reshape to subject-level
        z_e = z_e.view(B, K, -1)  # [B, K, 256]
        z_a = z_a.view(B, K, -1)  # [B, K, 1608]

        # Dropout on features (regularization)
        if self.training:
            z_e = nn.functional.dropout(z_e, p=0.3)
            z_a = nn.functional.dropout(z_a, p=0.3)

        # Fusion
        logits = self.fusion(z_e, z_a, mask=mask)  # [B]

        return logits, win_logits

    def load_fase1_init(self, eeg_ckpt_dir, aud_ckpt_dir, fold_idx):
        self.eeg_bb.load_fase1_weights(eeg_ckpt_dir, fold_idx)
        self.aud_bb.load_fase1_weights(aud_ckpt_dir, fold_idx)

    def get_param_groups(self, bb_lr=1e-5, fusion_lr=5e-4, wd=1e-2):
        bb_params = list(self.eeg_bb.parameters()) + list(self.aud_bb.parameters())
        fusion_params = list(self.fusion.parameters())
        if hasattr(self, 'win_cls'):
            fusion_params += list(self.win_cls.parameters())
        return [
            {'params': bb_params, 'lr': bb_lr, 'weight_decay': wd},
            {'params': fusion_params, 'lr': fusion_lr, 'weight_decay': 1e-3},
        ]


# ── Mixup helper ─────────────────────────────────────────────────────────

def mixup_features(z_e, z_a, y, alpha=0.2):
    """Apply mixup at subject level."""
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


# ── Training ─────────────────────────────────────────────────────────────

def train_e2e_fold(model, tr_loader, val_loader, args, fold_idx):
    """Train end-to-end, return (best_model_state, best_val_bacc)."""
    opt = torch.optim.AdamW(model.get_param_groups(
        bb_lr=args.bb_lr, fusion_lr=args.fusion_lr, wd=args.wd))
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

            # Forward
            logits, win_logits = model(X_e, X_a, mask, return_window=args.window_aux)

            # Subject-level loss (label smoothing)
            y_smooth = yb * 0.95 + 0.025
            loss = crit(logits, y_smooth)

            # Window-level auxiliary loss
            if win_logits is not None:
                # Repeat subject labels to window level
                B, K = X_e.shape[0], X_e.shape[1]
                y_win = yb.unsqueeze(1).expand(-1, K).reshape(-1)
                mask_flat = mask.reshape(-1)
                win_loss = crit(win_logits, y_win * 0.95 + 0.025)
                # Only count unmasked windows
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

        # Validation
        model.eval()
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for X_e, X_a, mask, yb in val_loader:
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
    return model, best_vb


def evaluate(model, X_e, X_a, mask, y):
    """Subject-level evaluation for test set."""
    model.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []
    with torch.no_grad():
        for i in range(len(y)):
            ze_s = X_e[i:i+1].to(device)
            za_s = X_a[i:i+1].to(device)
            mk_s = mask[i:i+1].to(device)
            logits, _ = model(ze_s, za_s, mk_s)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true_list.append(int(y[i]))
            y_pred_list.append(int((probs >= 0.5).astype(int)[0]))
            y_prob_list.append(float(probs[0]))
    return np.array(y_true_list), np.array(y_pred_list), np.array(y_prob_list)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='End-to-end crossmodal training')
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bb-lr', type=float, default=1e-5, help='Backbone LR')
    parser.add_argument('--fusion-lr', type=float, default=5e-4, help='Fusion LR')
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=20)
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
    args = parser.parse_args()

    cfg_name = f'{args.fusion}'
    if args.n_self_attn_layers > 0:
        cfg_name += f'_self{args.n_self_attn_layers}L'
    if args.bottleneck_dim is not None:
        cfg_name += f'_bn{args.bottleneck_dim}'
    if args.param_control:
        cfg_name += '_paramctrl'
    cfg_name += f'_e2e_w{args.max_windows}'
    out_dir = os.path.join(OUTPUT_DIR, cfg_name)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'End-to-End CrossModalAttention — {cfg_name}')
    print(f'  Backbone LR={args.bb_lr}  Fusion LR={args.fusion_lr}')
    print(f'  Window aux={args.window_aux} (weight={args.window_aux_weight})')
    print(f'  Mixup alpha={args.mixup_alpha}  Epochs={args.epochs}  Patience={args.patience}')

    # Load data
    eeg_subjs, _, _ = _load_cache(EEG_CACHE)
    aud_subjs, _, _ = _load_cache(AUDIO_CACHE)
    pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
    labels = np.array([p[2] for p in pairs])
    group_ids = np.array([f'p{i}' for i in range(len(pairs))])
    print(f'  Multimodal pairs: {len(pairs)} ({int(labels.sum())} MDD, '
          f'{len(pairs) - int(labels.sum())} HC)')

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(pairs)), labels, groups=group_ids)):
        print(f'\n─── Fold {fi + 1} ───')
        try:
            # Inner split
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                         random_state=RANDOM_STATE + fi)
            tr_i, vl_i = next(inner.split(np.zeros(len(tvi)),
                                          labels[tvi], groups=group_ids[tvi]))
            tr_idx, vl_idx = tvi[tr_i], tvi[vl_i]

            # Build datasets
            tr_ds = SubjectWindowDataset(eeg_subjs, aud_subjs, pairs, tr_idx, args.max_windows)
            vl_ds = SubjectWindowDataset(eeg_subjs, aud_subjs, pairs, vl_idx, args.max_windows)
            tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                                   collate_fn=e2e_collate, num_workers=0)
            vl_loader = DataLoader(vl_ds, batch_size=args.bs, shuffle=False,
                                   collate_fn=e2e_collate, num_workers=0)

            # Build model
            model = E2EModel(args).to(device)
            model.load_fase1_init(EEG_CKPT_DIR, AUDIO_CKPT_DIR, fi + 1)

            total_p = sum(p.numel() for p in model.parameters())
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if fi == 0:
                print(f'    Model params: {total_p:,} total, {trainable_p:,} trainable')
                print(f'    Train subjects: {len(tr_idx)}  Val: {len(vl_idx)}  Test: {len(tei)}')

            # Train
            model, best_vb = train_e2e_fold(model, tr_loader, vl_loader, args, fi)
            print(f'    Best val bacc = {best_vb:.4f}')

            # Save checkpoint
            if args.save_model:
                ckpt_path = os.path.join(ckpt_dir, f'fold_{fi+1}.pt')
                torch.save({
                    'fold': fi + 1,
                    'best_val_bacc': best_vb,
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                }, ckpt_path)
                print(f'    Saved: {ckpt_path}')

            # Evaluate test subjects
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
            'probe_type': 'e2e_fusion',
            'args': vars(args),
            'folds': fold_results,
            'summary': summary,
        }
        out_path = os.path.join(out_dir, 'results.json')
        with open(out_path, 'w') as f:
            json.dump(out_results, f, indent=2)
        print(f'Saved: {out_path}')

        # Append to consolidated CSV
        csv_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'consolidated_results.csv')
        header = 'config_name,probe_type,fusion,n_self_attn,bottleneck_dim,' \
                 'max_windows,bacc_mean,bacc_std,auc_mean,auc_std\n'
        row = f'{cfg_name},e2e_fusion,{args.fusion},{args.n_self_attn_layers},' \
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

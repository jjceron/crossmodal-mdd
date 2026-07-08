"""
Train CrossModalAttention: two-stage protocol (frozen backbones) or end-to-end.

Two-stage (--channels 64, default):
  Stage 1 — Extract frozen backbone features per window.
  Stage 2 — Train fusion + self-attn + head on cached features.

End-to-end (--channels ftsmK or --from-scratch):
  Train EEG encoder + fusion jointly on raw windows with nested CV.
  Audio backbone is frozen (pretrained on 64 mel channels).

Usage:
  py src/training/run_crossmodal.py --fusion concat --channels 64
  py src/training/run_crossmodal.py --fusion concat --channels ftsm16
  py src/training/run_crossmodal.py --fusion concat --from-scratch
"""
import sys, os, json, argparse, warnings, copy
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.models.crossmodal_attn import CrossModalAttention, SelfAttentionBlock
from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.utils.training_logger import ClassificationLogger

AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/crossmodal'
RANDOM_STATE = 42
N_FOLDS = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ── Data loading ──────────────────────────────────────────────────────────

def _load_cache(npz_path):
    c = np.load(npz_path, allow_pickle=True)
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
    """Equispaced window selection (deterministic, no randomness)."""
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    indices = np.linspace(0, n - 1, max_windows, dtype=int)
    return windows[indices]


def _zscore(w):
    return (w - w.mean()) / (w.std() + 1e-8)


# ── Backbone loaders ──────────────────────────────────────────────────────

def _load_eeg_backbone(ckpt_dir, fi, n_channels, n_samples=500):
    ckpt = os.path.join(ckpt_dir, f'fold_{fi}.pt')
    if not os.path.exists(ckpt):
        # fallback to fold_1
        ckpt = os.path.join(ckpt_dir, 'fold_1.pt')
    state = torch.load(ckpt, map_location='cpu')['model_state_dict']

    model = DeepConvNet(n_channels, 1, n_samples, 0.5)
    # Strip 'm.' prefix from wrapper, skip classifier keys
    bb_state = {}
    for k, v in state.items():
        if 'classifier' in k:
            continue
        k_clean = k[2:] if k.startswith('m.') else k
        bb_state[k_clean] = v
    model.load_state_dict(bb_state, strict=False)
    model.to(device)
    model.eval()
    print(f'    [EEG backbone] loaded {ckpt} — frozen, '
          f'conv weights only (classifier SKIPPED)')
    return model


def _load_audio_backbone(ckpt_dir, fi, n_channels=64, n_samples=200):
    ckpt = os.path.join(ckpt_dir, f'fold_{fi}.pt')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ckpt_dir, 'fold_1.pt')
    state = torch.load(ckpt, map_location='cpu')['model_state_dict']

    model = ShallowConvNet(n_channels, 1, n_samples, 0.5)
    bb_state = {}
    for k, v in state.items():
        if 'classifier' in k:
            continue
        k_clean = k[2:] if k.startswith('m.') else k
        bb_state[k_clean] = v
    model.load_state_dict(bb_state, strict=False)
    model.to(device)
    model.eval()
    print(f'    [Audio backbone] loaded {ckpt} — frozen, '
          f'conv weights only (classifier SKIPPED)')
    return model


def _extract_eeg(model, windows, device):
    """Forward through DeepConvNet conv blocks, return features [K, 256]."""
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        with torch.no_grad():
            x = model.block1(batch)
            x = model.block2(x)
            x = model.block3(x)
            x = model.block4(x)
            feats.append(x.flatten(start_dim=1).cpu())
    return torch.cat(feats, dim=0).numpy()


def _extract_audio(model, windows, device):
    """Forward through ShallowConvNet, return features [K, 1608]."""
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        with torch.no_grad():
            x = model.temporal_conv(batch)
            x = model.spatial_conv(x)
            x = model.bn(x)
            x = torch.square(x)
            x = model.pool(x)
            x = torch.log(torch.clamp(x, min=1e-7))
            x = model.dropout(x)
            feats.append(x.flatten(start_dim=1).cpu())
    return torch.cat(feats, dim=0).numpy()


# ── Stage 1: Extract and cache ──────────────────────────────────────────

def extract_fold(eeg_model, aud_model, pairs, fold_idx, args):
    """Extract z_eeg [N, K, eeg_dim] and z_audio [N, K, aud_dim] for subjects."""

    print(f'  Fold {fold_idx + 1}: extracting features...')
    all_ze, all_za, all_y, all_masks = [], [], [], []

    for pair_idx, (eid, aid, lbl) in enumerate(pairs):
        we = eeg_subjs[eid]['windows']
        wa = aud_subjs[aid]['windows']

        we = _select_windows_deterministic(we, args.max_windows)
        wa = _select_windows_deterministic(wa, args.max_windows)

        K = min(len(we), len(wa))
        we, wa = we[:K], wa[:K]

        # Normalize each window
        we = np.array([_zscore(we[i]) for i in range(len(we))])
        wa = np.array([_zscore(wa[i]) for i in range(len(wa))])

        ze = _extract_eeg(eeg_model, we, device)  # [K, 256]
        za = _extract_audio(aud_model, wa, device)  # [K, 1608]

        all_ze.append(ze)
        all_za.append(za)
        all_y.append(lbl)
        all_masks.append(np.ones(K, dtype=np.float32))

    # Pad to same K
    max_K = max(m.shape[0] for m in all_masks)
    Z_e = np.zeros((len(pairs), max_K, all_ze[0].shape[1]), dtype=np.float32)
    Z_a = np.zeros((len(pairs), max_K, all_za[0].shape[1]), dtype=np.float32)
    masks = np.zeros((len(pairs), max_K), dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32)

    for i in range(len(pairs)):
        k = len(all_ze[i])
        Z_e[i, :k] = all_ze[i]
        Z_a[i, :k] = all_za[i]
        masks[i, :k] = all_masks[i]

    return Z_e, Z_a, masks, y_arr


# ── Stage 2: Train fusion ────────────────────────────────────────────────

def train_fold(model, Z_e_tr, Z_a_tr, mask_tr, y_tr,
               Z_e_vl, Z_a_vl, mask_vl, y_vl, args):
    """Train CrossModalAttention on cached features."""
    ds_tr = TensorDataset(torch.FloatTensor(Z_e_tr), torch.FloatTensor(Z_a_tr),
                          torch.FloatTensor(mask_tr), torch.FloatTensor(y_tr))
    ds_vl = TensorDataset(torch.FloatTensor(Z_e_vl), torch.FloatTensor(Z_a_vl),
                          torch.FloatTensor(mask_vl), torch.FloatTensor(y_vl))
    tr_ldr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True)
    vl_ldr = DataLoader(ds_vl, batch_size=args.bs, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, foreach=False)
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
        for ze, za, m, yb in tr_ldr:
            ze, za, m, yb = ze.to(device), za.to(device), m.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(ze, za, mask=m)
            y_smooth = yb * 0.95 + 0.025
            loss = crit(logits, y_smooth)
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
            for ze, za, m, yb in vl_ldr:
                logits = model(ze.to(device), za.to(device), mask=m.to(device))
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


def evaluate(model, Z_e_te, Z_a_te, mask_te, y_te):
    """Subject-level evaluation."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(Z_e_te).to(device),
                       torch.FloatTensor(Z_a_te).to(device),
                       mask=torch.FloatTensor(mask_te).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    prob_subj = float(probs.mean())  # single subject
    pred = (probs >= 0.5).astype(int)
    return np.array([y_te]), np.array([pred]), np.array([prob_subj])


# ── End-to-end training (no pretrained backbones) ──────────────────────────

def _parse_channels(ch_str):
    """Parse --channels value into (n_channels, cache_suffix)."""
    v = ch_str.lower()
    if v == '128':
        return 128, '128ch'
    if v == '64':
        return 64, '64ch'
    if v == '19':
        return 19, '19ch'
    if v.startswith('ftsm'):
        return int(v.replace('ftsm', '')), v
    return int(v), f'{v}ch'


def _encode_audio(aud_encoder, windows, device):
    """Forward audio windows through ShallowConvNet conv blocks (no classifier)."""
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        x = aud_encoder.temporal_conv(batch)
        x = aud_encoder.spatial_conv(x)
        x = aud_encoder.bn(x)
        x = torch.square(x)
        x = aud_encoder.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))
        x = aud_encoder.dropout(x)
        feats.append(x.flatten(start_dim=1))
    return torch.cat(feats, dim=0)


def _encode_eeg(eeg_encoder, windows, device):
    """Forward EEG windows through DeepConvNet conv blocks (no classifier)."""
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        x = eeg_encoder.block1(batch)
        x = eeg_encoder.block2(x)
        x = eeg_encoder.block3(x)
        x = eeg_encoder.block4(x)
        feats.append(x.flatten(start_dim=1))
    return torch.cat(feats, dim=0)


def train_e2e_fold(eeg_encoder, aud_encoder, fusion_model,
                   tr_subjs, vl_subjs, args):
    """Train eeg_encoder + aud_encoder + fusion jointly from scratch."""
    params = (list(eeg_encoder.parameters())
              + list(aud_encoder.parameters())
              + list(fusion_model.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, foreach=False)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    crit = nn.BCEWithLogitsLoss()

    best_vb, best_st, pat = 0.0, None, 0
    logger = ClassificationLogger()
    logger.log_header()

    for ep in range(1, args.epochs + 1):
        eeg_encoder.train()
        aud_encoder.train()
        fusion_model.train()
        tr_loss, tr_n = 0.0, 0
        tr_logits, tr_labels = [], []

        for eid, aid, lbl in tr_subjs:
            we = eeg_subjs[eid]['windows']
            wa = aud_subjs[aid]['windows']
            we = _select_windows_deterministic(we, args.max_windows)
            wa = _select_windows_deterministic(wa, args.max_windows)

            K = min(len(we), len(wa))
            we, wa = we[:K], wa[:K]
            we = np.array([_zscore(we[i]) for i in range(K)])
            wa = np.array([_zscore(wa[i]) for i in range(K)])

            opt.zero_grad()
            ze = _encode_eeg(eeg_encoder, we, device)      # [K, 256]
            za = _encode_audio(aud_encoder, wa, device)     # [K, 1608]
            ze, za = ze.unsqueeze(0), za.unsqueeze(0)      # [1, K, D]
            logit = fusion_model(ze, za, mask=torch.ones(1, K, device=device))
            y_smooth = torch.FloatTensor([lbl]).to(device) * 0.95 + 0.025
            loss = crit(logit, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            tr_loss += loss.item()
            tr_n += 1
            tr_logits.append(logit.detach())
            tr_labels.append(torch.FloatTensor([lbl]).to(device))

        tr_loss /= tr_n
        tr_logits_cat = torch.cat(tr_logits)
        tr_labels_cat = torch.cat(tr_labels)
        tr_pred = (torch.sigmoid(tr_logits_cat).cpu().numpy() >= 0.5).astype(int)
        tr_m = logger.metrics(tr_labels_cat.cpu().numpy(), tr_pred)

        # Validation
        eeg_encoder.eval()
        aud_encoder.eval()
        fusion_model.eval()
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for eid, aid, lbl in vl_subjs:
                we = eeg_subjs[eid]['windows']
                wa = aud_subjs[aid]['windows']
                we = _select_windows_deterministic(we, args.max_windows)
                wa = _select_windows_deterministic(wa, args.max_windows)
                K = min(len(we), len(wa))
                we, wa = we[:K], wa[:K]
                we = np.array([_zscore(we[i]) for i in range(K)])
                wa = np.array([_zscore(wa[i]) for i in range(K)])

                ze = _encode_eeg(eeg_encoder, we, device).unsqueeze(0)
                za = _encode_audio(aud_encoder, wa, device).unsqueeze(0)
                logit = fusion_model(ze, za, mask=torch.ones(1, K, device=device))
                vl_logits.append(logit.cpu())
                vl_labels.append(torch.FloatTensor([lbl]))

        vl_logits = torch.cat(vl_logits)
        vl_labels = torch.cat(vl_labels)
        vl_loss = crit(vl_logits, vl_labels).item()
        vl_pred = (torch.sigmoid(vl_logits).numpy() >= 0.5).astype(int)
        vl_m = logger.metrics(vl_labels.numpy(), vl_pred)
        sched.step(vl_m['bacc'])

        if vl_m['bacc'] > best_vb:
            best_vb = vl_m['bacc']
            best_st = (copy.deepcopy(eeg_encoder.state_dict()),
                       copy.deepcopy(aud_encoder.state_dict()),
                       copy.deepcopy(fusion_model.state_dict()))
            pat = 0
        else:
            pat += 1

        if ep == 1 or pat == 0 or ep % 10 == 0:
            logger.log_epoch(ep, tr_loss, vl_loss, tr_m, vl_m, pat)

        if pat >= args.patience:
            break

    if best_st is None:
        # No improvement during training; use last state
        best_vb = 0.0
    else:
        eeg_encoder.load_state_dict(best_st[0])
        aud_encoder.load_state_dict(best_st[1])
        fusion_model.load_state_dict(best_st[2])
    return eeg_encoder, aud_encoder, fusion_model, best_vb


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CrossModalAttention training')
    parser.add_argument('--fusion', choices=['concat', 'gating', 'cross_attn'],
                        default='cross_attn')
    parser.add_argument('--n-self-attn-layers', type=int, default=0)
    parser.add_argument('--self-attn-heads', type=int, default=4)
    parser.add_argument('--self-attn-dropout', type=float, default=0.1)
    parser.add_argument('--bottleneck-dim', type=int, default=None,
                        help='Compress backbone output to N before proj to hidden')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=2)
    parser.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max-windows', type=int, default=50,
                        help='Equispaced windows per subject')
    parser.add_argument('--channels', type=str, default='64',
                        help='EEG channel selection: 64, 128, 19, or ftsm4|8|16|32|64')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train EEG backbone from scratch (no pretrained). Auto-enabled when channels!=64.')
    parser.add_argument('--eeg-ckpt-dir', type=str,
                        default='outputs/results/classical_dl/trained_eeg/deepconvnet_64ch')
    parser.add_argument('--audio-ckpt-dir', type=str,
                        default='outputs/results/classical_dl/trained_audio/shallowconvnet_64mel')
    parser.add_argument('--from-cache', type=str, default=None,
                        help='Path to cache dir (Stage 1 output). Skips backbone loading/extraction.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--param-control', action='store_true',
                        help='Add MLP with matching param count when n_self_attn=0')
    parser.add_argument('--save-model', action='store_true',
                        help='Save best model checkpoint per fold')
    args = parser.parse_args()

    n_channels, cache_suffix = _parse_channels(args.channels)
    eeg_cache = f'data/processed/eeg_preprocessed_{cache_suffix}.npz'
    from_scratch = args.from_scratch or (cache_suffix != '64ch')

    # Config name for output
    cfg_name = f'{args.fusion}'
    if args.n_self_attn_layers > 0:
        cfg_name += f'_self{args.n_self_attn_layers}L'
    elif args.param_control:
        cfg_name += '_paramctrl'
    if args.bottleneck_dim is not None:
        cfg_name += f'_bn{args.bottleneck_dim}'
    if cache_suffix != '64ch':
        cfg_name += f'_{cache_suffix}'
    elif args.from_scratch:
        cfg_name += '_scratch'
    cfg_name += f'_w{args.max_windows}'

    out_dir = os.path.join(OUTPUT_DIR, cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    mode_str = 'from-scratch' if from_scratch else 'two-stage (frozen backbones)'
    print(f'Device: {device}')
    print(f'CrossModalAttention — {cfg_name}  [{mode_str}]')
    print(f'  EEG cache: {eeg_cache}  ({n_channels} channels)')
    if not from_scratch:
        print(f'  Backbones: EEG={args.eeg_ckpt_dir}  Audio={args.audio_ckpt_dir}')
    print(f'  Fusion={args.fusion}  Self-attn layers={args.n_self_attn_layers}')
    print(f'  Hidden={args.hidden}  Bottleneck={args.bottleneck_dim}')
    print(f'  Max windows={args.max_windows}  Pooling={args.pooling}')
    print(f'  Epochs={args.epochs}  LR={args.lr}  BS={args.bs}')

    # Load data
    global eeg_subjs, aud_subjs, pairs
    eeg_subjs, eeg_ids, _ = _load_cache(eeg_cache)
    aud_subjs, aud_ids, _ = _load_cache(AUDIO_CACHE)
    pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
    labels = np.array([p[2] for p in pairs])
    group_ids = np.array([f'p{i}' for i in range(len(pairs))])
    print(f'  Multimodal pairs: {len(pairs)} ({int(labels.sum())} MDD, '
          f'{len(pairs) - int(labels.sum())} HC)')
    print(f'  EEG channels: {eeg_subjs[pairs[0][0]]["windows"].shape[1]}')

    # Subject-level GKF (same seed for all configs)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    # Save fold split for audit
    fold_mapping = {}
    for fi, (_, tei) in enumerate(skf.split(np.zeros(len(pairs)), labels, groups=group_ids)):
        fold_mapping[f'fold_{fi+1}'] = [pairs[i][0] for i in tei]

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(pairs)), labels, groups=group_ids)):
        print(f'\n  Fold {fi + 1}')
        try:
            # Inner split train/val
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                         random_state=RANDOM_STATE + fi)
            tr_i, vl_i = next(inner.split(np.zeros(len(tvi)),
                                          labels[tvi], groups=group_ids[tvi]))
            tr_idx = tvi[tr_i]
            vl_idx = tvi[vl_i]

            tr_subjs = [pairs[i] for i in tr_idx]
            vl_subjs = [pairs[i] for i in vl_idx]
            te_subjs = [pairs[i] for i in tei]

            if from_scratch:
                # ── End-to-end: train all models jointly from scratch ──
                eeg_encoder = DeepConvNet(n_channels, 1, 500, 0.5).to(device)
                aud_encoder = ShallowConvNet(64, 1, 200, 0.5).to(device)

                # Compute encoder output dims dynamically
                with torch.no_grad():
                    de = torch.randn(1, 1, n_channels, 500, device=device)
                    da = torch.randn(1, 1, 64, 200, device=device)
                    eeg_dim = eeg_encoder.block4(
                        eeg_encoder.block3(
                            eeg_encoder.block2(
                                eeg_encoder.block1(de)
                            )
                        )
                    ).flatten(start_dim=1).shape[1]
                    aud_dim = aud_encoder.pool(
                        torch.square(
                            aud_encoder.bn(
                                aud_encoder.spatial_conv(
                                    aud_encoder.temporal_conv(da)
                                )
                            )
                        )
                    ).flatten(start_dim=1).shape[1]

                fusion_model = CrossModalAttention(
                    eeg_dim=eeg_dim, aud_dim=aud_dim,
                    hidden=args.hidden, n_heads=args.n_heads,
                    bottleneck_dim=args.bottleneck_dim,
                    n_self_attn_layers=args.n_self_attn_layers,
                    self_attn_heads=args.self_attn_heads,
                    self_attn_dropout=args.self_attn_dropout,
                    fusion=args.fusion, pooling=args.pooling, dropout=args.dropout,
                ).to(device)

                if fi == 0:
                    total_p = (sum(p.numel() for p in eeg_encoder.parameters())
                               + sum(p.numel() for p in aud_encoder.parameters())
                               + sum(p.numel() for p in fusion_model.parameters()))
                    print(f'    E2E model params: {total_p:,} total  '
                          f'(eeg={sum(p.numel() for p in eeg_encoder.parameters()):,} '
                          f'+ aud={sum(p.numel() for p in aud_encoder.parameters()):,} '
                          f'+ fusion={sum(p.numel() for p in fusion_model.parameters()):,})')

                eeg_encoder, aud_encoder, fusion_model, best_vb = train_e2e_fold(
                    eeg_encoder, aud_encoder, fusion_model,
                    tr_subjs, vl_subjs, args)

                # Evaluate on test subjects
                eeg_encoder.eval()
                aud_encoder.eval()
                fusion_model.eval()
                y_true_list, y_pred_list, y_prob_list = [], [], []
                with torch.no_grad():
                    for eid, aid, lbl in te_subjs:
                        we = eeg_subjs[eid]['windows']
                        wa = aud_subjs[aid]['windows']
                        we = _select_windows_deterministic(we, args.max_windows)
                        wa = _select_windows_deterministic(wa, args.max_windows)
                        K = min(len(we), len(wa))
                        we, wa = we[:K], wa[:K]
                        we = np.array([_zscore(we[i]) for i in range(K)])
                        wa = np.array([_zscore(wa[i]) for i in range(K)])

                        ze = _encode_eeg(eeg_encoder, we, device).unsqueeze(0)
                        za = _encode_audio(aud_encoder, wa, device).unsqueeze(0)
                        logit = fusion_model(ze, za, mask=torch.ones(1, K, device=device))
                        prob = float(torch.sigmoid(logit).cpu().numpy().mean())
                        pred = int(prob >= 0.5)
                        y_true_list.append(lbl)
                        y_pred_list.append(pred)
                        y_prob_list.append(prob)

                del eeg_encoder, aud_encoder, fusion_model
            else:
                # ── Two-stage: frozen backbones + trained fusion ──
                if args.from_cache:
                    c = np.load(os.path.join(args.from_cache, f'fold_{fi+1}.npz'),
                                allow_pickle=True)
                    Z_e, Z_a, masks, y = c['Z_e'], c['Z_a'], c['mask'], c['y']
                    tr_idx, vl_idx = c['tr_idx'], c['vl_idx']
                    print(f'  Loaded cached features: Z_e {Z_e.shape}, Z_a {Z_a.shape}')
                else:
                    eeg_model = _load_eeg_backbone(args.eeg_ckpt_dir, fi + 1, n_channels)
                    aud_model = _load_audio_backbone(args.audio_ckpt_dir, fi + 1)
                    pairs_arr = np.array(pairs, dtype=object)
                    Z_e, Z_a, masks, y = extract_fold(
                        eeg_model, aud_model, pairs_arr, fi, args)

                # Split
                Z_e_tr, Z_a_tr = Z_e[tr_idx], Z_a[tr_idx]
                Z_e_vl, Z_a_vl = Z_e[vl_idx], Z_a[vl_idx]
                Z_e_te, Z_a_te = Z_e[tei], Z_a[tei]
                mask_tr, mask_vl = masks[tr_idx], masks[vl_idx]
                mask_te = masks[tei]
                y_tr, y_vl = y[tr_idx], y[vl_idx]
                y_te = y[tei]

                eeg_dim = Z_e.shape[2]
                aud_dim = Z_a.shape[2]
                if fi == 0:
                    print(f'    EEG feat dim={eeg_dim}  Audio feat dim={aud_dim}')
                    print(f'    Train={len(tr_idx)}  Val={len(vl_idx)}  Test={len(tei)}')

                model = CrossModalAttention(
                    eeg_dim=eeg_dim, aud_dim=aud_dim,
                    hidden=args.hidden, n_heads=args.n_heads,
                    bottleneck_dim=args.bottleneck_dim,
                    n_self_attn_layers=args.n_self_attn_layers,
                    self_attn_heads=args.self_attn_heads,
                    self_attn_dropout=args.self_attn_dropout,
                    fusion=args.fusion, pooling=args.pooling, dropout=args.dropout,
                ).to(device)

                if fi == 0:
                    total_p = sum(p.numel() for p in model.parameters())
                    print(f'    Fusion model params: {total_p:,} trainable')

                if args.param_control and args.n_self_attn_layers == 0:
                    dummy = SelfAttentionBlock(args.hidden, args.self_attn_heads,
                                              args.self_attn_dropout)
                    n_ctrl = sum(p.numel() for p in dummy.parameters())
                    model.ctrl_mlp = nn.Sequential(
                        nn.Linear(args.hidden, n_ctrl // args.hidden),
                        nn.ReLU(),
                        nn.Linear(n_ctrl // args.hidden, args.hidden),
                    ).to(device)
                    print(f'    Param control MLP added ({n_ctrl:,} target params)')

                model, best_vb = train_fold(model, Z_e_tr, Z_a_tr, mask_tr, y_tr,
                                            Z_e_vl, Z_a_vl, mask_vl, y_vl, args)

                # Evaluate subject-level
                y_true_list, y_pred_list, y_prob_list = [], [], []
                for si in range(len(tei)):
                    yt_s, yp_s, ypr_s = evaluate(
                        model, Z_e_te[si:si+1], Z_a_te[si:si+1],
                        mask_te[si:si+1], y_te[si:si+1])
                    y_true_list.append(yt_s[0])
                    y_pred_list.append(yp_s[0])
                    y_prob_list.append(ypr_s[0])

                if not args.from_cache:
                    del eeg_model, aud_model
                del model

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
                'best_val_bacc': float(best_vb),
                'test_metrics': fm,
                'test_bacc': float(bacc),
                'test_auc': roc_auc,
                'test_cm': cm,
                'test_roc': {'y_true': y_true_s.tolist(), 'y_prob': y_prob_s.tolist()},
            })

            print(f'  Fold {fi + 1}: bacc={bacc:.3f} AUC={roc_auc:.3f}')

            if args.save_model:
                ckpt_dir = os.path.join(out_dir, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    'fold': fi + 1, 'best_val_bacc': float(best_vb),
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                }, os.path.join(ckpt_dir, f'fold_{fi+1}.pt'))
                print(f'    Saved checkpoint: fold_{fi+1}.pt')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  Fold {fi + 1} FAILED: {e}')
            import traceback
            traceback.print_exc()

    if fold_results:
        baccs = [r['test_bacc'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        summary = {
            'bacc_mean': float(np.mean(baccs)),
            'bacc_std': float(np.std(baccs)),
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
        }

        print(f'\n{"=" * 55}')
        print(f'  {cfg_name}')
        print(f'  bacc = {summary["bacc_mean"]:.3f} ± {summary["bacc_std"]:.3f}')
        print(f'  auc  = {summary["auc_mean"]:.3f} ± {summary["auc_std"]:.3f}')
        print(f'{"=" * 55}')

        out_results = {
            'config_name': cfg_name,
            'args': vars(args),
            'probe_type': 'fusion_probe',
            'folds': fold_results,
            'summary': summary,
            'fold_subject_split': fold_mapping,
        }
        out_path = os.path.join(out_dir, 'results.json')
        with open(out_path, 'w') as f:
            json.dump(out_results, f, indent=2)
        print(f'Saved: {out_path}')

        # Consolidated CSV
        csv_path = os.path.join(OUTPUT_DIR, 'consolidated_results.csv')
        header = 'config_name,probe_type,fusion,n_self_attn,bottleneck_dim,' \
                 'max_windows,bacc_mean,bacc_std,auc_mean,auc_std\n'
        row = f'{cfg_name},fusion_probe,{args.fusion},{args.n_self_attn_layers},' \
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
    eeg_subjs = aud_subjs = pairs = None
    main()

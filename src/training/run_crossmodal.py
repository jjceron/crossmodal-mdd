"""
Train CrossModalAttention: two-stage protocol.
Stage 1 — Extract frozen backbone features per window.
Stage 2 — Train fusion + self-attn + head on cached features.

Usage:
  py src/training/run_crossmodal.py --fusion cross_attn --n-self-attn-layers 1
  py src/training/run_crossmodal.py --fusion cross_attn --n-self-attn-layers 0
  py src/training/run_crossmodal.py --fusion concat --n-self-attn-layers 0
"""
import sys, os, json, argparse, warnings, copy
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.models.crossmodal_attention import CrossModalAttention
from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.utils.training_logger import ClassificationLogger

EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
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
    subjects = {}
    for i, sid in enumerate(ids):
        subjects[sid] = {'windows': wins[i], 'label': int(labels[i])}
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

def _load_eeg_backbone(ckpt_dir, fi, n_channels=64, n_samples=500):
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

    # Config name for output
    cfg_name = f'{args.fusion}'
    if args.n_self_attn_layers > 0:
        cfg_name += f'_self{args.n_self_attn_layers}L'
    elif args.param_control:
        cfg_name += '_paramctrl'
    if args.bottleneck_dim is not None:
        cfg_name += f'_bn{args.bottleneck_dim}'
    cfg_name += f'_w{args.max_windows}'

    out_dir = os.path.join(OUTPUT_DIR, cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'CrossModalAttention — {cfg_name}')
    print(f'  Backbones: EEG={args.eeg_ckpt_dir}  Audio={args.audio_ckpt_dir}')
    print(f'  Fusion={args.fusion}  Self-attn layers={args.n_self_attn_layers}')
    print(f'  Hidden={args.hidden}  Bottleneck={args.bottleneck_dim}')
    print(f'  Max windows={args.max_windows}  Pooling={args.pooling}')
    print(f'  Epochs={args.epochs}  LR={args.lr}  BS={args.bs}')

    # Load data
    global eeg_subjs, aud_subjs, pairs
    if args.from_cache:
        # Cache mode: load pairs only for fold mapping
        eeg_subjs, _, _ = _load_cache(EEG_CACHE)
        aud_subjs, _, _ = _load_cache(AUDIO_CACHE)
        pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
        labels = np.array([p[2] for p in pairs])
        group_ids = np.array([f'p{i}' for i in range(len(pairs))])
        print(f'  Cache mode: {args.from_cache}  |  {len(pairs)} multimodal pairs')
    else:
        eeg_subjs, eeg_ids, _ = _load_cache(EEG_CACHE)
        aud_subjs, aud_ids, _ = _load_cache(AUDIO_CACHE)
        pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
        labels = np.array([p[2] for p in pairs])
        group_ids = np.array([f'p{i}' for i in range(len(pairs))])
        print(f'  Extract mode: EEG={args.eeg_ckpt_dir}  Audio={args.audio_ckpt_dir}')
        print(f'  Multimodal pairs: {len(pairs)} ({int(labels.sum())} MDD, '
              f'{len(pairs) - int(labels.sum())} HC)')

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

            if args.from_cache:
                # ── Load from cache (Stage 1 output) ──
                c = np.load(os.path.join(args.from_cache, f'fold_{fi+1}.npz'),
                            allow_pickle=True)
                Z_e, Z_a, masks, y = c['Z_e'], c['Z_a'], c['mask'], c['y']
                tr_idx, vl_idx = c['tr_idx'], c['vl_idx']
                print(f'  Loaded cached features: Z_e {Z_e.shape}, Z_a {Z_a.shape}')
            else:
                # Load backbones
                eeg_model = _load_eeg_backbone(args.eeg_ckpt_dir, fi + 1)
                aud_model = _load_audio_backbone(args.audio_ckpt_dir, fi + 1)

                # ── Stage 1: Extract features ──
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

            # ── Stage 2: Build and train model ──
            model = CrossModalAttention(
                eeg_dim=eeg_dim,
                aud_dim=aud_dim,
                hidden=args.hidden,
                n_heads=args.n_heads,
                bottleneck_dim=args.bottleneck_dim,
                n_self_attn_layers=args.n_self_attn_layers,
                self_attn_heads=args.self_attn_heads,
                self_attn_dropout=args.self_attn_dropout,
                fusion=args.fusion,
                pooling=args.pooling,
                dropout=args.dropout,
            ).to(device)

            if fi == 0:
                total_p = sum(p.numel() for p in model.parameters())
                trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f'    Model params: {total_p:,} total, {trainable_p:,} trainable')

            # Add param-control MLP if requested
            if args.param_control and args.n_self_attn_layers == 0:
                # Count params of one SelfAttentionBlock
                dummy = SelfAttentionBlock(args.hidden, args.self_attn_heads,
                                          args.self_attn_dropout)
                n_ctrl = sum(p.numel() for p in dummy.parameters())
                # Add an MLP with ~same count
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
                ze_s = Z_e_te[si:si+1]
                za_s = Z_a_te[si:si+1]
                mk_s = mask_te[si:si+1]
                yt = y_te[si:si+1]
                yt_s, yp_s, ypr_s = evaluate(model, ze_s, za_s, mk_s, yt)
                y_true_list.append(yt_s[0])
                y_pred_list.append(yp_s[0])
                y_prob_list.append(ypr_s[0])

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

            if not args.from_cache:
                del eeg_model, aud_model
            del model
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
    # Workaround for global usage in extract_fold
    eeg_subjs = aud_subjs = pairs = None
    from src.models.crossmodal_attention import SelfAttentionBlock
    main()

"""Train CAMP-Net: cross-modal fusion with nested backbone pre-training.

Nested protocol (zero leakage):
  For each CAMP-Net fold:
    1. Pre-train EEG backbone DCNN ONLY on train subjects of this fold
    2. Pre-train Audio backbone SCNN ONLY on train subjects of this fold
    3. Freeze both backbones
    4. Train CAMP-Net fusion + head on multimodal train data
    5. Evaluate on test data

Modes:
  eeg_only      — EEG backbone → pool → head
  audio_only    — Audio backbone → pool → head
  early_fusion  — Both backbones → concat pooled → head
  late_fusion   — Both backbones → average logits
  cross_attn    — Both backbones → cross-MHA → pool → head

Data:
  EEG:  data/processed/eeg_preprocessed_64ch.npz (53 subjects)
  Audio: data/processed/audio_gcn_cache.npz (52 subjects)
  Multimodal mapping: data/processed/multimodal_mapping.json

Usage:
  python src/training/run_campnet.py --mode cross_attn --epochs 20
  python src/training/run_campnet.py --mode eeg_only --epochs 20
"""
import sys, os, json, argparse, warnings, copy
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.models.campnet import CrossModalDL
from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.utils.training_logger import ClassificationLogger
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score

EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/ocampnet'
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ── Data loading ────────────────────────────────────────────────────────

def load_cached(npz_path):
    c = np.load(npz_path, allow_pickle=True)
    wins = c['windows']
    labels = c['labels']
    ids = c['subject_ids']
    subjects = {}
    for i in range(len(ids)):
        subjects[str(ids[i])] = {'windows': wins[i], 'label': int(labels[i])}
    return subjects, sorted(subjects.keys()), labels


def load_multimodal_pairs(eeg_subjs, aud_subjs):
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    pairs = []
    for aud_id, eeg_id in mapping['orig_to_bids'].items():
        if eeg_id in eeg_subjs and aud_id in aud_subjs:
            pairs.append((eeg_id, aud_id, eeg_subjs[eeg_id]['label']))
    return pairs


def _stack_windows(mode, idx_list, eeg_ids, aud_ids, pairs, eeg_subjs, aud_subjs):
    if mode == 'eeg_only':
        sids = [eeg_ids[i] for i in idx_list]
        X = np.concatenate([eeg_subjs[s]['windows'] for s in sids])
        y = np.concatenate([np.full(len(eeg_subjs[s]['windows']), eeg_subjs[s]['label']) for s in sids])
        return X, y
    elif mode == 'audio_only':
        sids = [aud_ids[i] for i in idx_list]
        X = np.concatenate([aud_subjs[s]['windows'] for s in sids])
        y = np.concatenate([np.full(len(aud_subjs[s]['windows']), aud_subjs[s]['label']) for s in sids])
        return X, y
    else:
        Xe_p, Xa_p, y_p = [], [], []
        for i in idx_list:
            eid, aid = pairs[i][0], pairs[i][1]
            we = eeg_subjs[eid]['windows']; wa = aud_subjs[aid]['windows']
            n = min(len(we), len(wa))
            Xe_p.append(we[:n]); Xa_p.append(wa[:n])
            y_p.append(np.full(n, eeg_subjs[eid]['label']))
        return (np.concatenate(Xe_p), np.concatenate(Xa_p)), np.concatenate(y_p)


def _zscore_per_window(X):
    if isinstance(X, tuple) and len(X) == 2:
        m_e = X[0].mean(axis=(1, 2), keepdims=True)
        s_e = X[0].std(axis=(1, 2), keepdims=True) + 1e-8
        m_a = X[1].mean(axis=(1, 2), keepdims=True)
        s_a = X[1].std(axis=(1, 2), keepdims=True) + 1e-8
        return ((X[0] - m_e) / s_e, (X[1] - m_a) / s_a)
    return (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-8)


def _make_loader(X, y, bs, shuffle):
    if isinstance(X, tuple):
        ds = TensorDataset(
            torch.FloatTensor(X[0]), torch.FloatTensor(X[1]), torch.FloatTensor(y))
    else:
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def _forward_batch(model, mode, batch):
    if mode == 'eeg_only':
        logits, _ = model(eeg_x=batch[0].to(device), mode=mode)
        return logits, batch[-1].to(device)
    elif mode == 'audio_only':
        logits, _ = model(audio_x=batch[0].to(device), mode=mode)
        return logits, batch[-1].to(device)
    else:
        logits, _ = model(eeg_x=batch[0].to(device), audio_x=batch[1].to(device), mode=mode)
        return logits, batch[-1].to(device)


# ── Backbone pre-training (per fold, no leakage) ────────────────

class _BackboneWrapper(nn.Module):
    """Trainable wrapper: model σ 1 → 1 logit (BCE)."""
    def __init__(self, backbone, n_channels, n_samples):
        super().__init__()
        self.backbone = backbone(n_channels, 1, n_samples, 0.5)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


def _pretrain_backbone_tensors(X, y, backbone_cls, n_channels, n_samples, args, label=''):
    """Pre-train backbone on given windows only. Returns best model."""
    n_total = len(X)
    n_tr = max(int(n_total * 0.8), 2)
    idx = np.random.RandomState(RANDOM_STATE).permutation(n_total)
    Xtr, ytr = X[idx[:n_tr]], y[idx[:n_tr]]
    Xvl, yvl = X[idx[n_tr:]], y[idx[n_tr:]]

    Xtr_n = _zscore_per_window(Xtr)
    Xvl_n = _zscore_per_window(Xvl)
    tr_ldr = DataLoader(TensorDataset(torch.FloatTensor(Xtr_n), torch.FloatTensor(ytr)),
                        batch_size=32, shuffle=True)
    vl_ldr = DataLoader(TensorDataset(torch.FloatTensor(Xvl_n), torch.FloatTensor(yvl)),
                        batch_size=32, shuffle=False)

    model = _BackboneWrapper(backbone_cls, n_channels, n_samples).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3, foreach=False)
    crit = nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    best_vb, best_st, pat = 0.0, None, 0

    for ep in range(1, args.backbone_epochs + 1):
        model.train()
        for batch in tr_ldr:
            opt.zero_grad()
            logits = model(batch[0].to(device))
            yb = batch[1].to(device)
            y_smooth = yb * 0.95 + 0.025
            loss = crit(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl_loss, vl_n = 0.0, 0
        all_p, all_t = [], []
        with torch.no_grad():
            for batch in vl_ldr:
                logits = model(batch[0].to(device))
                yb = batch[1].to(device)
                vl_loss += crit(logits, yb).item() * yb.size(0)
                vl_n += yb.size(0)
                all_p.append((torch.sigmoid(logits) >= 0.5).float().cpu().numpy())
                all_t.append(yb.cpu().numpy())
        vl_bacc = float((np.concatenate(all_p) == np.concatenate(all_t)).mean())
        sched.step(vl_bacc)

        if vl_bacc > best_vb:
            best_vb = vl_bacc
            best_st = copy.deepcopy(model.backbone.state_dict())
            pat = 0
        else:
            pat += 1
        if pat >= args.patience:
            break

    model.backbone.load_state_dict(best_st)
    if label:
        print(f'  [{label}] backbone: {ep} epochs, val_bacc={best_vb:.3f}')
    return model.backbone


# ── CAMP-Net training helpers ──────────────────────────────────

def _logits_to_binary(logits):
    return (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)


def _compute_epoch_metrics(model, loader, mode, crit, logger):
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits, y = _forward_batch(model, mode, batch)
            total_loss += crit(logits, y).item() * y.size(0)
            n += y.size(0)
            all_logits.append(logits)
            all_labels.append(y)
    loss = total_loss / n
    preds = _logits_to_binary(torch.cat(all_logits))
    trues = torch.cat(all_labels).cpu().numpy()
    return loss, logger.metrics(trues, preds)


def _train_fusion_fold(model, mode, tr_loader, vl_loader, args, logger):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=False)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    crit = nn.BCEWithLogitsLoss()
    best_vb, best_st, pat = 0.0, None, 0
    history = {k: [] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc',
                                'val_bacc', 'val_f1', 'val_sens', 'val_spec')}

    logger.log_header()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        tr_logits, tr_labels = [], []
        for batch in tr_loader:
            opt.zero_grad()
            logits, y = _forward_batch(model, mode, batch)
            if torch.isnan(logits).any():
                raise RuntimeError('NaN in logits — training diverged')
            y_smooth = y * 0.95 + 0.025
            loss = crit(logits, y_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * y.size(0)
            tr_n += y.size(0)
            tr_logits.append(logits.detach())
            tr_labels.append(y)
        tr_loss /= tr_n

        tr_pred = _logits_to_binary(torch.cat(tr_logits))
        tr_true = torch.cat(tr_labels).cpu().numpy()
        tr_m = logger.metrics(tr_true, tr_pred)

        vl_loss, vl_m = _compute_epoch_metrics(model, vl_loader, mode, crit, logger)
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

        if pat >= args.patience:
            break

    return best_st, best_vb, history


def _evaluate_subject_level(model, mode, eeg_subjs, aud_subjs, pairs,
                            eeg_ids, aud_ids, tei, test_bs=32):
    model.eval()
    test_preds, test_true, test_sids = [], [], []
    with torch.no_grad():
        for si in tei:
            if mode == 'eeg_only':
                sid = eeg_ids[si]
                w = eeg_subjs[sid]['windows']; lbl = eeg_subjs[sid]['label']
                w = _zscore_per_window(w)
                probs = []
                for i in range(0, len(w), test_bs):
                    logits, _ = model(eeg_x=torch.FloatTensor(w[i:i+test_bs]).to(device), mode=mode)
                    probs.append(torch.sigmoid(logits).cpu().numpy())
                p = np.concatenate(probs)
            elif mode == 'audio_only':
                sid = aud_ids[si]
                w = aud_subjs[sid]['windows']; lbl = aud_subjs[sid]['label']
                w = _zscore_per_window(w)
                probs = []
                for i in range(0, len(w), test_bs):
                    logits, _ = model(audio_x=torch.FloatTensor(w[i:i+test_bs]).to(device), mode=mode)
                    probs.append(torch.sigmoid(logits).cpu().numpy())
                p = np.concatenate(probs)
            else:
                eid, aid, lbl = pairs[si]; sid = eid
                we = eeg_subjs[eid]['windows']; wa = aud_subjs[aid]['windows']
                n_min = min(len(we), len(wa))
                we = we[:n_min]; wa = wa[:n_min]
                we = _zscore_per_window(we); wa = _zscore_per_window(wa)
                probs = []
                for i in range(0, n_min, test_bs):
                    be = torch.FloatTensor(we[i:i+test_bs]).to(device)
                    ba = torch.FloatTensor(wa[i:i+test_bs]).to(device)
                    logits, _ = model(eeg_x=be, audio_x=ba, mode=mode)
                    probs.append(torch.sigmoid(logits).cpu().numpy())
                p = np.concatenate(probs)
            test_preds.append(float(p.mean()))
            test_true.append(lbl)
            test_sids.append(sid)
    y_true = np.array(test_true)
    y_prob = np.array(test_preds)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


# ── Main training loop ──────────────────────────────────────────

def train(mode, eeg_subjs, aud_subjs, args):
    """5-fold nested SGKF: pre-train backbones per fold, then train fusion."""
    eeg_ids_all, aud_ids_all, pairs = [], [], []
    if mode == 'eeg_only':
        eeg_ids_all = sorted(eeg_subjs.keys())
        n = len(eeg_ids_all)
        labels = np.array([eeg_subjs[s]['label'] for s in eeg_ids_all])
        groups = np.array(eeg_ids_all)
    elif mode == 'audio_only':
        aud_ids_all = sorted(aud_subjs.keys())
        n = len(aud_ids_all)
        labels = np.array([aud_subjs[s]['label'] for s in aud_ids_all])
        groups = np.array(aud_ids_all)
    else:
        pairs = load_multimodal_pairs(eeg_subjs, aud_subjs)
        n = len(pairs)
        labels = np.array([p[2] for p in pairs])
        groups = np.array([f'p{i}' for i in range(n)])
        print(f'  Multimodal pairs: {n} ({int(labels.sum())} MDD, {n - int(labels.sum())} HC)')

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_results, all_histories = [], []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(n), labels, groups=groups)):
        try:
            print(f'\n  Fold {fi + 1}: pre-training backbones...')

            # ── Pre-train EEG backbone ──────────────────────────
            eeg_Xtr, eeg_ytr = _stack_windows('eeg_only', tvi, eeg_ids_all,
                                               aud_ids_all, pairs, eeg_subjs, aud_subjs)
            eeg_bb = _pretrain_backbone_tensors(
                eeg_Xtr, eeg_ytr, DeepConvNet, 64, 500, args, label=f'EEG F{fi+1}')

            # ── Pre-train Audio backbone ────────────────────────
            aud_Xtr, aud_ytr = _stack_windows('audio_only', tvi, eeg_ids_all,
                                               aud_ids_all, pairs, eeg_subjs, aud_subjs)
            aud_bb = _pretrain_backbone_tensors(
                aud_Xtr, aud_ytr, ShallowConvNet, 64, 200, args, label=f'AUD F{fi+1}')

            # ── Inner train/val split for fusion ────────────────
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + fi)
            tr_i, vl_i = next(inner.split(np.zeros(len(tvi)), labels[tvi], groups=groups[tvi]))
            tr_idx = tvi[tr_i]; vl_idx = tvi[vl_i]

            Xtr, ytr = _stack_windows(mode, tr_idx, eeg_ids_all, aud_ids_all,
                                       pairs, eeg_subjs, aud_subjs)
            Xvl, yvl = _stack_windows(mode, vl_idx, eeg_ids_all, aud_ids_all,
                                       pairs, eeg_subjs, aud_subjs)

            tr_ldr = _make_loader(_zscore_per_window(Xtr), ytr, args.bs, shuffle=True)
            vl_ldr = _make_loader(_zscore_per_window(Xvl), yvl, args.bs, shuffle=False)

            # ── Train fusion ────────────────────────────────────
            model = CrossModalDL(eeg_bb, aud_bb, mode=mode,
                                 dropout=args.dropout, freeze_backbones=True).to(device)
            logger = ClassificationLogger()
            if fi == 0:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_p = sum(p.numel() for p in model.parameters())
                print(f'  Params: {total_p:,} total, {trainable:,} trainable')

            best_st, best_vb, history = _train_fusion_fold(model, mode, tr_ldr, vl_ldr, args, logger)
            model.load_state_dict(best_st)

            y_true_s, y_pred_s, y_prob_s = _evaluate_subject_level(
                model, mode, eeg_subjs, aud_subjs, pairs,
                eeg_ids_all, aud_ids_all, tei, test_bs=args.bs)

            cm = confusion_matrix(y_true_s, y_pred_s).tolist()
            roc_auc = float(roc_auc_score(y_true_s, y_prob_s))
            roc_data = {'y_true': y_true_s.tolist(), 'y_prob': y_prob_s.tolist()}
            fm = logger.log_fold_test(y_true_s, y_pred_s)
            fold_results.append({'fold': fi + 1, 'best_val_bacc': float(best_vb),
                                  'n_epochs': len(history['train_loss']),
                                  'test_metrics': fm, 'history': history,
                                  'test_cm_subject': cm,
                                  'test_roc': roc_data, 'test_roc_auc': roc_auc})
            all_histories.append({'fold': fi + 1, 'history': history})
            print(f'  Fold {fi + 1}: bacc={fm["bacc"]:.3f}')
        except Exception as e:
            print(f'  Fold {fi + 1} FAILED: {e}')
            import traceback
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if fold_results:
        logger.log_summary(n_folds=5)
        baccs = [r['test_metrics']['bacc'] for r in fold_results]
        summary = {'bacc_mean': float(np.mean(baccs)), 'bacc_std': float(np.std(baccs))}
        print(f'\n{mode}: bacc={summary["bacc_mean"]:.3f} +/- {summary["bacc_std"]:.3f}')

        out_results = {
            'mode': mode, 'model': 'CAMP-Net (nested pre-training)',
            'args': vars(args), 'folds': fold_results, 'summary': summary,
        }
        out_path = os.path.join(OUTPUT_DIR, f'campnet_{mode}.json')
        with open(out_path, 'w') as f:
            json.dump(out_results, f, indent=2)
        print(f'Saved: {out_path}')
    else:
        print('\nAll folds failed — no results saved')


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train CAMP-Net (nested backbone pre-training)')
    parser.add_argument('--mode', required=True,
                        choices=['eeg_only', 'audio_only', 'early_fusion',
                                 'late_fusion', 'cross_attn'])
    parser.add_argument('--epochs', type=int, default=20,
                        help='Fusion training epochs per fold')
    parser.add_argument('--backbone-epochs', type=int, default=15,
                        help='Backbone pre-training epochs per fold')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for fusion training')
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    print(f'Device: {device}')
    print(f'{"=" * 55}')
    print(f'  CAMP-Net — {args.mode}')
    print(f'  Nested protocol: backbone pre-training per fold')
    print(f'  Backbone epochs: {args.backbone_epochs}  Fusion epochs: {args.epochs}')
    print(f'  LR={args.lr}  WD={args.wd}  BS={args.bs}  Drop={args.dropout}')
    print(f'{"=" * 55}')

    print('\nLoading cached data...')
    eeg_subjs, _, _ = load_cached(EEG_CACHE)
    aud_subjs, _, _ = load_cached(AUDIO_CACHE)
    print(f'  EEG: {len(eeg_subjs)} subjects')
    print(f'  Audio: {len(aud_subjs)} subjects')

    train(args.mode, eeg_subjs, aud_subjs, args)


if __name__ == '__main__':
    main()

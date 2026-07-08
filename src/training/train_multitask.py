"""
Multitask end-to-end: classification + contrastive regularisation.

  L = BCE(y, ŷ) + λ · NT-Xent(z_eeg, z_aud)

All modules trained jointly from scratch on raw windows.
No pretrained backbones, no separate phases.

Usage:
  py src/training/train_multitask.py
"""
import sys
import os
import json
import copy
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.models.latent_projection import LatentProjection

OUTPUT_DIR = 'outputs/results/multitask'
SEED = 42
N_FOLDS = 5
N_CHANNELS = 64
PROJ_DIM = 128
HIDDEN = 64
LR = 0.0005
WD = 0.001
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 8
K_WINDOWS = 32
MAX_WINDOWS = 50
LAMBDA_INIT = 0.1


# ── Data ──────────────────────────────────────────────────────────────────

def _load_cache(npz_path):
    c = np.load(npz_path, allow_pickle=True)
    wins = c['windows']
    ids = [str(s) for s in c['subject_ids']]
    labels = [int(l) for l in c['labels']]
    has_mask = 'window_mask' in c
    subjects = {}
    for i, sid in enumerate(ids):
        if has_mask:
            mask = c['window_mask'][i]
            w = wins[i][mask]
        else:
            w = wins[i]
        subjects[sid] = {'windows': w, 'label': labels[i]}
    return subjects


def _load_multimodal_pairs(eeg_subjs, aud_subjs):
    with open('data/processed/multimodal_mapping.json') as f:
        mapping = json.load(f)
    pairs = []
    for aud_id, eeg_id in mapping['orig_to_bids'].items():
        if eeg_id in eeg_subjs and aud_id in aud_subjs:
            pairs.append((eeg_id, aud_id, eeg_subjs[eeg_id]['label']))
    return pairs


def _zscore(w):
    return (w - w.mean()) / (w.std() + 1e-8)


def _select_windows_deterministic(windows, max_wins):
    n = windows.shape[0]
    if n <= max_wins:
        return windows
    indices = np.linspace(0, n - 1, max_wins, dtype=int)
    return windows[indices]


# ── Encoder helpers (conv blocks only) ────────────────────────────────────

def _encode_eeg(model, windows, dev):
    K = windows.shape[0]
    out = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(dev)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        x = model.block1(batch)
        x = model.block2(x)
        x = model.block3(x)
        x = model.block4(x)
        out.append(x.flatten(start_dim=1))
    return torch.cat(out, dim=0)


def _encode_audio(model, windows, dev):
    K = windows.shape[0]
    out = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(dev)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        x = model.temporal_conv(batch)
        x = model.spatial_conv(x)
        x = model.bn(x)
        x = torch.square(x)
        x = model.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))
        x = model.dropout(x)
        out.append(x.flatten(start_dim=1))
    return torch.cat(out, dim=0)


# ── Loss ──────────────────────────────────────────────────────────────────

def nt_xent_loss(z_eeg, z_aud, logit_scale):
    B = z_eeg.shape[0]
    logits = (z_eeg @ z_aud.T) * logit_scale
    labels = torch.arange(B, device=z_eeg.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── Model ─────────────────────────────────────────────────────────────────

class MultitaskModel(nn.Module):
    def __init__(self, n_channels, eeg_dim=128, aud_dim=576,
                 proj_dim=PROJ_DIM, hidden=HIDDEN, logit_scale_init=0.07):
        super().__init__()
        self.eeg_encoder = DeepConvNet(n_channels, 1, 500, 0.5)
        self.aud_encoder = ShallowConvNet(64, 1, 200, 0.5)
        self.proj_eeg = LatentProjection(eeg_dim, proj_dim)
        self.proj_aud = LatentProjection(aud_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / logit_scale_init)))
        self.classifier = nn.Sequential(
            nn.Linear(2 * proj_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward_subject(self, eeg_wins, aud_wins, dev):
        """K windows → mean → (proj_e, proj_a, logit)."""
        ze = _encode_eeg(self.eeg_encoder, eeg_wins, dev)
        za = _encode_audio(self.aud_encoder, aud_wins, dev)
        raw_e = ze.mean(dim=0, keepdim=True)
        raw_a = za.mean(dim=0, keepdim=True)
        pe = self.proj_eeg(raw_e)
        pa = self.proj_aud(raw_a)
        logit = self.classifier(torch.cat([pe, pa], dim=1))
        return pe, pa, logit


# ── Meter ─────────────────────────────────────────────────────────────────

class AvgMeter:
    def __init__(self): self.reset()
    def reset(self): self.vals = []
    def add(self, v): self.vals.append(v)
    @property
    def avg(self): return float(np.mean(self.vals)) if self.vals else 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Multitask: BCE + NT-Xent')
    parser.add_argument('--lambda-cls', type=float, default=LAMBDA_INIT)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    args = parser.parse_args()

    lam = args.lambda_cls
    out_name = f'multitask_64ch_b{BATCH_SIZE}_k{K_WINDOWS}_lam{lam}'
    out_dir = os.path.join(OUTPUT_DIR, out_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'Multitask — {out_name}')
    print(f'  L = BCE + {lam}·NT-Xent')
    print(f'  Batch={BATCH_SIZE}  K={K_WINDOWS}  lr={args.lr}')
    print(f'  epochs={args.epochs}  patience={PATIENCE}')

    # Data
    eeg_subjs = _load_cache('data/processed/eeg_preprocessed_64ch.npz')
    aud_subjs = _load_cache('data/processed/audio_mel_cache.npz')
    pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
    labels = np.array([p[2] for p in pairs])
    groups = np.array([f'p{i}' for i in range(len(pairs))])
    n_pairs, n_mdd = len(pairs), int(labels.sum())
    print(f'  Multimodal pairs: {n_pairs} ({n_mdd} MDD, {n_pairs-n_mdd} HC)')

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(n_pairs), labels, groups=groups)):
        print(f'\n{"="*55}')
        print(f'  FOLD {fi+1}/{N_FOLDS}')
        print(f'{"="*55}')

        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED+fi)
        tr_i, vl_i = next(inner.split(np.zeros(len(tvi)), labels[tvi], groups=groups[tvi]))
        tr_idx, vl_idx = tvi[tr_i], tvi[vl_i]
        tr_pairs = [pairs[i] for i in tr_idx]
        vl_pairs = [pairs[i] for i in vl_idx]
        te_pairs = [pairs[i] for i in tei]
        print(f'  Train: {len(tr_pairs)}  Val: {len(vl_pairs)}  Test: {len(te_pairs)}')

        model = MultitaskModel(N_CHANNELS).to(device)
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=WD, foreach=False)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
        crit = nn.BCEWithLogitsLoss()

        best_vb, best_st, pat = 0.0, None, 0
        print(f'{"Ep":>4s}  {"L_bce":>8s}  {"L_con":>8s}  {"L_tot":>8s}  '
              f'{"V_bacc":>8s}  {"pos":>8s}  {"neg":>8s}')

        for ep in range(1, args.epochs + 1):
            model.train()
            meters = {'bce': AvgMeter(), 'con': AvgMeter(), 'tot': AvgMeter(),
                      'pos': AvgMeter(), 'neg': AvgMeter()}
            random.shuffle(tr_pairs)
            batch_pe, batch_pa, batch_le = [], [], []
            opt.zero_grad()

            for eid, aid, lbl in tr_pairs:
                we = eeg_subjs[eid]['windows']
                wa = aud_subjs[aid]['windows']
                we = _select_windows_deterministic(we, K_WINDOWS)
                wa = _select_windows_deterministic(wa, K_WINDOWS)
                K = min(len(we), len(wa))
                we, wa = we[:K], wa[:K]
                we = np.array([_zscore(we[i]) for i in range(K)])
                wa = np.array([_zscore(wa[i]) for i in range(K)])

                # Forward ensures mini-batches of 32 inside the encoder
                pe, pa, _ = model.forward_subject(we, wa, device)
                batch_pe.append(pe)
                batch_pa.append(pa)
                batch_le.append(lbl)

                if len(batch_pe) >= BATCH_SIZE:
                    pe_cat = torch.cat(batch_pe, dim=0)      # [B, D]
                    pa_cat = torch.cat(batch_pa, dim=0)      # [B, D]
                    y_smooth = torch.tensor(batch_le, device=device).float() * 0.95 + 0.025

                    # BCE: re-forward classifier on stacked projections
                    logits = model.classifier(
                        torch.cat([pe_cat, pa_cat], dim=1))  # [B, 1]
                    l_bce = crit(logits, y_smooth.unsqueeze(1))
                    meters['bce'].add(l_bce.item())

                    l_con = nt_xent_loss(pe_cat, pa_cat, model.logit_scale.exp())
                    meters['con'].add(l_con.item())

                    l_tot = l_bce + lam * l_con
                    meters['tot'].add(l_tot.item())
                    l_tot.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad()

                    with torch.no_grad():
                        sim = pe_cat @ pa_cat.T
                        meters['pos'].add(sim.diag().mean().item())
                        meters['neg'].add(
                            sim[~torch.eye(BATCH_SIZE, dtype=bool, device=device)].mean().item())

                    batch_pe, batch_pa, batch_le = [], [], []

            # Validation
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for eid, aid, lbl in vl_pairs:
                    we = eeg_subjs[eid]['windows']
                    wa = aud_subjs[aid]['windows']
                    we = _select_windows_deterministic(we, MAX_WINDOWS)
                    wa = _select_windows_deterministic(wa, MAX_WINDOWS)
                    K = min(len(we), len(wa))
                    we, wa = we[:K], wa[:K]
                    we = np.array([_zscore(we[i]) for i in range(K)])
                    wa = np.array([_zscore(wa[i]) for i in range(K)])
                    _, _, logit = model.forward_subject(we, wa, device)
                    prob = float(torch.sigmoid(logit).cpu().item())
                    y_true.append(lbl)
                    y_pred.append(int(prob >= 0.5))

            vb = balanced_accuracy_score(y_true, y_pred)
            sched.step(vb)

            if ep == 1 or ep % 10 == 0 or vb > best_vb:
                print(f'{ep:4d}  {meters["bce"].avg:8.4f}  {meters["con"].avg:8.4f}  '
                      f'{meters["tot"].avg:8.4f}  {vb:8.4f}  '
                      f'{meters["pos"].avg:8.4f}  {meters["neg"].avg:8.4f}')

            if vb > best_vb:
                best_vb = vb
                best_st = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    print(f'  Early stopping at epoch {ep}')
                    break

        # Restore best & evaluate on test
        model.load_state_dict(best_st)
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for eid, aid, lbl in te_pairs:
                we = eeg_subjs[eid]['windows']
                wa = aud_subjs[aid]['windows']
                we = _select_windows_deterministic(we, MAX_WINDOWS)
                wa = _select_windows_deterministic(wa, MAX_WINDOWS)
                K = min(len(we), len(wa))
                we, wa = we[:K], wa[:K]
                we = np.array([_zscore(we[i]) for i in range(K)])
                wa = np.array([_zscore(wa[i]) for i in range(K)])
                _, _, logit = model.forward_subject(we, wa, device)
                prob = float(torch.sigmoid(logit).cpu().item())
                y_true.append(lbl)
                y_pred.append(int(prob >= 0.5))

        test_bacc = balanced_accuracy_score(y_true, y_pred)
        print(f'  >>> test bacc = {test_bacc:.4f}')
        fold_results.append({
            'fold': fi + 1, 'val_bacc': float(best_vb), 'test_bacc': float(test_bacc)})

        # Save checkpoint
        ckpt_dir = os.path.join(out_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            'fold': fi + 1, 'test_bacc': float(test_bacc),
            'model_state_dict': best_st,
        }, os.path.join(ckpt_dir, f'fold_{fi+1}.pt'))

    # Summary
    baccs = [r['test_bacc'] for r in fold_results]
    print(f'\n{"="*55}')
    print(f'  {out_name}')
    print(f'  test bacc = {np.mean(baccs):.3f} ± {np.std(baccs):.3f}')
    print(f'{"="*55}')

    out = {
        'config': {
            'batch_size': BATCH_SIZE, 'k_windows': K_WINDOWS, 'lambda': lam,
            'lr': args.lr, 'epochs': args.epochs, 'patience': PATIENCE,
        },
        'folds': fold_results,
        'summary': {
            'bacc_mean': float(np.mean(baccs)), 'bacc_std': float(np.std(baccs)),
        },
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {os.path.join(out_dir, "results.json")}')


if __name__ == '__main__':
    main()

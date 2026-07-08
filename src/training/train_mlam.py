"""
MLAM — Multimodal Latent Alignment Module (Fase 1).

Contrastive pretraining: align EEG and audio representations
in a shared latent space using symmetric NT-Xent loss.

Usage:
  py src/training/train_mlam.py [--config src/configs/config_mlam.yaml]
"""
import sys, os, json, yaml, argparse, copy, random, warnings
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.models.latent_projection import LatentProjection
from src.utils.training_logger import ClassificationLogger

MAPPING_PATH = 'data/processed/multimodal_mapping.json'
OUTPUT_DIR = 'outputs/results/mlam'
N_FOLDS = 5

# ── Data loading (shared with run_crossmodal) ─────────────────────────────

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
    return subjects, ids


def _load_multimodal_pairs(eeg_subjs, aud_subjs):
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    pairs = []
    for aud_id, eeg_id in mapping['orig_to_bids'].items():
        if eeg_id in eeg_subjs and aud_id in aud_subjs:
            pairs.append((eeg_id, aud_id, eeg_subjs[eeg_id]['label']))
    return pairs


def _zscore(w):
    return (w - w.mean()) / (w.std() + 1e-8)


def _select_windows_deterministic(windows, max_windows):
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    indices = np.linspace(0, n - 1, max_windows, dtype=int)
    return windows[indices]


# ── Contrastive loss ──────────────────────────────────────────────────────

def nt_xent_loss(z_eeg, z_aud, logit_scale):
    """Symmetric NT-Xent loss (CLIP-style).

    Args:
        z_eeg: [B, D] normalized EEG embeddings
        z_aud: [B, D] normalized audio embeddings
        logit_scale: scalar (exp'd)
    Returns:
        loss, logits (for logging)
    """
    B = z_eeg.shape[0]
    logits = (z_eeg @ z_aud.T) * logit_scale
    labels = torch.arange(B, device=z_eeg.device)

    loss_e2a = F.cross_entropy(logits, labels)
    loss_a2e = F.cross_entropy(logits.T, labels)
    return (loss_e2a + loss_a2e) / 2, logits


# ── Retrieval validation ─────────────────────────────────────────────────

@torch.no_grad()
def compute_retrieval_top1(eeg_embeds, aud_embeds, subj_ids_eeg, subj_ids_aud):
    """For each EEG embedding, find nearest audio embedding (cosine sim).
    Returns: top-1 accuracy (fraction where best match is same subject)."""
    sim = eeg_embeds @ aud_embeds.T
    top1_idx = sim.argmax(dim=1).cpu().numpy()
    correct = 0
    for i, idx in enumerate(top1_idx):
        if subj_ids_eeg[i] == subj_ids_aud[idx]:
            correct += 1
    return correct / len(subj_ids_eeg)


# ── Forward helpers (with gradients) ──────────────────────────────────────

def _encode_eeg(model, windows, dev):
    K = windows.shape[0]
    feats = []
    for i in range(0, K, 32):
        batch = torch.from_numpy(windows[i:i+32]).float().to(dev)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        x = model.block1(batch)
        x = model.block2(x)
        x = model.block3(x)
        x = model.block4(x)
        feats.append(x.flatten(start_dim=1))
    return torch.cat(feats, dim=0)


def _encode_audio(model, windows, dev):
    K = windows.shape[0]
    feats = []
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
        feats.append(x.flatten(start_dim=1))
    return torch.cat(feats, dim=0)


# ── MLAM Model ────────────────────────────────────────────────────────────

class MLAM(nn.Module):
    """Encoders + Projection heads + logit scale."""

    def __init__(self, n_channels, proj_dim=128, logit_scale_init=0.07):
        super().__init__()
        self.eeg_encoder = DeepConvNet(n_channels, 1, 500, 0.5)
        self.aud_encoder = ShallowConvNet(64, 1, 200, 0.5)
        self.proj_eeg = LatentProjection(128, proj_dim)
        self.proj_aud = LatentProjection(576, proj_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / logit_scale_init)))
        self.cls_head = nn.Linear(proj_dim, 1)

    def encode_eeg(self, windows, dev):
        raw = _encode_eeg(self.eeg_encoder, windows, dev)
        return self.proj_eeg(raw)

    def encode_aud(self, windows, dev):
        raw = _encode_audio(self.aud_encoder, windows, dev)
        return self.proj_aud(raw)

    def forward(self, eeg_wins, aud_wins, dev):
        ze = self.encode_eeg(eeg_wins, dev)
        za = self.encode_aud(aud_wins, dev)
        return ze, za


# ── Epoch logging ─────────────────────────────────────────────────────────

class AvgMeter:
    def __init__(self): self.reset()
    def reset(self): self.vals = []
    def add(self, v): self.vals.append(v)
    @property
    def avg(self): return float(np.mean(self.vals)) if self.vals else 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MLAM — Multimodal Latent Alignment Module')
    parser.add_argument('--config', type=str, default='configs/config_mlam.yaml')
    parser.add_argument('--channels', type=str, default='64',
                        help='EEG channel selection: 64, 128, 19, or ftsm4|8|16|32|64')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--proj-dim', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.proj_dim is not None:
        cfg['proj_dim'] = args.proj_dim
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.channels != '64':
        cfg['eeg_cache'] = f'data/processed/eeg_preprocessed_{args.channels}.npz'

    # Parse n_channels
    v = args.channels.lower()
    if v in ('128',):
        n_channels = 128
    elif v == '64':
        n_channels = 64
    elif v == '19':
        n_channels = 19
    elif v.startswith('ftsm'):
        n_channels = int(v.replace('ftsm', ''))
    else:
        n_channels = int(v)

    proj_dim = cfg['proj_dim']
    batch_size = cfg.get('batch_size', 8)
    out_name = f'mlam_{args.channels}ch_d{proj_dim}_b{batch_size}'
    out_dir = os.path.join(OUTPUT_DIR, out_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'MLAM — {out_name}')
    print(f'  EEG cache: {cfg["eeg_cache"]}  ({n_channels} channels)')
    print(f'  Audio cache: {cfg["audio_cache"]}')
    print(f'  proj_dim={proj_dim}  batch_size={batch_size}  lr={cfg["lr"]}  λ_cls={cfg["lambda_cls"]}')
    print(f'  τ_init={cfg["logit_scale_init"]}  epochs={cfg["epochs"]}  patience={cfg["patience"]}')

    # Load data
    eeg_subjs, _ = _load_cache(cfg['eeg_cache'])
    aud_subjs, _ = _load_cache(cfg['audio_cache'])
    pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
    labels = np.array([p[2] for p in pairs])
    groups = np.array([f'p{i}' for i in range(len(pairs))])
    n_pairs = len(pairs)
    n_mdd = int(labels.sum())
    print(f'  Multimodal pairs: {n_pairs} ({n_mdd} MDD, {n_pairs - n_mdd} HC)')

    # Nested CV
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg['seed'])
    fold_results = []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(n_pairs), labels, groups=groups)):
        print(f'\n{"=" * 55}')
        print(f'  FOLD {fi + 1}/{N_FOLDS}')
        print(f'{"=" * 55}')

        # Inner split
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg['seed'] + fi)
        tr_i, vl_i = next(inner.split(np.zeros(len(tvi)), labels[tvi], groups=groups[tvi]))
        tr_idx, vl_idx = tvi[tr_i], tvi[vl_i]

        tr_pairs = [pairs[i] for i in tr_idx]
        vl_pairs = [pairs[i] for i in vl_idx]
        te_pairs = [pairs[i] for i in tei]

        print(f'  Train: {len(tr_pairs)}  Val: {len(vl_pairs)}  Test: {len(te_pairs)}')

        # Model
        model = MLAM(n_channels, proj_dim, cfg['logit_scale_init']).to(device)

        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'], foreach=False)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=5)
        crit_cls = nn.BCEWithLogitsLoss()
        max_wins = cfg['max_windows']

        best_val_retrieval, best_st, pat = 0.0, None, 0
        log_header = f'{"Ep":>4s}  {"L_con":>8s}  {"L_cls":>8s}  {"L_tot":>8s}  '
        log_header += f'{"pos_cos":>8s}  {"neg_cos":>8s}  {"val_ret":>8s}  {"scale":>8s}'
        print(log_header)

        for ep in range(1, cfg['epochs'] + 1):
            model.train()
            meters = {'l_con': AvgMeter(), 'l_cls': AvgMeter(), 'l_tot': AvgMeter(),
                      'pos': AvgMeter(), 'neg': AvgMeter()}

            # Batch accumulation: collect batch_size pairs, stack, NT-Xent on [B, D]
            random.shuffle(tr_pairs)
            batch_ze, batch_za, batch_lbls = [], [], []
            opt.zero_grad()

            for eid, aid, lbl in tr_pairs:
                we = eeg_subjs[eid]['windows']
                wa = aud_subjs[aid]['windows']
                kw = min(len(we), len(wa), max_wins)
                if kw < 1:
                    continue
                idx = np.random.randint(kw)
                we_i = _zscore(we[idx])[np.newaxis, ...].astype(np.float32)
                wa_i = _zscore(wa[idx])[np.newaxis, ...].astype(np.float32)

                ze, za = model(we_i, wa_i, device)
                batch_ze.append(ze)
                batch_za.append(za)
                batch_lbls.append(lbl)

                if len(batch_ze) >= batch_size:
                    ze_cat = torch.cat(batch_ze, dim=0)
                    za_cat = torch.cat(batch_za, dim=0)
                    B_act = ze_cat.shape[0]

                    # NT-Xent over the batch
                    l_con, logits = nt_xent_loss(ze_cat, za_cat, model.logit_scale.exp())
                    meters['l_con'].add(l_con.item())

                    # Aux classification
                    z_avg = (ze_cat + za_cat) / 2
                    logits_cls = model.cls_head(z_avg).squeeze(1)
                    y_smooth = torch.tensor(batch_lbls, device=device).float() * 0.95 + 0.025
                    l_cls = crit_cls(logits_cls, y_smooth)
                    meters['l_cls'].add(l_cls.item())

                    l_tot = l_con + cfg['lambda_cls'] * l_cls
                    meters['l_tot'].add(l_tot.item())
                    l_tot.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
                    opt.step()
                    opt.zero_grad()

                    # Cosine sim logging
                    with torch.no_grad():
                        sim_all = ze_cat @ za_cat.T
                        meters['pos'].add(sim_all.diag().mean().item())
                        meters['neg'].add(sim_all[~torch.eye(B_act, dtype=bool, device=device)].mean().item())

                    batch_ze, batch_za, batch_lbls = [], [], []

            # Validation: retrieval Top-1
            model.eval()
            all_ze, all_za, vids_e, vids_a = [], [], [], []
            pair_idx = 0
            with torch.no_grad():
                for eid, aid, _ in vl_pairs:
                    we = eeg_subjs[eid]['windows']
                    wa = aud_subjs[aid]['windows']
                    we = _select_windows_deterministic(we, max_wins)
                    wa = _select_windows_deterministic(wa, max_wins)
                    K = min(len(we), len(wa))
                    we, wa = we[:K], wa[:K]
                    we = np.array([_zscore(we[i]) for i in range(K)])
                    wa = np.array([_zscore(wa[i]) for i in range(K)])

                    ze = model.encode_eeg(we, device).cpu()
                    za = model.encode_aud(wa, device).cpu()
                    all_ze.append(ze)
                    all_za.append(za)
                    vids_e.extend([pair_idx] * K)
                    vids_a.extend([pair_idx] * K)
                    pair_idx += 1

            all_ze = torch.cat(all_ze, dim=0)
            all_za = torch.cat(all_za, dim=0)
            val_ret = compute_retrieval_top1(all_ze, all_za, vids_e, vids_a)
            sched.step(val_ret)

            scale_val = model.logit_scale.exp().item()
            print(f'{ep:4d}  {meters["l_con"].avg:8.4f}  {meters["l_cls"].avg:8.4f}  '
                  f'{meters["l_tot"].avg:8.4f}  {meters["pos"].avg:8.4f}  '
                  f'{meters["neg"].avg:8.4f}  {val_ret:8.4f}  {scale_val:8.4f}')

            # Early stopping
            if val_ret > best_val_retrieval:
                best_val_retrieval = val_ret
                best_st = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1

            if pat >= cfg['patience']:
                print(f'  Early stopping at epoch {ep}')
                break

        if best_st is not None:
            model.load_state_dict(best_st)
        fold_results.append({'fold': fi + 1, 'best_val_retrieval': float(best_val_retrieval)})

        # Save checkpoint
        torch.save({
            'fold': fi + 1, 'best_val_retrieval': float(best_val_retrieval),
            'model_state_dict': model.state_dict(),
            'args': vars(args),
        }, os.path.join(out_dir, f'fold_{fi + 1}.pt'))
        print(f'  Saved: fold_{fi + 1}.pt (val_retrieval={best_val_retrieval:.4f})')

    # Summary
    rets = [r['best_val_retrieval'] for r in fold_results]
    print(f'\n{"=" * 55}')
    print(f'  {out_name}')
    print(f'  Retrieval Top-1: {np.mean(rets):.3f} ± {np.std(rets):.3f}')
    print(f'{"=" * 55}')

    out_results = {
        'config_name': out_name,
        'config': cfg,
        'args': vars(args),
        'folds': fold_results,
        'summary': {'retrieval_mean': float(np.mean(rets)),
                     'retrieval_std': float(np.std(rets))},
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(out_results, f, indent=2)
    print(f'Saved: {os.path.join(out_dir, "results.json")}')


if __name__ == '__main__':
    main()

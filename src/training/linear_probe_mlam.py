"""
Linear probe on frozen MLAM features (Fase 1.5).

Evaluates whether MLAM-pretrained encoders produce useful
representations for MDD classification.

Usage:
  py src/training/linear_probe_mlam.py
"""
import sys
import os
import json
import yaml
import warnings
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.training.train_mlam import MLAM

CONFIG_PATH = 'configs/config_mlam.yaml'
MLAM_CKPT_DIR = 'outputs/results/mlam/mlam_64ch_d128_b8'
OUTPUT_DIR = 'outputs/results/mlam/linear_probe_64ch_d128_b8'
N_FOLDS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    with open('data/processed/multimodal_mapping.json') as f:
        mapping = json.load(f)
    pairs = []
    for aud_id, eeg_id in mapping['orig_to_bids'].items():
        if eeg_id in eeg_subjs and aud_id in aud_subjs:
            pairs.append((eeg_id, aud_id, eeg_subjs[eeg_id]['label']))
    return pairs


def _zscore(w):
    return (w - w.mean()) / (w.std() + 1e-8)


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


def extract_subject_embeddings(model, eeg_subjs, aud_subjs, pairs, max_wins=50):
    """Mean-pool window embeddings per subject → [N_subj, proj_dim*2]."""
    model.eval()
    X, y = [], []
    with torch.no_grad():
        for eid, aid, lbl in pairs:
            we = eeg_subjs[eid]['windows']
            wa = aud_subjs[aid]['windows']
            # Use all windows (up to max_wins for speed)
            if max_wins and len(we) > max_wins:
                idx = np.linspace(0, len(we)-1, max_wins, dtype=int)
                we = we[idx]
            if max_wins and len(wa) > max_wins:
                idx = np.linspace(0, len(wa)-1, max_wins, dtype=int)
                wa = wa[idx]
            K = min(len(we), len(wa))
            we, wa = we[:K], wa[:K]
            we = np.array([_zscore(we[i]) for i in range(K)])
            wa = np.array([_zscore(wa[i]) for i in range(K)])

            ze = _encode_eeg(model.eeg_encoder, we, device)
            za = _encode_audio(model.aud_encoder, wa, device)
            ze = model.proj_eeg(ze).cpu().numpy()  # [K, D]
            za = model.proj_aud(za).cpu().numpy()  # [K, D]

            # Mean pool windows → [D] per modality
            z_subj = np.concatenate([ze.mean(0), za.mean(0)])  # [D*2]
            X.append(z_subj)
            y.append(lbl)
    return np.array(X), np.array(y)


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Device: {device}')
    print('Linear probe on frozen MLAM features')

    # Load data
    eeg_subjs, _ = _load_cache(cfg['eeg_cache'])
    aud_subjs, _ = _load_cache(cfg['audio_cache'])
    pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
    labels = np.array([p[2] for p in pairs])
    n_mdd = int(labels.sum())
    print(f'  Multimodal pairs: {len(pairs)} ({n_mdd} MDD, {len(pairs)-n_mdd} HC)')

    # Split
    groups = np.array([f'p{i}' for i in range(len(pairs))])
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg['seed'])
    fold_results = []

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(pairs)), labels, groups=groups)):
        print(f'\n{"="*55}')
        print(f'  FOLD {fi+1}/{N_FOLDS}')
        print(f'{"="*55}')

        # Load MLAM checkpoint
        ckpt_path = os.path.join(MLAM_CKPT_DIR, f'fold_{fi+1}.pt')
        if not os.path.exists(ckpt_path):
            print(f'  SKIP: {ckpt_path} not found')
            continue

        n_channels = 64
        proj_dim = cfg['proj_dim']
        model = MLAM(n_channels, proj_dim, cfg['logit_scale_init']).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        print(f'  Loaded: fold_{fi+1}.pt (val_retrieval={state["best_val_retrieval"]:.3f})')

        # Extract frozen features
        X, y = extract_subject_embeddings(
            model, eeg_subjs, aud_subjs, pairs, cfg['max_windows'])

        # Split
        X_tr, X_te = X[tvi], X[tei]
        y_tr, y_te = y[tvi], y[tei]

        # Train linear classifier
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        bacc = balanced_accuracy_score(y_te, y_pred)
        print(f'  Test bacc = {bacc:.4f}')

        fold_results.append({'fold': fi+1, 'bacc': float(bacc)})

    if not fold_results:
        print('\nNo folds completed.')
        return

    baccs = [r['bacc'] for r in fold_results]
    print(f'\n{"="*55}')
    print('  Linear probe MLAM (frozen)')
    print(f'  bacc = {np.mean(baccs):.3f} ± {np.std(baccs):.3f}')
    fold_strs = ' '.join(str(round(r['bacc'], 3)) for r in fold_results)
    print('  folds: [' + fold_strs + ']')
    print(f'{"="*55}')

    # Gate check
    mean_bacc = float(np.mean(baccs))
    gate_bacc = cfg['linear_probe_min_bacc']
    print(f'\n  Gate check: linear_probe_min_bacc = {gate_bacc}')
    if mean_bacc >= gate_bacc:
        print(f'  ✅ PASS (bacc={mean_bacc:.3f} >= {gate_bacc})')
    else:
        print(f'  ❌ FAIL (bacc={mean_bacc:.3f} < {gate_bacc})')

    out = {
        'config': cfg,
        'folds': fold_results,
        'summary': {
            'bacc_mean': mean_bacc,
            'bacc_std': float(np.std(baccs)),
            'gate_bacc': gate_bacc,
            'gate_passed': mean_bacc >= gate_bacc,
        },
    }
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {os.path.join(OUTPUT_DIR, "results.json")}')


if __name__ == '__main__':
    main()

"""
Stage 1: Train backbones from scratch per fold + extract features + cache to disk.
Run ONCE before the ablation grid (Stage 2).

Usage:
  py src/training/run_crossmodal_extract.py [--max-windows 50] [--epochs 100]
"""
import sys, os, json, argparse, warnings
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '.')

from src.models.deepconvnet import DeepConvNet
from src.models.shallowconvnet import ShallowConvNet
from src.training.dl_eeg_benchmark import load_cached_eeg, train_eeg_backbone
from src.training.dl_audio_benchmark import load_cached_audio, train_audio_backbone

EEG_CACHE = 'data/processed/eeg_preprocessed_64ch.npz'
AUDIO_CACHE = 'data/processed/audio_mel_cache.npz'
MAPPING_PATH = 'data/processed/multimodal_mapping.json'
CACHE_DIR = 'cache/crossmodal_features'
N_FOLDS = 5
RANDOM_STATE = 42
N_EEG_CH = 64
N_AUDIO_MELS = 64
N_AUDIO_FRAMES = 200

os.makedirs(CACHE_DIR, exist_ok=True)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


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


def _extract_eeg(model, windows, device):
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


def _extract_subject_features(eeg_model, aud_model, eeg_subjs, aud_subjs, pairs, max_windows):
    all_ze, all_za, all_y, all_masks = [], [], [], []
    for eid, aid, lbl in pairs:
        we = eeg_subjs[eid]['windows']
        wa = aud_subjs[aid]['windows']
        we = _select_windows_deterministic(we, max_windows)
        wa = _select_windows_deterministic(wa, max_windows)
        K = min(len(we), len(wa))
        we, wa = we[:K], wa[:K]
        we = np.array([_zscore(we[i]) for i in range(len(we))])
        wa = np.array([_zscore(wa[i]) for i in range(len(wa))])
        ze = _extract_eeg(eeg_model, we, device)
        za = _extract_audio(aud_model, wa, device)
        all_ze.append(ze)
        all_za.append(za)
        all_y.append(lbl)
        all_masks.append(np.ones(K, dtype=np.float32))

    max_K = max(m.shape[0] for m in all_masks)
    N = len(pairs)
    Z_e = np.zeros((N, max_K, all_ze[0].shape[1]), dtype=np.float32)
    Z_a = np.zeros((N, max_K, all_za[0].shape[1]), dtype=np.float32)
    masks = np.zeros((N, max_K), dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32)
    for i in range(N):
        k = len(all_ze[i])
        Z_e[i, :k] = all_ze[i]
        Z_a[i, :k] = all_za[i]
        masks[i, :k] = all_masks[i]
    return Z_e, Z_a, masks, y_arr


def _unwrap_backbone(trained_model, backbone_cls, *bb_args):
    """Extract conv weights from trained wrapper model into a plain backbone."""
    bb = backbone_cls(*bb_args)
    bb_state = {}
    for k, v in trained_model.state_dict().items():
        if 'classifier' in k:
            continue
        k_clean = k[2:] if k.startswith('m.') else k
        bb_state[k_clean] = v
    bb.load_state_dict(bb_state, strict=False)
    return bb


def main():
    parser = argparse.ArgumentParser(description='Stage 1: backbone extraction + cache')
    parser.add_argument('--max-windows', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--bs', type=int, default=32)
    args = parser.parse_args()

    print(f'Device: {device}')
    print('=' * 60)
    print('Stage 1: Backbone training + feature extraction')
    print(f'  Max windows={args.max_windows}  Epochs={args.epochs}')
    print('=' * 60)

    # Load data (list form for training, dict form for extraction)
    eeg_data, eeg_labels, eeg_ids, n_samples = load_cached_eeg(N_EEG_CH)
    aud_data, aud_labels, aud_ids = load_cached_audio()
    eeg_subjs, _, _ = _load_cache(EEG_CACHE)
    aud_subjs, _, _ = _load_cache(AUDIO_CACHE)

    n_mdd_eeg = int(eeg_labels.sum())
    n_mdd_aud = int(aud_labels.sum())
    print(f'\nEEG: {len(eeg_ids)} subjects ({n_mdd_eeg} MDD)')
    print(f'Audio: {len(aud_ids)} subjects ({n_mdd_aud} MDD)')

    # Build subject-id → index maps
    eeg_id_to_idx = {str(sid): i for i, sid in enumerate(eeg_ids)}
    aud_id_to_idx = {str(sid): i for i, sid in enumerate(aud_ids)}
    assert len(eeg_id_to_idx) == len(eeg_ids)
    assert len(aud_id_to_idx) == len(aud_ids)

    # Multimodal pairs
    pairs = _load_multimodal_pairs(eeg_subjs, aud_subjs)
    labels = np.array([p[2] for p in pairs])
    group_ids = np.array([f'p{i}' for i in range(len(pairs))])
    print(f'Multimodal pairs: {len(pairs)} ({int(labels.sum())} MDD, '
          f'{len(pairs) - int(labels.sum())} HC)')

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {}

    for fi, (tvi, tei) in enumerate(skf.split(np.zeros(len(pairs)), labels, groups=group_ids)):
        print(f'\n─── Fold {fi + 1} ───')

        # Inner split train/val
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                     random_state=RANDOM_STATE + fi)
        tr_i, vl_i = next(inner.split(np.zeros(len(tvi)),
                                      labels[tvi], groups=group_ids[tvi]))
        tr_idx = tvi[tr_i]
        vl_idx = tvi[vl_i]

        # Map pair indices → real subject IDs → indices in EEG/Audio arrays
        tr_eeg_ids = [pairs[i][0] for i in tr_idx]
        tr_aud_ids = [pairs[i][1] for i in tr_idx]
        vl_eeg_ids = [pairs[i][0] for i in vl_idx]
        vl_aud_ids = [pairs[i][1] for i in vl_idx]

        for sid in tr_eeg_ids + vl_eeg_ids:
            assert sid in eeg_id_to_idx, f'EEG subject {sid} not found'
        for sid in tr_aud_ids + vl_aud_ids:
            assert sid in aud_id_to_idx, f'Audio subject {sid} not found'

        tr_eeg_i = [eeg_id_to_idx[sid] for sid in tr_eeg_ids]
        vl_eeg_i = [eeg_id_to_idx[sid] for sid in vl_eeg_ids]
        tr_aud_i = [aud_id_to_idx[sid] for sid in tr_aud_ids]
        vl_aud_i = [aud_id_to_idx[sid] for sid in vl_aud_ids]

        print(f'  Train: {len(tr_eeg_i)}  Val: {len(vl_eeg_i)}  Test: {len(tei)}')

        # ── Train EEG backbone ──
        print(f'  Training EEG backbone (DeepConvNet)...')
        eeg_model, eeg_vb = train_eeg_backbone(
            tr_eeg_i, vl_eeg_i,
            eeg_data, eeg_labels.tolist(), eeg_ids,
            n_channels=N_EEG_CH, n_samples=n_samples,
            args=args, model_key='deepconvnet')
        print(f'    EEG val bacc = {eeg_vb:.4f}')
        eeg_bb = _unwrap_backbone(eeg_model, DeepConvNet, N_EEG_CH, 1, n_samples, 0.5)
        eeg_bb.to(device).eval()

        # ── Train Audio backbone ──
        print(f'  Training Audio backbone (ShallowConvNet)...')
        aud_model, aud_vb = train_audio_backbone(
            tr_aud_i, vl_aud_i,
            aud_data, aud_labels.tolist(), aud_ids,
            args=args, model_key='shallowconvnet')
        print(f'    Audio val bacc = {aud_vb:.4f}')
        aud_bb = _unwrap_backbone(aud_model, ShallowConvNet, N_AUDIO_MELS, 1, N_AUDIO_FRAMES, 0.5)
        aud_bb.to(device).eval()

        # ── Extract features for ALL 38 subjects ──
        Z_e, Z_a, masks, y = _extract_subject_features(
            eeg_bb, aud_bb, eeg_subjs, aud_subjs, pairs, args.max_windows)
        print(f'    Features: Z_e {Z_e.shape}, Z_a {Z_a.shape}')

        # ── Save to cache (incl. inner split indices for reproducibility) ──
        subj_ids = np.array([f'{p[0]}::{p[1]}' for p in pairs], dtype=object)
        np.savez_compressed(
            os.path.join(CACHE_DIR, f'fold_{fi+1}.npz'),
            Z_e=Z_e, Z_a=Z_a, mask=masks, y=y,
            subject_ids=subj_ids,
            tr_idx=tr_idx, vl_idx=vl_idx)
        print(f'  ✓ Cached: fold_{fi+1}.npz (incl. tr_idx/vl_idx)')

        fold_metrics[fi + 1] = {
            'n_train': len(tr_eeg_i), 'n_val': len(vl_eeg_i), 'n_test': len(tei),
            'eeg_val_bacc': float(eeg_vb),
            'audio_val_bacc': float(aud_vb),
        }

        del eeg_model, eeg_bb, aud_model, aud_bb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(CACHE_DIR, 'fold_metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=2)
    print(f'\nSaved: {CACHE_DIR}/fold_metrics.json')
    print('Done. Ready for Stage 2 ablation grid.')


if __name__ == '__main__':
    main()

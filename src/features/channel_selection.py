"""MDD-Driven Channel Selection (MDD-CS) — ranking 128 EGI channels by
spectral discriminability (Cohen's d) between MDD and HC.

Also maps 10-10 positions (Prop1: 22 prefrontal, Prop2: 16 10-20) to EGI indices.

Output: data/processed/channel_selection.json
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
CACHE_PATH = 'data/processed/eeg_preprocessed_128ch.npz'
OUT_PATH = 'data/processed/channel_selection.json'

SFREQ = 250

# ── 10-20 target coordinates (from cache_modma_eeg.py) ──
_10_20_TARGETS = {
    'Fp1': (-2.7,  6.2,  1.8), 'Fp2': (2.7,  6.2,  1.8),
    'F3':  (-5.0,  4.0,  5.3), 'F4':  (5.0,  4.0,  5.3),
    'F7':  (-7.7,  3.3,  0.4), 'F8':  (7.7,  3.3,  0.4),
    'Fz':  (0.0,  2.6,  7.9),
    'C3':  (-7.7,  0.4,  3.5), 'C4':  (7.7,  0.4,  3.5),
    'Cz':  (0.0,  0.0,  8.8),
    'P3':  (-5.0, -4.0,  5.3), 'P4':  (5.0, -4.0,  5.3),
    'Pz':  (0.0, -2.6,  7.9),
    'O1':  (-2.7, -6.2,  1.8), 'O2':  (2.7, -6.2,  1.8),
    'T3':  (-8.6,  0.0,  0.4), 'T4':  (8.6,  0.0,  0.4),
    'T5':  (-7.7, -3.3,  0.4), 'T6':  (7.7, -3.3,  0.4),
}

# ── 10-10 coordinates for Prop1 (22 prefrontal, Huang Huang) ──
# Approximate positions in EGI coordinate system
_10_10_TARGETS = {
    'Fp1':  (-2.7,  6.2,  1.8), 'Fp2': (2.7,  6.2,  1.8),
    'AF3':  (-4.0,  5.5,  3.5), 'AF4': (4.0,  5.5,  3.5),
    'AF7':  (-5.0,  5.0,  1.5), 'AF8': (5.0,  5.0,  1.5),
    'F1':   (-2.5,  3.5,  6.5), 'F2':  (2.5,  3.5,  6.5),
    'F3':   (-5.0,  4.0,  5.3), 'F4':  (5.0,  4.0,  5.3),
    'F5':   (-6.5,  3.5,  3.5), 'F6':  (6.5,  3.5,  3.5),
    'F7':   (-7.7,  3.3,  0.4), 'F8':  (7.7,  3.3,  0.4),
    'Fz':   (0.0,  2.6,  7.9),
    'AFz':  (0.0,  5.0,  4.5),
    'FC1':  (-2.5,  1.5,  8.0), 'FC2': (2.5,  1.5,  8.0),
    'FC3':  (-5.5,  2.0,  6.0), 'FC4': (5.5,  2.0,  6.0),
    'FC5':  (-7.5,  1.5,  2.5), 'FC6': (7.5,  1.5,  2.5),
}

# Prop2 mapping: aliases T3→T7, T4→T8, T5→P7, T6→P8
_PROP2_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
                'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                'T3', 'T4', 'T5', 'T6']


def _load_electrode_coords():
    """Load EGI 128 electrode coordinates from first subject's electrodes.tsv."""
    sub_dirs = sorted(d for d in os.listdir(EEG_DIR) if d.startswith('sub-'))
    first_subj = sub_dirs[0]
    path = os.path.join(EEG_DIR, first_subj, 'eeg',
                        f'{first_subj}_task-Resting-state_electrodes.tsv')
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                name = parts[0].strip("'")
                if name.startswith('E'):
                    rows.append({'name': name, 'x': float(parts[1]),
                                 'y': float(parts[2]), 'z': float(parts[3])})
    df = pd.DataFrame(rows)
    return df[['x', 'y', 'z']].values  # (128, 3)


def _nearest_indices(targets, coords):
    """For each target (name → xyz), find nearest EGI index (0-based)."""
    mapping = {}
    for name, target_xyz in targets.items():
        t = np.array(target_xyz, dtype=np.float32)
        dists = np.sqrt(((coords - t) ** 2).sum(axis=1))
        idx = int(np.argmin(dists))
        mapping[name] = idx
    return mapping


def _extract_band_power(data_subj, valid_mask, sfreq=250):
    """Extract theta, alpha, beta band power for one subject.

    data_subj: (n_windows_padded, 128, 500) float32
    valid_mask: (n_windows_padded,) bool
    Returns: (128, 3) = [theta, alpha, beta] power per channel
    """
    X = data_subj[valid_mask]  # (n_valid, 128, 500)
    if len(X) == 0:
        return None

    fft = np.fft.rfft(X, axis=-1)   # (n_valid, 128, 251)
    psd = np.abs(fft) ** 2
    avg_psd = psd.mean(axis=0)       # (128, 251)
    freqs = np.fft.rfftfreq(500, 1.0 / sfreq)

    theta = avg_psd[:, (freqs >= 4) & (freqs < 8)].sum(axis=1)
    alpha = avg_psd[:, (freqs >= 8) & (freqs < 13)].sum(axis=1)
    beta  = avg_psd[:, (freqs >= 13) & (freqs < 30)].sum(axis=1)
    return np.column_stack([theta, alpha, beta])


def _cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp


def main():
    print('Loading 128ch cache...')
    c = np.load(CACHE_PATH, allow_pickle=True)
    windows = c['windows']       # (53, 335, 128, 500)
    mask = c['window_mask']      # (53, 335)
    labels = c['labels']         # (53,)
    print(f'  Shape: {windows.shape}, Subjects: {len(labels)}, '
          f'MDD: {(labels==1).sum()}, HC: {(labels==0).sum()}')

    # ── Phase 1: Spectral features ──
    print('\nExtracting band power (theta, alpha, beta) per subject...')
    n_subj = len(labels)
    band_power = np.zeros((n_subj, 128, 3), dtype=np.float64)  # (subj, ch, band)
    for i in range(n_subj):
        bp = _extract_band_power(windows[i], mask[i])
        if bp is not None:
            band_power[i] = bp
        if (i + 1) % 10 == 0:
            print(f'  Processed {i + 1}/{n_subj} subjects')
    print(f'  Done — band_power shape: {band_power.shape}')

    # Log-transform for normality
    bp_log = np.log1p(band_power)

    mdd_idx = labels == 1
    hc_idx = labels == 0
    print(f'\n  MDD subjects: {mdd_idx.sum()}, HC subjects: {hc_idx.sum()}')

    # ── Phase 2: Cohen's d per channel × band ──
    print('\nComputing Cohen\'s d per channel × band (MDD vs HC)...')
    bands = ['Theta', 'Alpha', 'Beta']
    d_matrix = np.zeros((128, 3), dtype=np.float64)  # (ch, band)
    for ch in range(128):
        for b in range(3):
            d_matrix[ch, b] = _cohens_d(
                bp_log[mdd_idx, ch, b], bp_log[hc_idx, ch, b])
        if (ch + 1) % 32 == 0:
            print(f'  Channels {ch + 1}/{128}')

    # Score = max absolute d across bands
    scores = np.abs(d_matrix).max(axis=1)  # (128,)
    ranked = np.argsort(-scores)

    # ── Phase 3: Frontal asymmetry bonus ──
    # Identify homologous pairs from 10-20 mapping
    print('\nComputing frontal asymmetry scores...')
    egi_coords = _load_electrode_coords()
    _20_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4',
                 'P3', 'P4', 'O1', 'O2', 'T3', 'T4', 'T5', 'T6']
    _20_map = _nearest_indices({n: _10_20_TARGETS[n] for n in _20_names}, egi_coords)

    # Pairs: left → right
    pairs = [('Fp1', 'Fp2'), ('F3', 'F4'), ('F7', 'F8'),
             ('C3', 'C4'), ('P3', 'P4'), ('O1', 'O2')]

    # For each asymmetric pair, compute Cohen's d of asymmetry (L-R) in alpha
    asymmetry_bonus = np.zeros(128)
    for l_name, r_name in pairs:
        l_idx = _20_map[l_name]
        r_idx = _20_map[r_name]
        asym_mdd = bp_log[mdd_idx, l_idx, 1] - bp_log[mdd_idx, r_idx, 1]  # alpha band
        asym_hc  = bp_log[hc_idx, l_idx, 1] - bp_log[hc_idx, r_idx, 1]
        d_asym = abs(_cohens_d(asym_mdd, asym_hc))
        # Add bonus to both channels
        asymmetry_bonus[l_idx] += d_asym
        asymmetry_bonus[r_idx] += d_asym
        print(f'  {l_name}–{r_name} asymmetry: d={d_asym:.3f}')

    # Final score: max |d| across bands + asymmetry bonus
    final_scores = scores + 0.3 * asymmetry_bonus  # weight asymmetry bonus
    final_ranked = np.argsort(-final_scores)

    # ── Phase 4: Map Prop1 (10-10 → EGI) and Prop2 (10-20 → EGI) ──
    print('\nMapping literature proposals to EGI indices...')

    # Prop1: 22 prefrontal
    prop1_map = _nearest_indices(_10_10_TARGETS, egi_coords)
    prop1_names = list(_10_10_TARGETS.keys())
    prop1_indices = sorted(set(prop1_map[n] for n in prop1_names))
    print(f'\n  Prop1 (22 prefrontal) → {len(prop1_indices)} unique EGI channels:')
    for name in prop1_names:
        idx = prop1_map[name]
        print(f'    {name:>4s} → E{idx + 1:>3d}')

    # Prop2: 16 channels (10-20 without midline)
    prop2_map = _nearest_indices({n: _10_20_TARGETS[n] for n in _PROP2_NAMES}, egi_coords)
    prop2_indices = sorted(set(prop2_map[n] for n in _PROP2_NAMES))
    print(f'\n  Prop2 (16 10-20) → {len(prop2_indices)} unique EGI channels:')
    for name in _PROP2_NAMES:
        idx = prop2_map[name]
        print(f'    {name:>4s} → E{idx + 1:>3d}')

    # ── Phase 5: MDD-CS top-K ──
    # Choose top 16 channels from ranking (matching Prop2 count for fair comparison)
    # Also export top 8, 16, 22, 32
    mddk_topk = {
        str(k): [int(ch) for ch in final_ranked[:k]]
        for k in [4, 8, 16, 22, 32]
    }
    print(f'\n  MDD-CS top-16: {[f"E{c+1}" for c in final_ranked[:16]]}')

    # ── Build output ──
    channel_info = []
    for rank_pos, ch_idx in enumerate(final_ranked):
        channel_info.append({
            'rank': int(rank_pos + 1),
            'channel_idx_0based': int(ch_idx),
            'channel_name': f'E{ch_idx + 1}',
            'score_mdd': float(final_scores[ch_idx]),
            'd_theta': float(d_matrix[ch_idx, 0]),
            'd_alpha': float(d_matrix[ch_idx, 1]),
            'd_beta': float(d_matrix[ch_idx, 2]),
            'asymmetry_bonus': float(asymmetry_bonus[ch_idx]),
        })

    output = {
        'description': 'MDD-Driven Channel Selection (MDD-CS): '
                       'ranked by max |Cohen\'s d| across theta/alpha/beta bands '
                       '+ frontal asymmetry bonus',
        'n_subjects': int(n_subj),
        'n_mdd': int(mdd_idx.sum()),
        'n_hc': int(hc_idx.sum()),
        'ranking': channel_info,
        'mddk_subsets': mddk_topk,
        'prop1_22prefrontal': {
            'name': '22 prefrontal (Huang Huang)',
            'channels_10_10': prop1_names,
            'egi_indices_0based': prop1_indices,
        },
        'prop2_16ch': {
            'name': '16 channels 10-20 (Zhuozheng Wang)',
            'channels_10_20': _PROP2_NAMES,
            'egi_indices_0based': prop2_indices,
        },
        'bands': ['Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 'Beta (13-30 Hz)'],
    }

    os.makedirs(os.path.dirname(OUT_PATH) or '.', exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved: {OUT_PATH}')
    print('Done.')


if __name__ == '__main__':
    main()

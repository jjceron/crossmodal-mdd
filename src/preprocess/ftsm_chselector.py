"""FTSM (Flexible Temporal Sequence Matching) channel selector.

Implements priority-based channel selection as described in:
  Esmi et al. "Multimodal transformer for depression detection based
  on EEG and interview data" (BSPC, 2026)

Algorithm:
  1. Load filtered 128ch EEG per subject
  2. DTW between every pair of channels → 128×128 cost matrix per subject
  3. Aggregate across subjects → global channel ranking
  4. Build nested subsets (4, 8, 16, 32, 64, 128)

Usage:
  py src/preprocess/ftsm_chselector.py [--segment-sec 2] [--radius 50]

Output:
  data/processed/ftsm_ranking.json
"""
import sys
import os
import glob
import json
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
PARTICIPANTS_PATH = f'{EEG_DIR}/participants.tsv'
OUTPUT_PATH = 'data/processed/ftsm_ranking.json'
SFREQ = 250
N_CHANS = 128


def _load_subjects():
    import pandas as pd
    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1,
                    on_bad_lines='skip', engine='python')
    p = p[[0, 6]]
    p.columns = ['pid', 'group']
    return dict(zip(p['pid'], p['group']))


def _load_and_filter(sub_dir):
    import mne
    edfs = glob.glob(os.path.join(sub_dir, 'eeg', '*.EDF'))
    if not edfs:
        return None
    try:
        raw = mne.io.read_raw_edf(edfs[0], preload=True, verbose=False)
    except Exception:
        return None
    raw.filter(0.5, 60, verbose=False)
    raw.notch_filter(50, verbose=False)
    data = raw.get_data()
    if data.shape[0] < N_CHANS:
        return None
    return data[:N_CHANS]


def _dtw_sakoe_chiba(x, y, radius):
    """DTW with Sakoe-Chiba band constraint.

    Args:
        x, y: 1D arrays of length n, m
        radius: max |i - j| allowed in warping path
    Returns:
        accumulated cost D[n, m]
    """
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        if j_start > j_end:
            continue
        js = np.arange(j_start, j_end + 1)
        costs = np.abs(x[i - 1] - y[js - 1])
        up = D[i - 1, js]
        left = D[i, js - 1]
        upleft = D[i - 1, js - 1]
        D[i, js] = costs + np.minimum(up, np.minimum(left, upleft))
    return float(D[n, m])


def _try_fastdtw(x, y):
    try:
        from fastdtw import fastdtw
        return fastdtw(x, y, dist=lambda a, b: abs(a - b))[0]
    except ImportError:
        return None


def compute_subject_cost_matrix(data, segment_sec, radius):
    """Compute N_CHANS×N_CHANS DTW cost matrix for one subject.

    Args:
        data: [N_CHANS, T] array
        segment_sec: seconds from middle to use
        radius: Sakoe-Chiba band radius
    Returns:
        [N_CHANS, N_CHANS] cost matrix (NaN on diagonal)
    """
    seg_len = int(segment_sec * SFREQ)
    T = data.shape[1]
    if T > seg_len:
        start = (T - seg_len) // 2
        data = data[:, start:start + seg_len]

    n = data.shape[0]
    cost = np.full((n, n), np.nan)

    first_pair = _try_fastdtw(data[0], data[1])
    use_fast = first_pair is not None
    if use_fast:
        from fastdtw import fastdtw
        _dtw = lambda a, b: fastdtw(a, b, dist=lambda p, q: abs(p - q))[0]
    else:
        _dtw = lambda a, b: _dtw_sakoe_chiba(a, b, radius)

    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw(data[i], data[j])
            cost[i, j] = d
            cost[j, i] = d

    return cost


def compute_channel_ranking(subject_cost_matrices):
    """Aggregate per-subject cost matrices into global channel ranking.

    For each channel i: score = mean DTW distance to all other channels.
    Lower score = more representative = higher priority.

    Args:
        subject_cost_matrices: list of [128,128] cost matrices
    Returns:
        list of (channel_idx_0based, score) sorted by priority
    """
    stacked = np.stack(subject_cost_matrices, axis=0)
    avg_cost = np.nanmean(stacked, axis=0)
    scores = np.nanmean(avg_cost, axis=1)
    ranking = sorted(enumerate(scores), key=lambda x: x[1])
    return ranking


def get_nested_subsets(ranking):
    """Build nested channel subsets (1-based, matching paper convention).

    Args:
        ranking: list of (channel_idx_0based, score) sorted by priority
    Returns:
        dict {k: [1-based channel indices]}
    """
    return {
        str(k): sorted(idx + 1 for idx, _ in ranking[:k])
        for k in [4, 8, 16, 32, 64, 128]
    }


def main():
    parser = argparse.ArgumentParser(
        description='FTSM channel ranking for MODMA EEG (128ch)')
    parser.add_argument('--segment-sec', type=float, default=2,
                        help='Seconds of EEG from middle to use per subject (default: 5)')
    parser.add_argument('--radius', type=int, default=50,
                        help='Sakoe-Chiba band radius (default: 50, ~10% of 500 samples)')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help=f'Output path (default: {OUTPUT_PATH})')
    args = parser.parse_args()

    import mne
    mne.set_log_level('WARNING')

    sg = _load_subjects()
    sub_dirs = sorted(glob.glob(os.path.join(EEG_DIR, 'sub-*')))

    all_cost_mats = []
    valid_count = 0
    skipped = 0

    print('Loading subjects and computing DTW matrices...')
    for sd in sub_dirs:
        sid = os.path.basename(sd)
        if sg.get(sid) not in ('MDD', 'HC'):
            continue
        data = _load_and_filter(sd)
        if data is None:
            skipped += 1
            continue
        valid_count += 1
        print(f'  [{valid_count}] {sid}: computing {N_CHANS}×{N_CHANS} DTW '
              f'({args.segment_sec}s segment, radius={args.radius})...')
        cm = compute_subject_cost_matrix(data, args.segment_sec, args.radius)
        all_cost_mats.append(cm)
        print('    done')

    if valid_count == 0:
        print('ERROR: no valid subjects found')
        sys.exit(1)

    print(f'\nAggregating {valid_count} subject matrices (skipped {skipped})...')
    ranking = compute_channel_ranking(all_cost_mats)
    subsets = get_nested_subsets(ranking)

    # Build result
    stacked = np.stack(all_cost_mats, axis=0)
    avg_cost = np.nanmean(stacked, axis=0)

    result = {
        'ranking': [
            {'channel': idx + 1, 'channel_0based': idx, 'score': float(score)}
            for idx, score in ranking
        ],
        'n_subjects': valid_count,
        'segment_sec': args.segment_sec,
        'radius': args.radius,
        'n_channels_total': N_CHANS,
        'nested_subsets': subsets,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    # Save averaged cost matrix for visualization
    cost_path = args.output.replace('.json', '_cost_matrix.npy')
    np.save(cost_path, avg_cost)

    print(f'\nSaved ranking to {args.output}')
    print(f'Saved cost matrix to {cost_path}')
    print(f'\nTop 4  (1-based): {subsets["4"]}')
    print(f'Top 8  (1-based): {subsets["8"]}')
    print(f'Top 16 (1-based): {subsets["16"]}')
    print(f'Top 32 (1-based): {subsets["32"]}')
    print(f'Top 64 (1-based): {subsets["64"][:10]}...')
    print('Top 128         : all channels')


if __name__ == '__main__':
    main()

"""Cache preprocessed EEG windows for fast DL training — v6.

Preprocessing pipeline:
  1. Bandpass filter 0.5–60 Hz
  2. Notch filter 50 Hz (power line)
  3. [--ica] ICA + auto-reject EOG/EMG components
  4. Channel selection:
        - 64: first 64 channels (0–63)
        - 19: clinical 10-20 subset via nearest-neighbor on EGI layout
        - 128: all 128 channels
        - ftsm4|8|16|32|64: top-K channels from FTSM ranking
  5. Average reference (computed on selected channels only)
  6. 2s windows, 50% overlap
  7. All windows retained

Run once (unimodal benchmarks):
  py src/preprocess/cache_modma_eeg.py --channels 64
  py src/preprocess/cache_modma_eeg.py --channels 19
  py src/preprocess/cache_modma_eeg.py --channels 128

Run after FTSM ranking computed:
  py src/preprocess/cache_modma_eeg.py --channels ftsm4
  py src/preprocess/cache_modma_eeg.py --channels ftsm8
  py src/preprocess/cache_modma_eeg.py --channels ftsm16
  py src/preprocess/cache_modma_eeg.py --channels ftsm32
  py src/preprocess/cache_modma_eeg.py --channels ftsm64

With ICA cleaning:
  py src/preprocess/cache_modma_eeg.py --channels 64 --ica

Output:
  data/processed/eeg_preprocessed_{n_ch}ch.npz         (64, 19, 128)
  data/processed/eeg_preprocessed_{n_ch}ch_ica.npz      (ICA-cleaned)
  data/processed/eeg_preprocessed_ftsm{k}.npz           (FTSM subsets)
  data/processed/eeg_preprocessed_ftsm{k}_ica.npz       (FTSM + ICA)
"""
import sys
import os
import glob
import json
import argparse
import numpy as np
from scipy import signal as sg
import pandas as pd
import mne
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
PARTICIPANTS_PATH = f'{EEG_DIR}/participants.tsv'
FTSM_RANKING_PATH = 'data/processed/ftsm_ranking.json'
SFREQ = 250
WINDOW_SEC = 2.0
OVERLAP = 0.5


# ── 10-20 clinical channel mapping (EGI GSN HydroCel 128) ──────────────

_10_20_TARGETS = {
    'Fp1': (-2.7,  6.2,  1.8),
    'Fp2': ( 2.7,  6.2,  1.8),
    'F3':  (-5.0,  4.0,  5.3),
    'F4':  ( 5.0,  4.0,  5.3),
    'F7':  (-7.7,  3.3,  0.4),
    'F8':  ( 7.7,  3.3,  0.4),
    'Fz':  ( 0.0,  2.6,  7.9),
    'C3':  (-7.7,  0.4,  3.5),
    'C4':  ( 7.7,  0.4,  3.5),
    'Cz':  ( 0.0,  0.0,  8.8),
    'P3':  (-5.0, -4.0,  5.3),
    'P4':  ( 5.0, -4.0,  5.3),
    'Pz':  ( 0.0, -2.6,  7.9),
    'O1':  (-2.7, -6.2,  1.8),
    'O2':  ( 2.7, -6.2,  1.8),
    'T3':  (-8.6,  0.0,  0.4),
    'T4':  ( 8.6,  0.0,  0.4),
    'T5':  (-7.7, -3.3,  0.4),
    'T6':  ( 7.7, -3.3,  0.4),
}


def _compute_10_20_indices(electrodes_path):
    """Find EGI channel indices (0-based) closest to standard 10-20 positions."""
    rows = []
    with open(electrodes_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                name = parts[0].strip("'")
                if name.startswith('E'):
                    rows.append([name, parts[1], parts[2], parts[3]])
    df = pd.DataFrame(rows, columns=['name', 'x', 'y', 'z'])
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float32)
    coords = df[['x', 'y', 'z']].values

    mapping = {}
    for name, target in _10_20_TARGETS.items():
        target = np.array(target, dtype=np.float32)
        dists = np.sqrt(np.sum((coords - target) ** 2, axis=1))
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        mapping[name] = (idx, dist)
        print(f'  {name:>4s} -> E{idx + 1:>3d}  (dist={dist:.2f})')

    indices = sorted(set(idx for idx, _ in mapping.values()))
    if len(indices) < len(_10_20_TARGETS):
        print(f'  WARNING: {len(indices)} unique channels for {len(_10_20_TARGETS)} targets '
              f'({len(_10_20_TARGETS) - len(indices)} collisions)')
    print(f'  Total unique channels: {len(indices)}')
    return indices


# ── Argument parsing ------------------------------------------------------

_VALID_CHANNEL_ARGS = ['64', '19', '128'] + [f'ftsm{k}' for k in (4, 8, 16, 32, 64)]


def _parse_channels(raw):
    """Parse --channels argument into (n_ch, selection_mode, ftsm_indices).

    Returns:
        n_ch: number of channels to keep
        out_suffix: string for output filename (e.g. '64ch', 'ftsm4')
        pick_indices: list of 0-based channel indices to select, or None for first-n
    """
    v = raw.lower()

    if v == '128':
        return 128, '128ch', None

    if v == '64':
        return 64, '64ch', None

    if v == '19':
        return 19, '19ch', None  # resolved later via electrodes.tsv

    if v.startswith('ftsm'):
        k = int(v.replace('ftsm', ''))
        if not os.path.exists(FTSM_RANKING_PATH):
            print(f'ERROR: FTSM ranking not found at {FTSM_RANKING_PATH}')
            print('  Run py src/preprocess/ftsm_chselector.py first.')
            sys.exit(1)
        with open(FTSM_RANKING_PATH) as f:
            ranking = json.load(f)
        ch_1based = ranking['nested_subsets'].get(str(k))
        if ch_1based is None:
            print(f'ERROR: no subset for k={k} in FTSM ranking')
            sys.exit(1)
        pick = [c - 1 for c in ch_1based]  # 1-based → 0-based
        return k, f'ftsm{k}', pick

    raise ValueError(f'Invalid --channels argument: {raw}. '
                     f'Choose from {_VALID_CHANNEL_ARGS}')


# ── ICA cleaning ──────────────────────────────────────────────────────────

def _apply_ica(raw, seed=42):
    """Fit ICA, auto-reject EOG and EMG components, return cleaned raw."""
    n_comp = min(40, raw.info['nchan'] - 1)
    raw_ica = raw.copy()
    ica = mne.preprocessing.ICA(n_components=n_comp, method='fastica',
                                 random_state=seed, fit_params=dict(tol=1e-4))
    ica.fit(raw_ica, verbose=False)

    exclude = []

    # EOG: find components correlated with frontal channels
    try:
        eog_idx, eog_scores = ica.find_bads_eog(raw_ica, threshold=2.5, verbose=False)
        exclude.extend(eog_idx)
    except Exception:
        pass

    # EMG: find components with elevated high-frequency content
    try:
        sources = ica.get_sources(raw_ica).get_data()
        for idx in range(min(sources.shape[0], 60)):
            f, Pxx = sg.periodogram(sources[idx], fs=SFREQ)
            low = Pxx[(f >= 0.5) & (f < 20)].sum()
            high = Pxx[(f >= 20) & (f <= 60)].sum()
            if low > 0 and (high / low) > 1.0:
                exclude.append(idx)
    except Exception:
        pass

    exclude = list(set(exclude))
    n_rej = len(exclude)
    if exclude:
        ica.exclude = exclude
        raw_ica = ica.apply(raw_ica, verbose=False)

    return raw_ica, n_rej


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Cache preprocessed MODMA EEG windows')
    parser.add_argument('--channels', type=str, default='64',
                        help=f'Channel selection. Options: {_VALID_CHANNEL_ARGS}')
    parser.add_argument('--ica', action='store_true',
                        help='Apply ICA + auto-reject EOG/EMG components before channel selection')
    args = parser.parse_args()

    n_ch, out_suffix, pick_indices = _parse_channels(args.channels)
    if args.ica:
        out_suffix += '_ica'
    out_path = f'data/processed/eeg_preprocessed_{out_suffix}.npz'

    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1,
                     on_bad_lines='skip', engine='python')
    p = p[[0, 6]]
    p.columns = ['pid', 'group']
    sg = dict(zip(p['pid'], p['group']))

    sub_dirs = sorted(glob.glob(os.path.join(EEG_DIR, 'sub-*')))

    # Resolve 10-20 clinical indices if needed
    is_19ch = out_suffix.startswith('19ch')
    if is_19ch and sub_dirs:
        electrodes_tsv = glob.glob(os.path.join(sub_dirs[0], 'eeg', '*_electrodes.tsv'))
        if electrodes_tsv:
            print('Computing 10-20 → EGI channel mapping:')
            pick_indices = _compute_10_20_indices(electrodes_tsv[0])
        else:
            print('ERROR: electrodes.tsv not found. Cannot compute 19-channel mapping.')
            return

    all_wins, all_ids, all_labels = [], [], []

    for sd in sub_dirs:
        sid = os.path.basename(sd)
        g = sg.get(sid)
        if g not in ('MDD', 'HC'):
            continue
        edfs = glob.glob(os.path.join(sd, 'eeg', '*.EDF'))
        if not edfs:
            continue
        try:
            raw = mne.io.read_raw_edf(edfs[0], preload=True, verbose=False)
        except Exception:
            continue

        raw.filter(0.5, 60, verbose=False)
        raw.notch_filter(50, verbose=False)

        if args.ica:
            raw, n_rej = _apply_ica(raw)
            print(f'    ICA: {n_rej} components rejected')

        if pick_indices is not None:
            if max(pick_indices) >= len(raw.ch_names):
                continue
            raw.pick([raw.ch_names[i] for i in pick_indices])

        raw.set_eeg_reference('average', verbose=False)

        if pick_indices is None:
            if len(raw.ch_names) < n_ch:
                continue
            data = raw.get_data()[:n_ch]
        else:
            data = raw.get_data()

        ws = int(WINDOW_SEC * SFREQ)
        stride = int(ws * (1 - OVERLAP))
        n_w = (data.shape[1] - ws) // stride + 1
        if n_w < 1:
            continue

        win = np.lib.stride_tricks.sliding_window_view(
            data, ws, axis=1)[:, ::stride].transpose(1, 0, 2)
        win = win[:n_w].astype(np.float32)

        all_wins.append(win)
        all_ids.append(sid)
        all_labels.append(1 if g == 'MDD' else 0)
        print(f'  {sid}: {win.shape[0]} windows, label={all_labels[-1]}')

    os.makedirs('data/processed', exist_ok=True)

    # Save as object array (variable-length per subject, compatible with original training pipeline)
    obj_arr = np.empty(len(all_wins), dtype=object)
    for i, win in enumerate(all_wins):
        obj_arr[i] = win

    np.savez(out_path,
             windows=obj_arr,
             subject_ids=np.array(all_ids),
             labels=np.array(all_labels, dtype=np.int32))
    n_mdd = sum(all_labels)
    print(f'\nSaved: {out_path}')
    print(f'  Subjects: {len(all_ids)} ({n_mdd} MDD, {len(all_ids) - n_mdd} HC)')
    print(f'  Total windows: {sum(w.shape[0] for w in all_wins)}')
    print(f'  Avg windows/subject: {np.mean([w.shape[0] for w in all_wins]):.0f}')
    print(f'  Min/Max windows: {min(w.shape[0] for w in all_wins)}/{max(w.shape[0] for w in all_wins)}')
    print(f'  Shape per window: ({all_wins[0].shape[1]}, {all_wins[0].shape[2]})')
    print(f'  Saved as object array — compatible with original training pipeline')


if __name__ == '__main__':
    main()



"""Cache preprocessed EEG windows for fast DL training — v4.

Preprocessing pipeline:
  1. Bandpass filter 0.5–60 Hz
  2. Notch filter 50 Hz (power line)
  3. Channel selection (64-first or 19-clinical via 10-20 mapping)
  4. Average reference (computed on selected channels only)
  5. 2s windows, 50% overlap
  6. All windows retained (no random subsampling)

Run once:
  py src/preprocess/cache_modma_eeg.py --channels 64
  py src/preprocess/cache_modma_eeg.py --channels 19
 
Output:
  data/processed/eeg_preprocessed_64ch.npz
  data/processed/eeg_preprocessed_19ch.npz
"""
import sys, os, glob, argparse, numpy as np, pandas as pd, mne, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
PARTICIPANTS_PATH = f'{EEG_DIR}/participants.tsv'
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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Cache preprocessed MODMA EEG windows')
    parser.add_argument('--channels', type=int, default=64, choices=[64, 19],
                        help='64 = first 64 channels, 19 = 10-20 clinical subset')
    args = parser.parse_args()

    n_ch = args.channels
    out_path = f'data/processed/eeg_preprocessed_{n_ch}ch.npz'

    p = pd.read_csv(PARTICIPANTS_PATH, sep='\t', header=None, skiprows=1,
                     on_bad_lines='skip', engine='python')
    p = p[[0, 6]]
    p.columns = ['pid', 'group']
    sg = dict(zip(p['pid'], p['group']))

    sub_dirs = sorted(glob.glob(os.path.join(EEG_DIR, 'sub-*')))

    if n_ch == 19 and sub_dirs:
        electrodes_tsv = glob.glob(os.path.join(sub_dirs[0], 'eeg', '*_electrodes.tsv'))
        if electrodes_tsv:
            print('Computing 10-20 → EGI channel mapping:')
            clinical_indices = _compute_10_20_indices(electrodes_tsv[0])
        else:
            print('ERROR: electrodes.tsv not found. Cannot compute 19-channel mapping.')
            return
    else:
        clinical_indices = None

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

        if clinical_indices is not None:
            if max(clinical_indices) >= len(raw.ch_names):
                continue
            raw.pick([raw.ch_names[i] for i in clinical_indices])

        raw.set_eeg_reference('average', verbose=False)

        if clinical_indices is None:
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
    np.savez(out_path,
             windows=np.array(all_wins, dtype=object),
             subject_ids=np.array(all_ids),
             labels=np.array(all_labels, dtype=np.int32),
             allow_pickle=True)
    n_mdd = sum(all_labels)
    print(f'\nSaved: {out_path}')
    print(f'  Subjects: {len(all_ids)} ({n_mdd} MDD, {len(all_ids) - n_mdd} HC)')
    print(f'  Total windows: {sum(w.shape[0] for w in all_wins)}')
    print(f'  Avg windows/subject: {np.mean([w.shape[0] for w in all_wins]):.0f}')
    print(f'  Min/Max windows: {min(w.shape[0] for w in all_wins)}/{max(w.shape[0] for w in all_wins)}')
    print(f'  Shape per window: {all_wins[0].shape[1:]}')


if __name__ == '__main__':
    main()



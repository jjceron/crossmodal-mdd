"""Create preprocessed EEG npz for a subset of channels from a 128ch cache.

Re-references to the average of the selected subset (correcting the original
128ch average reference).

Usage:
  python src/data/subset_channels.py --indices 0,1,5,10 --suffix 22ch
  python src/data/subset_channels.py --json-key mddk_subsets.16 --suffix mddk16
  python src/data/subset_channels.py --json-key prop1_22prefrontal --suffix 22ch
  python src/data/subset_channels.py --json-key prop2_16ch --suffix 16ch

  # From a different cache (e.g. ICA-cleaned):
  python src/data/subset_channels.py --json-key prop1_22prefrontal --suffix 22ch_clean --source 128ch_ica_rej
"""
import sys, os, argparse, json
import numpy as np

sys.path.insert(0, '.')
CACHE_PATH = 'data/processed/eeg_preprocessed_128ch.npz'
SELECTION_PATH = 'data/processed/channel_selection.json'
OUT_DIR = 'data/processed'


def load_selection_indices(json_key):
    """Parse a JSON key like 'mddk_subsets.16' or 'prop1_22prefrontal'."""
    with open(SELECTION_PATH) as f:
        sel = json.load(f)

    if '.' in json_key:
        outer, inner = json_key.split('.', 1)
        inner = int(inner) if inner.isdigit() else inner
        indices = sel[outer][str(inner) if isinstance(inner, int) else inner]
    else:
        indices = sel[json_key]['egi_indices_0based']

    return indices


def main():
    parser = argparse.ArgumentParser(description='Create channel subset npz from 128ch cache')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--indices', type=str,
                       help='Comma-separated 0-based channel indices')
    group.add_argument('--json-key', type=str,
                       help='Key into channel_selection.json, e.g. "mddk_subsets.16"')
    parser.add_argument('--suffix', type=str, required=True,
                       help='Output suffix (e.g. "22ch", "16ch", "mddk16")')
    parser.add_argument('--source', type=str, default='128ch',
                       help='Source cache suffix (e.g. "128ch", "128ch_ica", "128ch_ica_rej"). '
                            'Default: "128ch"')
    args = parser.parse_args()

    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(',')]
    else:
        indices = load_selection_indices(args.json_key)

    indices = np.array(sorted(set(indices)))
    n_ch = len(indices)
    print(f'Indices (0-based): {indices.tolist()}')
    print(f'EGI names: {[f"E{i+1}" for i in indices]}')
    print(f'Total channels: {n_ch}')

    # ── Load cache (object array or fixed-size) ──
    cache_path = f'data/processed/eeg_preprocessed_{args.source}.npz'
    print(f'\nLoading cache: {cache_path}')
    c = np.load(cache_path, allow_pickle=True)
    raw_windows = c['windows']
    labels = c['labels']
    subject_ids = c['subject_ids']

    # Convert to list: object array directly, fixed-size via mask
    if 'window_mask' in c:
        # Old format: (n_subj, n_max, 128, 500) + mask
        mask = c['window_mask']
        windows_list = [raw_windows[i, :int(mask[i].sum())] for i in range(len(raw_windows))]
    else:
        # New format: object array (n_subj,) each of shape (n_w, 128, 500)
        windows_list = list(raw_windows)

    # ── Extract subset and re-reference ──
    print('Extracting channels and re-referencing to subset average...')
    n_subj = len(windows_list)
    all_wins = []

    for i in range(n_subj):
        subj_data = windows_list[i]  # (n_valid, 128, 500)

        # Select channels
        subj_subset = subj_data[:, indices, :]  # (n_valid, n_ch, 500)

        # Re-reference: subtract mean of selected channels
        # Original: X_i_128ref = raw_i - mean(raw_all_128)
        # We want:  X_i_subref = raw_i - mean(raw_selected)
        # = (X_i_128ref + mean_all) - (X_i_128ref[S].mean() + mean_all)
        # = X_i_128ref - X_i_128ref[S].mean()
        ch_mean = subj_subset.mean(axis=1, keepdims=True)  # (n_valid, 1, 500)
        subj_subset = subj_subset - ch_mean

        all_wins.append(subj_subset.astype(np.float32))

        if (i + 1) % 10 == 0:
            print(f'  Processed {i + 1}/{n_subj} subjects')

    # ── Save as object array ──
    out_suffix = args.suffix
    out_path = os.path.join(OUT_DIR, f'eeg_preprocessed_{out_suffix}.npz')
    obj_arr = np.empty(len(all_wins), dtype=object)
    for i, win in enumerate(all_wins):
        obj_arr[i] = win

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(out_path,
             windows=obj_arr,
             subject_ids=subject_ids,
             labels=np.array(labels, dtype=np.int32))

    total_wins = sum(w.shape[0] for w in all_wins)
    print(f'\nSaved: {out_path}')
    print(f'  Subjects: {len(all_wins)} ({(labels==1).sum()} MDD, {(labels==0).sum()} HC)')
    print(f'  Channels: {n_ch}')
    print(f'  Total windows: {total_wins}')
    print(f'  Shape per window: ({n_ch}, 500)')
    print('  Format: object array — compatible with training pipeline')


if __name__ == '__main__':
    main()

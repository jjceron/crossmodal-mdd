import numpy as np
for k in [4, 8, 16]:
    c = np.load(f'data/processed/eeg_preprocessed_ftsm{k}.npz')
    mask = c['window_mask']
    dense = c['windows']
    obj = np.empty(dense.shape[0], dtype=object)
    for i in range(dense.shape[0]):
        obj[i] = dense[i][mask[i]]
    np.savez(f'data/processed/eeg_preprocessed_ftsm{k}.npz',
             windows=obj, subject_ids=c['subject_ids'], labels=c['labels'])
    print(f'ftsm{k}: {dense.shape[0]} subj, {obj[0].shape[1]}ch, '
          f'{obj[0].shape[0]} windows (first subj), '
          f'removed {dense.shape[1] - obj[0].shape[0]} padded windows')

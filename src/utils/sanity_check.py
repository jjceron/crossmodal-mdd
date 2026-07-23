"""Quick sanity check: band-power features + sklearn classifiers on EEG cache.

Usage:
  # Default: existing 128ch cache
  python src/utils/sanity_check.py

  # Specific cache (e.g. ICA + rejection)
  python src/utils/sanity_check.py --source 128ch_ica_rej

  # Subset of channels (e.g. 22 prefrontal)
  python src/utils/sanity_check.py --source 128ch --channels 22ch
  python src/utils/sanity_check.py --source 128ch_ica_rej --channels 22ch_ica_rej

Compares LDA and Random Forest with Leave-One-Subject-Out CV.
Reports BACC, AUC, Sens, Spec, and Cohen's d per channel.
"""
import sys, os, argparse, json
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score

sys.path.insert(0, '.')
SFREQ = 250


def _extract_band_power(data):
    """Extract theta, alpha, beta power from (n_windows, n_ch, 500) array."""
    fft = np.fft.rfft(data, axis=-1)
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(500, 1.0 / SFREQ)

    theta = psd[:, :, (freqs >= 4) & (freqs < 8)].mean(axis=-1)
    alpha = psd[:, :, (freqs >= 8) & (freqs < 13)].mean(axis=-1)
    beta  = psd[:, :, (freqs >= 13) & (freqs < 30)].mean(axis=-1)

    return np.log1p(np.stack([theta, alpha, beta], axis=-1))  # (n_w, n_ch, 3)


def _cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp


def run_loo(X, y, clf, clf_name):
    """Leave-One-Subject-Out CV."""
    loo = LeaveOneOut()
    preds, probs = np.zeros(len(y)), np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf.fit(X_train, y[train_idx])
        probs[test_idx] = clf.predict_proba(X_test)[:, 1]
        preds[test_idx] = clf.predict(X_test)

    bacc = balanced_accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    sens = recall_score(y, preds, pos_label=1)
    spec = recall_score(y, preds, pos_label=0)
    return {'model': clf_name, 'BACC': round(bacc, 4), 'AUC': round(auc, 4),
            'Sens': round(sens, 4), 'Spec': round(spec, 4)}


def main():
    parser = argparse.ArgumentParser(description='Quick sanity check on EEG cache')
    parser.add_argument('--source', type=str, default='128ch',
                        help='Cache suffix (default: "128ch")')
    parser.add_argument('--channels', type=str, default=None,
                        help='Channel subset file suffix (e.g. "22ch", "22ch_ica_rej"). '
                             'If None, uses full cache channels.')
    args = parser.parse_args()

    # Load
    if args.channels:
        path = f'data/processed/eeg_preprocessed_{args.channels}.npz'
        print(f'Loading channel subset: {path}')
    else:
        path = f'data/processed/eeg_preprocessed_{args.source}.npz'
        print(f'Loading cache: {path}')

    c = np.load(path, allow_pickle=True)
    wins, labels = c['windows'], c['labels']
    n_subj = len(wins)
    n_mdd = int((labels == 1).sum())
    n_hc = int((labels == 0).sum())
    print(f'  Subjects: {n_subj} ({n_mdd} MDD, {n_hc} HC)')

    # Extract band power per subject
    print('\nExtracting band power...')
    X_list, y_list = [], []
    ch_names = None
    for i in range(n_subj):
        w = wins[i]  # (n_w, n_ch, 500)
        if isinstance(w, np.ndarray) and w.ndim == 3:
            bp = _extract_band_power(w)  # (n_w, n_ch, 3)
            # Average over windows -> (n_ch * 3,) feature vector
            feat = bp.mean(axis=0).ravel()  # (n_ch * 3,)
            X_list.append(feat)
            y_list.append(int(labels[i]))
            if ch_names is None:
                n_ch = w.shape[1]
                ch_names = [f'ch{j}' for j in range(n_ch)]

    X = np.array(X_list)
    y = np.array(y_list)
    print(f'  Feature matrix: {X.shape} (subjects, channels x bands)')

    # Cohen's d per channel (alpha band, most relevant for MDD)
    print('\nCohen\'s d per channel (alpha band, MDD vs HC):')
    mdd_idx = y == 1
    hc_idx = y == 0
    d_vals = []
    for ch in range(n_ch):
        idx_alpha = ch * 3 + 1  # alpha is index 1 in [theta, alpha, beta]
        d = _cohens_d(X[mdd_idx, idx_alpha], X[hc_idx, idx_alpha])
        d_vals.append(d)
    d_vals = np.array(d_vals)
    print(f'  Mean |d|: {np.abs(d_vals).mean():.3f}')
    print(f'  Max |d|: {np.abs(d_vals).max():.3f}')
    print(f'  Channels with |d| > 0.2: {(np.abs(d_vals) > 0.2).sum()}/{n_ch}')
    print(f'  Channels with |d| > 0.5: {(np.abs(d_vals) > 0.5).sum()}/{n_ch}')

    # LDA
    print('\n--- LDA (Leave-One-Subject-Out) ---')
    res_lda = run_loo(X, y, LinearDiscriminantAnalysis(), 'LDA')
    print(f'  BACC={res_lda["BACC"]:.4f}, AUC={res_lda["AUC"]:.4f}, '
          f'Sens={res_lda["Sens"]:.4f}, Spec={res_lda["Spec"]:.4f}')

    # RF
    print('\n--- Random Forest (Leave-One-Subject-Out) ---')
    res_rf = run_loo(X, y, RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=42, class_weight='balanced'), 'RF')
    print(f'  BACC={res_rf["BACC"]:.4f}, AUC={res_rf["AUC"]:.4f}, '
          f'Sens={res_rf["Sens"]:.4f}, Spec={res_rf["Spec"]:.4f}')

    # Summary
    print('\n' + '=' * 50)
    print(f'SOURCE: {args.source}')
    if args.channels:
        print(f'CHANNELS: {args.channels}')
    print(f'SUBJECTS: {n_mdd} MDD / {n_hc} HC')
    print(f'FEATURES: {X.shape[1]} ({n_ch} ch x 3 bands)')
    print(f'Cohen\'s d: |d|_mean={np.abs(d_vals).mean():.3f}, |d|_max={np.abs(d_vals).max():.3f}')
    print(f'LDA: BACC={res_lda["BACC"]:.4f}, AUC={res_lda["AUC"]:.4f}')
    print(f'RF:  BACC={res_rf["BACC"]:.4f}, AUC={res_rf["AUC"]:.4f}')
    chance = 0.5
    print(f'Chance: BACC={chance:.2f}')
    print(f'LDA > chance by: {res_lda["BACC"] - chance:+.4f}')
    print(f'RF  > chance by: {res_rf["BACC"] - chance:+.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()

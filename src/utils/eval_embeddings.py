"""
Evaluate frozen backbone embeddings with linear classifiers.

For each fold checkpoint, extracts subject-level embeddings (mean-pooled
windows) using the frozen backbone without its classifier head, then trains
SVM and Logistic Regression on train embeddings and evaluates on test.

Answers: does the learned representation contain MDD-related signal?

Usage:
  py src/utils/eval_embeddings.py --modality eeg --model deepconvnet --channels 64
  py src/utils/eval_embeddings.py --modality audio --model shallowconvnet
  py src/utils/eval_embeddings.py --modality eeg --model eegnet --channels 64
  py src/utils/eval_embeddings.py --modality audio --model cnnlstm
"""
import sys, os, json, argparse, warnings
import numpy as np
import torch, torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5


# ── Model building ───────────────────────────────────────────────────────

def _build_emb_wrapper(model_key, n_channels, n_samples, modality, bottleneck_dim=None):
    """Build model with classifier replaced by Identity. Returns (model, embed_dim)."""
    from src.models.shallowconvnet import ShallowConvNet
    from src.models.deepconvnet import DeepConvNet

    if model_key == 'shallowconvnet':
        class Emb(nn.Module):
            def __init__(self):
                nonlocal bottleneck_dim
                super().__init__()
                self.m = ShallowConvNet(n_channels, 1, n_samples, 0.5)
                if bottleneck_dim is not None:
                    in_features = self.m.classifier.in_features
                    self.m.classifier = nn.Sequential(
                        nn.Linear(in_features, bottleneck_dim),
                    )
                    self.embed_dim = bottleneck_dim
                else:
                    self.embed_dim = self.m.classifier.in_features
                    self.m.classifier = nn.Identity()
            def forward(self, x):
                return self.m(x).squeeze(-1)
        m = Emb()
        return m, m.embed_dim

    if model_key == 'deepconvnet':
        if modality == 'eeg':
            class Emb(nn.Module):
                def __init__(self):
                    nonlocal bottleneck_dim
                    super().__init__()
                    self.m = DeepConvNet(n_channels, 1, n_samples, 0.5)
                    if bottleneck_dim is not None:
                        self.m.classifier = nn.Sequential(
                            nn.Linear(self.m.fc_features, bottleneck_dim),
                        )
                        self.embed_dim = bottleneck_dim
                    else:
                        self.embed_dim = self.m.fc_features
                        self.m.classifier = nn.Identity()
                def forward(self, x):
                    return self.m(x).squeeze(-1)
            m = Emb()
            return m, m.embed_dim
        else:  # audio: 4-block AudioDeepConvNet
            class Emb(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.block1 = nn.Sequential(
                        nn.Conv2d(1, 8, (1, 10)), nn.BatchNorm2d(8), nn.ELU(),
                        nn.MaxPool2d((1, 2)), nn.Dropout2d(0.25))
                    self.block2 = nn.Sequential(
                        nn.Conv2d(8, 16, (n_channels, 1)), nn.BatchNorm2d(16), nn.ELU(),
                        nn.MaxPool2d((1, 2)), nn.Dropout2d(0.25))
                    self.block3 = nn.Sequential(
                        nn.Conv2d(16, 32, (1, 10)), nn.BatchNorm2d(32), nn.ELU(),
                        nn.MaxPool2d((1, 2)), nn.Dropout2d(0.25))
                    self.block4 = nn.Sequential(
                        nn.Conv2d(32, 64, (1, 10)), nn.BatchNorm2d(64), nn.ELU(),
                        nn.MaxPool2d((1, 2)), nn.Dropout2d(0.5))
                    dummy = torch.randn(1, 1, n_channels, n_samples)
                    with torch.no_grad():
                        x = self.block1(dummy); x = self.block2(x)
                        x = self.block3(x); x = self.block4(x)
                    self.embed_dim = int(x.numel())
                    self.classifier = nn.Identity()
                def forward(self, x):
                    if x.dim() == 3: x = x.unsqueeze(1)
                    x = self.block1(x); x = self.block2(x)
                    x = self.block3(x); x = self.block4(x)
                    return self.classifier(x.flatten(start_dim=1)).squeeze(-1)
            m = Emb()
            return m, m.embed_dim

    if model_key == 'eegnet':
        from src.models.eegnet import EEGNet
        if modality == 'eeg':
            F1, D, F2 = 16, 4, 32
        else:  # audio
            F1, D, F2 = 32, 8, 64
        class Emb(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = EEGNet(n_channels, 1, F1=F1, D=D, F2=F2,
                                temporal_kern=31, separable_kern=15,
                                pool1=4, pool2=4, dropout=0.5,
                                meanmax_alpha=0.0)
                self.m.classifier = nn.Identity()
                self.embed_dim = F2
            def forward(self, x):
                return self.m(x)[0].squeeze(-1)
        m = Emb()
        return m, m.embed_dim

    if model_key == 'cnnlstm':
        from src.models.cnn_lstm import CNNLSTM
        class Emb(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = CNNLSTM(n_channels, 1, n_samples, dropout=0.5)
                self.embed_dim = self.m.classifier.in_features
                self.m.classifier = nn.Identity()
            def forward(self, x):
                return self.m(x).squeeze(-1)
        m = Emb()
        return m, m.embed_dim

    raise ValueError(f'Unknown model: {model_key}')


# ── Embedding extraction ─────────────────────────────────────────────────

def extract_embeddings(model, subject_data, subject_ids, device):
    """Pass all windows through frozen backbone, mean-pool per subject.
    Returns embeddings [N_subj × D]."""
    model.eval()
    model.to(device)
    subj_embs = {}

    for idx, wins in enumerate(subject_data):
        sid = str(subject_ids[idx])
        if wins.shape[0] == 0:
            continue
        wins_t = torch.from_numpy(wins).float()
        all_emb = []
        bs = 32
        for i in range(0, wins_t.shape[0], bs):
            batch = wins_t[i:i + bs].to(device)
            with torch.no_grad():
                emb = model(batch).cpu()
            all_emb.append(emb)
        subj_embs[sid] = torch.cat(all_emb, dim=0).mean(dim=0).numpy()

    subjects_sort = sorted(subj_embs.keys())
    emb_matrix = np.stack([subj_embs[s] for s in subjects_sort])
    return emb_matrix


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate frozen backbone embeddings')
    parser.add_argument('--modality', required=True, choices=['eeg', 'audio'])
    parser.add_argument('--model', required=True,
                        choices=['deepconvnet', 'shallowconvnet', 'eegnet', 'cnnlstm'])
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--bottleneck-dim', type=int, default=None,
                        help='Must match training bottleneck-dim if used')
    parser.add_argument('--fold', type=int, default=None,
                        help='Evaluate a single fold only')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.modality == 'eeg':
        n_samples = 500
        n_channels = args.channels
        cache_path = f'data/processed/eeg_preprocessed_{n_channels}ch.npz'
        ckpt_dir = f'outputs/results/classical_dl/trained_eeg/{args.model}_{n_channels}ch'
    else:
        n_samples = 200
        n_channels = 64
        cache_path = 'data/processed/audio_mel_cache.npz'
        ckpt_dir = f'outputs/results/classical_dl/trained_audio/{args.model}_64mel'

    c = np.load(cache_path, allow_pickle=True)
    subject_data = list(c['windows'])
    labels = c['labels'].astype(int)
    subject_ids = [str(s) for s in c['subject_ids']]

    n_subjects = len(subject_ids)
    n_mdd = int(labels.sum())
    n_hc = int((1 - labels).sum())
    print(f'Modality: {args.modality}  Model: {args.model}  '
          f'Subjects: {n_subjects} MDD={n_mdd} HC={n_hc}')

    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds_list = list(gkf.split(np.zeros(n_subjects), labels, groups=subject_ids))

    folds_to_eval = [args.fold] if args.fold else range(1, N_FOLDS + 1)

    svm_baccs, lr_baccs, svm_aucs, lr_aucs = [], [], [], []

    for fi_1based in folds_to_eval:
        fi = fi_1based - 1
        train_val_idx, test_idx = folds_list[fi]

        ckpt_path = os.path.join(ckpt_dir, f'fold_{fi_1based}.pt')
        if not os.path.exists(ckpt_path):
            print(f'  Fold {fi_1based}: checkpoint not found at {ckpt_path}')
            continue

        model, embed_dim = _build_emb_wrapper(args.model, n_channels, n_samples, args.modality,
                                               bottleneck_dim=args.bottleneck_dim)
        state = torch.load(ckpt_path, map_location='cpu')
        sd = state['model_state_dict']
        if args.bottleneck_dim is not None:
            sd = {k: v for k, v in sd.items() if 'classifier.2' not in k}
        model.load_state_dict(sd, strict=False)

        if fi == 0:
            print(f'Embedding dim: {embed_dim}')

        embeddings = extract_embeddings(model, subject_data, subject_ids, device)
        y_all = labels.copy()

        y_train = y_all[train_val_idx]
        y_test = y_all[test_idx]
        X_train = embeddings[train_val_idx]
        X_test = embeddings[test_idx]

        # SVM
        svm = LinearSVC(C=1.0, dual='auto', max_iter=5000, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        if hasattr(svm, 'decision_function'):
            y_score_svm = svm.decision_function(X_test)
        else:
            y_score_svm = y_pred_svm.astype(float)
        svm_bacc = balanced_accuracy_score(y_test, y_pred_svm)
        svm_auc = roc_auc_score(y_test, y_score_svm) if len(np.unique(y_test)) > 1 else 0.5
        svm_baccs.append(svm_bacc)
        svm_aucs.append(svm_auc)

        # Logistic Regression
        lr = LogisticRegression(C=1.0, max_iter=5000, random_state=RANDOM_STATE)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]
        lr_bacc = balanced_accuracy_score(y_test, y_pred_lr)
        lr_auc = roc_auc_score(y_test, y_prob_lr) if len(np.unique(y_test)) > 1 else 0.5
        lr_baccs.append(lr_bacc)
        lr_aucs.append(lr_auc)

        print(f'  Fold {fi_1based}: SVM bacc={svm_bacc:.3f} AUC={svm_auc:.3f}  '
              f'LR bacc={lr_bacc:.3f} AUC={lr_auc:.3f}')

    if len(svm_baccs) > 1:
        print(f"\n{'=' * 60}")
        print(f"  EMBEDDING QUALITY ({len(svm_baccs)} folds)")
        print(f"  {'':>12s} | {'mean':>8s} {'+-':>2s} {'std':>8s}")
        print(f"  {'':->12s}-+-{'-' * 20}")
        print(f"  {'SVM bacc':>12s} | {np.mean(svm_baccs):>8.3f} {'+-':>2s} {np.std(svm_baccs):>8.3f}")
        print(f"  {'SVM AUC':>12s} | {np.mean(svm_aucs):>8.3f} {'+-':>2s} {np.std(svm_aucs):>8.3f}")
        print(f"  {'LR  bacc':>12s} | {np.mean(lr_baccs):>8.3f} {'+-':>2s} {np.std(lr_baccs):>8.3f}")
        print(f"  {'LR  AUC':>12s} | {np.mean(lr_aucs):>8.3f} {'+-':>2s} {np.std(lr_aucs):>8.3f}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

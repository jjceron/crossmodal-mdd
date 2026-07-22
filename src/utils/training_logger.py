"""Standardized logger for binary classification training and evaluation."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


# ── Classification Logger ──────────────────────────────────────────────

_KEYS_CLAS = ['acc', 'bacc', 'f1', 'sens', 'spec']


class ClassificationLogger:
    """Logger for binary classification: metrics, epoch log, fold test, summary."""

    def __init__(self):
        self.fold_metrics = []

    def metrics(self, true, pred):
        """Compute classification metrics dict."""
        t, p = np.array(true, dtype=int), np.array(pred, dtype=int)
        tp = int(((p == 1) & (t == 1)).sum())
        tn = int(((p == 0) & (t == 0)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        return {
            'acc':  float(accuracy_score(t, p)),
            'bacc': float(balanced_accuracy_score(t, p)),
            'f1':   float(f1_score(t, p, zero_division=0)),
            'tp':   tp,
            'tn':   tn,
            'fp':   fp,
            'fn':   fn,
            'sens': float(tp / max(tp + fn, 1e-12)),
            'spec': float(tn / max(tn + fp, 1e-12)),
        }

    @staticmethod
    def log_header():
        print(f"  {'Epoch':>5s} | {'T_loss':>8s} {'V_loss':>8s} "
              f"{'T_acc':>6s} {'V_acc':>6s} | "
              f"{'V_bacc':>6s} {'V_f1':>6s} {'V_sens':>6s} {'V_spec':>6s} | pat")

    @staticmethod
    def log_epoch(epoch, tr_loss, vl_loss, tr_m, vl_m, patience):
        print(f"  {epoch:5d} | {tr_loss:8.4f} {vl_loss:8.4f} "
              f"{tr_m['acc']:6.3f} {vl_m['acc']:6.3f} | "
              f"{vl_m['bacc']:6.3f} {vl_m['f1']:6.3f} "
              f"{vl_m['sens']:6.3f} {vl_m['spec']:6.3f} | {patience:2d}")

    def log_fold_test(self, test_true, test_pred):
        m = self.metrics(test_true, test_pred)
        self.fold_metrics.append(m)
        print(f"  >>> test: acc={m['acc']:.3f} bacc={m['bacc']:.3f} "
              f"f1={m['f1']:.3f}")
        return m

    def log_summary(self, n_folds=None, split_type='gkf'):
        if n_folds is None:
            n_folds = len(self.fold_metrics)
        arrays = {k: np.array([m[k] for m in self.fold_metrics]) for k in _KEYS_CLAS}
        title = 'GKF' if split_type == 'gkf' else 'LOSO'
        print(f"\n{'=' * 60}")
        print(f"  {title} RESULT ({n_folds} folds)")
        print(f"  {'':>7s} | {'mean':>8s} {'+-':>2s} {'std':>8s}")
        print(f"  {'':->7s}-+-{'-' * 20}")
        for k in _KEYS_CLAS:
            mn, sd = float(np.mean(arrays[k])), float(np.std(arrays[k]))
            print(f"  {k:>7s} | {mn:>8.3f} {'+-':>2s} {sd:>8.3f}")
        if n_folds > 1:
            print()
            hdr = ' | '.join(f"{k:>8s}" for k in _KEYS_CLAS)
            print(f"  {'Fold':>7s} | {hdr}")
            print(f"  {'':->7s}-+-{'-' * (9 * len(_KEYS_CLAS) - 1)}")
            for fi in range(n_folds):
                vals = ' | '.join(f"{arrays[k][fi]:>8.3f}" for k in _KEYS_CLAS)
                print(f"  {fi + 1:>7d} | {vals}")
        print(f"{'=' * 60}")



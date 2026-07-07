"""
Standardized training logger for classification and regression pipelines.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score, mean_absolute_error


# ── Shared helpers ─────────────────────────────────────────────────────

def _check_binary(t, p):
    return len(np.unique(t)) >= 2 and len(np.unique(p)) >= 2


# ── Classification Logger ──────────────────────────────────────────────

_KEYS_CLAS = ['acc', 'bacc', 'f1', 'sens', 'spec']


class ClassificationLogger:
    """Logger for binary classification: metrics, epoch log, fold test, summary."""

    def __init__(self):
        self.fold_metrics = []

    def metrics(self, true, pred):
        """Compute classification metrics dict."""
        t, p = np.array(true, dtype=int), np.array(pred, dtype=int)
        if not _check_binary(t, p):
            return {k: 0.0 for k in _KEYS_CLAS}
        return {
            'acc':  float(accuracy_score(t, p)),
            'bacc': float(balanced_accuracy_score(t, p)),
            'f1':   float(f1_score(t, p, zero_division=0)),
            'sens': float(f1_score(t, p, pos_label=1, zero_division=0)),
            'spec': float(f1_score(t, p, pos_label=0, zero_division=0)),
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


# ── Regression Logger ──────────────────────────────────────────────────

_KEYS_REGR = ['r2', 'mae', 'spear', 'pear', 'nrmse']


class RegressionLogger:
    """Logger for regression: metrics, epoch log, fold test, summary."""

    def __init__(self):
        self.fold_metrics = []

    def metrics(self, true, pred):
        t, p = np.array(true, dtype=float), np.array(pred, dtype=float)
        n = len(t)
        mae = mean_absolute_error(t, p) if n > 1 else float('nan')
        r2 = r2_score(t, p) if n > 1 else float('nan')
        nrmse = np.sqrt(np.mean((t - p) ** 2)) / (t.std() + 1e-10) if n > 1 else float('nan')
        sr, sp = stats.spearmanr(t, p) if n > 2 else (float('nan'), 1.0)
        pr, pp = stats.pearsonr(t, p) if n > 2 else (float('nan'), 1.0)
        return {'mae': mae, 'r2': r2, 'nrmse': nrmse,
                'spear': sr, 'spear_p': sp, 'pear': pr, 'pear_p': pp}

    @staticmethod
    def log_header():
        print(f"  {'Epoch':>5s} | {'T_loss':>8s} {'V_loss':>8s} "
              f"{'T_mae':>7s} {'V_mae':>7s} | "
              f"{'V_r2':>7s} {'V_spear':>7s} {'V_pear':>7s} {'V_nrmse':>8s} | pat")

    @staticmethod
    def log_epoch(epoch, tr_loss, vl_loss, tr_m, vl_m, patience):
        print(f"  {epoch:5d} | {tr_loss:8.4f} {vl_loss:8.4f} "
              f"{tr_m['mae']:7.3f} {vl_m['mae']:7.3f} | "
              f"{vl_m['r2']:7.3f} {vl_m['spear']:7.3f} "
              f"{vl_m['pear']:7.3f} {vl_m['nrmse']:8.3f} | {patience:2d}")

    def log_fold_test(self, test_true, test_pred):
        m = self.metrics(test_true, test_pred)
        self.fold_metrics.append(m)
        print(f"  >>> test: mae={m['mae']:.3f} r2={m['r2']:+.3f} "
              f"spear={m['spear']:+.3f} pear={m['pear']:+.3f} nrmse={m['nrmse']:.3f}")
        return m

    def log_summary(self, n_folds=None, split_type='gkf'):
        if n_folds is None:
            n_folds = len(self.fold_metrics)
        arrays = {k: np.array([m[k] for m in self.fold_metrics]) for k in _KEYS_REGR}
        title = 'GKF' if split_type == 'gkf' else 'LOSO'
        print(f"\n{'=' * 60}")
        print(f"  {title} RESULT ({n_folds} folds)")
        print(f"  {'':>7s} | {'mean':>8s} {'+-':>2s} {'std':>8s}")
        print(f"  {'':->7s}-+-{'-' * 20}")
        for k in _KEYS_REGR:
            mn, sd = float(np.mean(arrays[k])), float(np.std(arrays[k]))
            print(f"  {k:>7s} | {mn:>8.3f} {'+-':>2s} {sd:>8.3f}")
        if n_folds > 1:
            print()
            hdr = ' | '.join(f"{k:>8s}" for k in _KEYS_REGR)
            print(f"  {'Fold':>7s} | {hdr}")
            print(f"  {'':->7s}-+-{'-' * (9 * len(_KEYS_REGR) - 1)}")
            for fi in range(n_folds):
                vals = ' | '.join(f"{arrays[k][fi]:>8.3f}" for k in _KEYS_REGR)
                print(f"  {fi + 1:>7d} | {vals}")
        print(f"{'=' * 60}")


# ── Legacy aliases (backward compat) ───────────────────────────────────

def classification_metrics(true, pred):
    return ClassificationLogger().metrics(true, pred)

def regression_metrics(true, pred):
    return RegressionLogger().metrics(true, pred)

def subject_aggregate(preds_window, trues_window, cods, subjects):
    true_s, pred_s = [], []
    offset = 0
    for cod in cods:
        nw = len(subjects[cod]['windows'])
        pred_s.append(np.mean(preds_window[offset:offset + nw]))
        sub_trues = np.array(trues_window[offset:offset + nw])
        true_s.append(sub_trues[0] if len(sub_trues) > 0 else subjects[cod].get('cog', 0))
        offset += nw
    return np.array(true_s), np.array(pred_s)

def log_header(mode='regr'):
    (RegressionLogger if mode == 'regr' else ClassificationLogger).log_header()

def log_epoch(epoch, tr_loss, vl_loss, tr_m, vl_m, patience, mode='regr'):
    (RegressionLogger if mode == 'regr' else ClassificationLogger).log_epoch(
        epoch, tr_loss, vl_loss, tr_m, vl_m, patience)

def log_fold_test(test_true, test_pred, mode='regr'):
    logger = RegressionLogger() if mode == 'regr' else ClassificationLogger()
    return logger.log_fold_test(test_true, test_pred)

def log_summary(fold_metrics, n_folds=None, mode='regr', split_type='gkf'):
    logger = RegressionLogger() if mode == 'regr' else ClassificationLogger()
    logger.fold_metrics = fold_metrics
    logger.log_summary(n_folds, split_type)
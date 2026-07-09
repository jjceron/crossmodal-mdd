"""
Plot training curves, confusion matrices, and ROC from saved JSON results.

Usage:
  python src/utils/plot_training.py --type classical_dl --modality eeg --model deepconvnet --fold 1 --metric bacc
  python src/utils/plot_training.py --type classical_dl --modality audio --model deepconvnet --all --metric acc
  python src/utils/plot_training.py --type classical_dl --modality eeg --model deepconvnet --fold 3 --cm
  python src/utils/plot_training.py --type classical_dl --modality audio --model deepconvnet --all --metric roc
  python src/utils/plot_training.py --type classical_ml --model xgboost --all --metric bacc
  python src/utils/plot_training.py --type ocampnet --model cross_attn --all --cm --save_png
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib 
matplotlib.use('Agg' if not os.environ.get('DISPLAY') and os.name != 'nt' else 'TkAgg')
import matplotlib.pyplot as plt

RESULTS_ROOT = 'outputs/results'
FIGURES_ROOT = 'outputs/figures'

METRIC_KEYS = {'bacc': 'val_bacc', 'acc': 'val_acc', 'f1': 'val_f1',
               'sens': 'val_sens', 'spec': 'val_spec'}

TYPE_MAP = {
    'classical_dl': 'classical_dl',
    'classical_ml': 'classical_ml',
    'ocampnet':     'ocampnet',
    'crossmodal_strict': 'crossmodal',
}


def _load_curves(benchmark, model, channels, suffix='ch'):
    path = os.path.join(RESULTS_ROOT, benchmark, f'{model}_{channels}{suffix}_curves.json')
    with open(path) as f:
        return json.load(f)


def _load_results(benchmark, model, channels, suffix='ch'):
    path = os.path.join(RESULTS_ROOT, benchmark, f'{model}_{channels}{suffix}.json')
    with open(path) as f:
        return json.load(f)


def _merge_folds(curves, results):
    """Merge curves history into results folds. Returns list of fold dicts."""
    if results is None:
        return curves['folds'] if curves else []
    merged = []
    for rf in results['folds']:
        fn = rf.get('fold')
        cf = None
        if curves:
            cf = next((c for c in curves.get('folds', []) if c.get('fold') == fn), None)
        entry = dict(rf)
        if cf:
            entry['history'] = cf['history']
        merged.append(entry)
    return merged


def _plot_loss_metric(fig, axes, history, fold_num, metric_label, show_legend=False):
    epochs = np.arange(1, len(history['train_loss']) + 1)
    ax_l, ax_r = axes

    ax_l.plot(epochs, history['train_loss'], color='blue', linestyle='-', label='Train Loss', linewidth=1.5)
    ax_l.plot(epochs, history['val_loss'], color='orange', linestyle='--', label='Val Loss', linewidth=1.5)
    ax_l.set_xlabel('Epoch', fontsize=9)
    ax_l.set_ylabel('Loss', fontsize=9)
    ax_l.set_title(f'Fold {fold_num} — Loss', fontsize=10)
    ax_l.grid(True, alpha=0.3)

    if metric_label == 'roc':
        ax_r.axis('off')
    else:
        mk = METRIC_KEYS[metric_label]
        ax_r.plot(epochs, history['train_acc'], color='blue', linestyle='-', label='Train Acc', linewidth=1.5)
        ax_r.plot(epochs, history[mk], color='orange', linestyle='--', label=f'Val {metric_label.upper()}', linewidth=1.5)
        ax_r.set_xlabel('Epoch', fontsize=9)
        ax_r.set_ylabel(metric_label.upper(), fontsize=9)
        ax_r.set_title(f'Fold {fold_num} — {metric_label.upper()}', fontsize=10)
        ax_r.grid(True, alpha=0.3)

    if show_legend:
        ax_l.legend(fontsize=7, loc='upper right')
        if metric_label != 'roc':
            ax_r.legend(fontsize=7, loc='lower right')


def _plot_cm_pair(fig, axes, fold_entry, fold_num, show_title=True):
    ax_l, ax_r = axes
    cm_w = np.array(fold_entry.get('test_cm_window', fold_entry.get('test_cm', [[0,0],[0,0]])))
    cm_s = np.array(fold_entry.get('test_cm_subject', fold_entry.get('test_cm', [[0,0],[0,0]])))

    for ax, cm, label in [(ax_l, cm_w, 'Windows'), (ax_r, cm_s, 'Subjects')]:
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['HC', 'MDD'])
        ax.set_yticklabels(['HC', 'MDD'])
        title = f'Fold {fold_num} — {label}' if show_title else label
        ax.set_title(title, fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() * 0.5 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_roc_pair(fig, axes, fold_entry, fold_num, show_legend=False):
    ax_l, ax_r = axes

    history = fold_entry.get('history', {})
    if history:
        epochs = np.arange(1, len(history['train_loss']) + 1)
        ax_l.plot(epochs, history['train_loss'], color='blue', linestyle='-', label='Train Loss', linewidth=1.5)
        ax_l.plot(epochs, history['val_loss'], color='orange', linestyle='--', label='Val Loss', linewidth=1.5)
        ax_l.set_xlabel('Epoch', fontsize=9)
        ax_l.set_ylabel('Loss', fontsize=9)
        ax_l.set_title(f'Fold {fold_num} — Loss', fontsize=10)
        ax_l.grid(True, alpha=0.3)
        if show_legend:
            ax_l.legend(fontsize=7, loc='upper right')
    else:
        ax_l.text(0.5, 0.5, 'No training curves', ha='center', va='center', transform=ax_l.transAxes)

    roc = fold_entry.get('test_roc', {})
    if roc and 'y_true' in roc:
        y_true = np.array(roc['y_true'], dtype=np.float64)
        y_prob = np.array(roc['y_prob'], dtype=np.float64)
        thresholds = np.sort(np.unique(y_prob))[::-1]
        tpr, fpr = [0.0], [0.0]
        for t in thresholds:
            yp = (y_prob >= t).astype(np.float64)
            tp = (yp * y_true).sum()
            fp = (yp * (1 - y_true)).sum()
            fn = ((1 - yp) * y_true).sum()
            tn = ((1 - yp) * (1 - y_true)).sum()
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        tpr.append(1.0)
        fpr.append(1.0)
        auc = fold_entry.get('test_roc_auc', 0)
        ax_r.plot(fpr, tpr, color='orange', linewidth=1.5, label=f'ROC (AUC={auc:.3f})')
        ax_r.plot([0, 1], [0, 1], color='blue', linestyle='--', linewidth=1.0, label='Chance')
        ax_r.legend(fontsize=7, loc='lower right')
        ax_r.set_title(f'Fold {fold_num} — ROC', fontsize=10)
    else:
        ax_r.text(0.5, 0.5, 'No ROC data', ha='center', va='center', transform=ax_r.transAxes)
        ax_r.set_title(f'Fold {fold_num} — ROC', fontsize=10)


def main():
    parser = argparse.ArgumentParser(description='Plot training diagnostics')
    parser.add_argument('--type', type=str, required=True,
                        choices=list(TYPE_MAP.keys()),
                        help='Experiment type: classical_dl, classical_ml, ocampnet, crossmodal_strict')
    parser.add_argument('--model', type=str, required=True,
                        help='Model key (e.g. deepconvnet)')
    parser.add_argument('--channels', type=int, default=64,
                        help='Feature count (channels for EEG, mels for audio)')
    parser.add_argument('--modality', type=str, default='eeg', choices=['eeg', 'audio'],
                        help='Data modality (eeg=64ch, audio=64mel)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number (1-based)')
    parser.add_argument('--all', action='store_true',
                        help='Show all folds in a grid')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['bacc', 'acc', 'f1', 'sens', 'spec', 'roc'],
                        help='Metric to plot alongside loss')
    parser.add_argument('--cm', action='store_true',
                        help='Show confusion matrices')
    parser.add_argument('--cm-overall', action='store_true',
                        help='Show overall confusion matrix (aggregated across folds)')
    parser.add_argument('--save_png', action='store_true',
                        help='Save PNG instead of displaying')
    parser.add_argument('--subtype', type=str, default=None,
                        choices=['eeg', 'aud', 'fusion'],
                        help='For crossmodal_strict: which history to plot (eeg, aud, fusion)')
    args = parser.parse_args()

    is_strict = args.type == 'crossmodal_strict'

    if is_strict:
        results_path = os.path.join(RESULTS_ROOT, 'crossmodal', args.model, 'results.json')
        results = json.load(open(results_path)) if os.path.exists(results_path) else None
        curves = None
        if results is None:
            print(f'ERROR: no results at {results_path}')
            sys.exit(1)
        out_dir = os.path.join(FIGURES_ROOT, 'crossmodal', args.model)
        os.makedirs(out_dir, exist_ok=True)
        subtype = args.subtype or 'fusion'
        hist_key = {'eeg': 'eeg_history', 'aud': 'aud_history', 'fusion': 'fusion_history'}[subtype]
        folds_data = results['folds']
        # Validate history exists
        for fe in folds_data:
            if hist_key not in fe or not fe[hist_key]:
                print(f'  Fold {fe.get("fold","?")}: no {hist_key} saved (re-run with updated script)')
    else:
        benchmark = f'{TYPE_MAP[args.type]}/{args.modality}'
        suffix = 'mel' if args.modality == 'audio' else 'ch'
        curves_path = os.path.join(RESULTS_ROOT, benchmark,
                                   f'{args.model}_{args.channels}{suffix}_curves.json')
        results_path = os.path.join(RESULTS_ROOT, benchmark,
                                     f'{args.model}_{args.channels}{suffix}.json')
        curves = json.load(open(curves_path)) if os.path.exists(curves_path) else None
        results = json.load(open(results_path)) if os.path.exists(results_path) else None
        if curves is None and results is None:
            print(f'ERROR: no files found for {args.model}_{args.channels}ch')
            sys.exit(1)
        out_dir = os.path.join(FIGURES_ROOT, benchmark, f'{args.model}_{args.channels}{suffix}')
        os.makedirs(out_dir, exist_ok=True)
        folds_data = _merge_folds(curves, results)
        hist_key = 'history'

    # ── Confusion matrix ───────────────────────────────────────────────
    if args.cm:
        if results is None:
            print('ERROR: no results JSON with confusion matrix data')
            sys.exit(1)

        show_all = args.all or args.fold is None

        if show_all:
            k = len(folds_data)
            fig, axes = plt.subplots(k, 2, figsize=(7, 3.2 * k), constrained_layout=True)
            if k == 1:
                axes = np.array([axes])
            for i, fe in enumerate(folds_data):
                _plot_cm_pair(fig, axes[i], fe, fe.get('fold', i + 1), show_title=k > 1)
            name = 'all_folds_cm'
        else:
            fe = next((f for f in folds_data if f.get('fold') == args.fold), None)
            if fe is None:
                print(f'ERROR: fold {args.fold} not found')
                sys.exit(1)
            fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
            _plot_cm_pair(fig, axes, fe, args.fold)
            name = f'fold{args.fold}_cm'

        if args.save_png:
            fname = os.path.join(out_dir, f'{name}.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'Saved: {fname}')
        else:
            plt.show()
        return

    # ── Overall confusion matrix ────────────────────────────────────────
    if args.cm_overall:
        if results is None:
            print('ERROR: no results JSON with confusion matrix data')
            sys.exit(1)

        folds_data = results['folds']
        cm_w = np.sum([np.array(f.get('test_cm_window', f.get('test_cm', [[0,0],[0,0]]))) for f in folds_data], axis=0)
        cm_s = np.sum([np.array(f.get('test_cm_subject', f.get('test_cm', [[0,0],[0,0]]))) for f in folds_data], axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), constrained_layout=True)
        fe = {'test_cm_window': cm_w, 'test_cm_subject': cm_s}
        _plot_cm_pair(fig, axes, fe, 0, show_title=False)
        axes[0].set_title('Overall — Windows', fontsize=10)
        axes[1].set_title('Overall — Subjects', fontsize=10)
        if args.save_png:
            fname = os.path.join(out_dir, 'overall_cm.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'Saved: {fname}')
        else:
            plt.show()
        return

    # ── ROC curves ─────────────────────────────────────────────────────
    if args.metric == 'roc':
        if results is None:
            print('ERROR: no results JSON with ROC data')
            sys.exit(1)

        show_all = args.all or args.fold is None

        if show_all:
            k = len(folds_data)
            fig, axes = plt.subplots(k, 2, figsize=(10, 3.2 * k), constrained_layout=True)
            if k == 1:
                axes = np.array([axes])
            for i, fe in enumerate(folds_data):
                _plot_roc_pair(fig, axes[i], fe, fe.get('fold', i + 1), show_legend=i == 0)
            name = 'all_folds_roc'
        else:
            fe = next((f for f in folds_data if f.get('fold') == args.fold), None)
            if fe is None:
                print(f'ERROR: fold {args.fold} not found')
                sys.exit(1)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            _plot_roc_pair(fig, axes, fe, args.fold, show_legend=True)
            name = f'fold{args.fold}_roc'

        if args.save_png:
            fname = os.path.join(out_dir, f'{name}.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'Saved: {fname}')
        else:
            plt.show()
        return

    # ── Training curves (bacc, acc, f1, sens, spec) ────────────────────
    if args.metric:
        show_all = args.all or args.fold is None

        if show_all:
            k = len(folds_data)
            fig, axes = plt.subplots(k, 2, figsize=(10, 3 * k), constrained_layout=True)
            if k == 1:
                axes = np.array([axes])
            for i, fe in enumerate(folds_data):
                hist = fe.get(hist_key)
                if hist is None or len(hist.get('train_loss', [])) == 0:
                    ax_l, ax_r = axes[i]
                    ax_l.text(0.5, 0.5, 'No curves', ha='center', va='center', transform=ax_l.transAxes)
                    ax_r.text(0.5, 0.5, 'No curves', ha='center', va='center', transform=ax_r.transAxes)
                    continue
                _plot_loss_metric(fig, axes[i], hist,
                                  fe.get('fold', i + 1), args.metric, show_legend=i == 0)
            name = f'all_folds_{args.metric}'
        else:
            fe = next((f for f in folds_data if f.get('fold') == args.fold), None)
            if fe is None:
                print(f'ERROR: fold {args.fold} not found')
                sys.exit(1)
            hist = fe.get(hist_key)
            if hist is None or len(hist.get('train_loss', [])) == 0:
                print(f'ERROR: fold {args.fold} has no training curves')
                sys.exit(1)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            _plot_loss_metric(fig, axes, hist,
                              args.fold, args.metric, show_legend=True)
            name = f'fold{args.fold}_{args.metric}'

        if args.save_png:
            fname = os.path.join(out_dir, f'{name}.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'Saved: {fname}')
        else:
            plt.show()
        return

    # ── Default: loss + bacc for all folds ─────────────────────────────
    folds_data = folds_data  # already loaded above
    show_all = args.all or args.fold is None

    if show_all:
        k = len(folds_data)
        fig, axes = plt.subplots(k, 2, figsize=(10, 3 * k), constrained_layout=True)
        if k == 1:
            axes = np.array([axes])
        for i, fe in enumerate(folds_data):
            hist = fe.get(hist_key)
            if hist is None or len(hist.get('train_loss', [])) == 0:
                ax_l, ax_r = axes[i]
                ax_l.text(0.5, 0.5, 'No curves', ha='center', va='center', transform=ax_l.transAxes)
                ax_r.text(0.5, 0.5, 'No curves', ha='center', va='center', transform=ax_r.transAxes)
                continue
            _plot_loss_metric(fig, axes[i], hist, fe.get('fold', i + 1), 'bacc', show_legend=i == 0)
        name = 'all_folds_bacc'
    else:
        fe = next((f for f in folds_data if f.get('fold') == args.fold), None)
        if fe is None:
            print(f'ERROR: fold {args.fold} not found')
            sys.exit(1)
        hist = fe.get(hist_key)
        if hist is None or len(hist.get('train_loss', [])) == 0:
            print(f'ERROR: fold {args.fold} has no training curves')
            sys.exit(1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _plot_loss_metric(fig, axes, hist, args.fold, 'bacc', show_legend=True)
        name = f'fold{args.fold}_bacc'

    if args.save_png:
        fname = os.path.join(out_dir, f'{name}.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Saved: {fname}')
    else:
        plt.show()


if __name__ == '__main__':
    main()



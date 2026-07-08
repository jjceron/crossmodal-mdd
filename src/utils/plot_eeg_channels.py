"""Plot EEG channel selections with multi-panel informative visualization.

For FTSM subsets (--channels ftsm4|8|16|32|64):
  3 panels: head map (score-colored) + score ranking bar + DTW cost matrix heatmap

For static selections (--channels 19|64|128):
  2 panels: head map (selected highlighted) + selection index diagram

For comparison (--channels all):
  2x3 grid of head maps for all FTSM subsets (4,8,16,32,64,128)

Usage:
  py src/utils/plot_eeg_channels.py --channels ftsm4
  py src/utils/plot_eeg_channels.py --channels all
  py src/utils/plot_eeg_channels.py --channels 19 --save-plot outputs/plots/19ch.png
  py src/utils/plot_eeg_channels.py --channels ftsm16 --save-plot outputs/plots/ftsm16.png
"""
import sys, os, json, argparse
import numpy as np
import matplotlib
if any(a.startswith('--save') for a in sys.argv):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

EEG_DIR = 'data/raw/modma/MODMA_EEG_BIDS_format/EEG_LZU_2015_2_resting state'
FTSM_RANKING_PATH = 'data/processed/ftsm_ranking.json'
FTSM_COST_PATH = 'data/processed/ftsm_ranking_cost_matrix.npy'
CACHE_DIR = 'data/processed'
N_CHANS = 128

# ── Electrode positions --------------------------------------------------

def _load_electrode_positions():
    """Load 128 EGI electrode positions from first subject's electrodes.tsv.

    Returns:
        names: list of 128 channel names (E1-E128)
        pos_3d: [128, 3] array of (x, y, z) coordinates
    """
    sub_dirs = sorted([
        d for d in os.listdir(EEG_DIR)
        if d.startswith('sub-') and os.path.isdir(os.path.join(EEG_DIR, d))
    ])
    if not sub_dirs:
        raise FileNotFoundError(f'No subject directories found in {EEG_DIR}')

    tsv_path = os.path.join(
        EEG_DIR, sub_dirs[0], 'eeg',
        f'{sub_dirs[0]}_task-Resting-state_electrodes.tsv'
    )
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f'Electrodes TSV not found: {tsv_path}')

    names, pos = [], []
    with open(tsv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                name = parts[0].strip("'")
                if name.startswith('E') and name[1:].isdigit():
                    names.append(name)
                    pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return names, np.array(pos)


def _pos_2d(pos_3d):
    """Project 3D EGI coordinates to 2D head plot (top view).

    Uses x (left-right) and y (anterior-posterior).
    Returns normalized positions in [-1, 1] for plotting.
    """
    x, y = pos_3d[:, 0], pos_3d[:, 1]
    r = np.sqrt(x**2 + y**2)
    max_r = r.max() * 1.05
    return np.column_stack([x / max_r, y / max_r])


# ── Cache helpers --------------------------------------------------------

def _resolve_selection(channel_arg):
    """Parse --channels into (n_selected, selection_name, selected_0based_indices)."""
    v = channel_arg.lower()

    if v == '128':
        return 128, 'All 128 channels', list(range(128))

    if v == '64':
        return 64, 'First 64 channels (E1-E64)', list(range(64))

    if v == '19':
        # Reuse the 10-20 mapping from cache_modma_eeg
        from src.preprocess.cache_modma_eeg import _compute_10_20_indices
        sub_dirs = sorted([
            d for d in os.listdir(EEG_DIR)
            if d.startswith('sub-') and os.path.isdir(os.path.join(EEG_DIR, d))
        ])
        tsv_path = os.path.join(
            EEG_DIR, sub_dirs[0], 'eeg',
            f'{sub_dirs[0]}_task-Resting-state_electrodes.tsv'
        )
        indices = _compute_10_20_indices(tsv_path)
        return 19, '10-20 clinical subset', indices

    if v.startswith('ftsm'):
        k = int(v.replace('ftsm', ''))
        if not os.path.exists(FTSM_RANKING_PATH):
            print(f'ERROR: run ftsm_chselector.py first (no ranking at {FTSM_RANKING_PATH})')
            sys.exit(1)
        with open(FTSM_RANKING_PATH) as f:
            ranking_data = json.load(f)
        ch_1based = ranking_data['nested_subsets'].get(str(k))
        if ch_1based is None:
            print(f'ERROR: no subset for k={k} in ranking')
            sys.exit(1)
        selected = [c - 1 for c in ch_1based]
        return k, f'FTSM Top {k}', selected

    raise ValueError(f'Unknown --channels: {channel_arg}')


# ── Panels ---------------------------------------------------------------

def _draw_head(ax, pos_2d, selected_indices, all_names, scores=None,
               title='Head Map', show_labels=True):
    """Draw a 2D head map with electrode positions.

    Args:
        ax: matplotlib axis
        pos_2d: [128, 2] normalized positions
        selected_indices: list of ints (0-based) to highlight
        all_names: list of 128 channel names
        scores: [128] scores for coloring (None = binary)
        title: plot title
        show_labels: annotate selected channels with names
    """
    # Head outline
    circle = Circle((0, 0), 1.0, fill=False, color='#444', lw=1.5, zorder=0)
    ax.add_patch(circle)
    # Nose
    ax.plot(0, 1.02, '^', color='#444', markersize=6, zorder=1)
    # Ears
    ax.plot(-1.08, 0, '<', color='#444', markersize=6, zorder=1)
    ax.plot(1.08, 0, '>', color='#444', markersize=6, zorder=1)

    n = len(pos_2d)
    selected_set = set(selected_indices)

    if scores is not None:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        colors = plt.cm.viridis(norm_scores)
        sizes = np.full(n, 12)
        sizes[list(selected_set)] = 30
        for i in range(n):
            edge = 'red' if i in selected_set else 'none'
            lw = 1.5 if i in selected_set else 0
            ax.scatter(pos_2d[i, 0], pos_2d[i, 1], c=[colors[i]],
                       s=sizes[i], edgecolors=edge, linewidths=lw,
                       zorder=3 if i in selected_set else 2)
    else:
        for i in range(n):
            c = '#e74c3c' if i in selected_set else '#cccccc'
            s = 30 if i in selected_set else 10
            ax.scatter(pos_2d[i, 0], pos_2d[i, 1], c=c, s=s,
                       edgecolors='none', zorder=3 if i in selected_set else 2)

    if show_labels and len(selected_indices) <= 32:
        for idx in selected_indices:
            x, y = pos_2d[idx]
            ax.annotate(all_names[idx], (x, y),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=5, color='#c0392b', fontweight='bold')

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    # Colorbar for score mode
    if scores is not None:
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(scores.min(), scores.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('DTW cost (lower = more representative)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)


def _draw_score_bars(ax, ranking, selected_count, title='Channel Priority Scores'):
    """Bar chart of all channels sorted by FTSM score.

    Args:
        ax: matplotlib axis
        ranking: list of dicts with 'channel' and 'score'
        selected_count: how many top channels to highlight
    """
    channels = [r['channel'] for r in ranking]
    scores = [r['score'] for r in ranking]
    colors = ['#e74c3c' if i < selected_count else '#cccccc'
              for i in range(len(channels))]

    ax.bar(range(len(channels)), scores, color=colors, width=0.8)
    ax.axvline(selected_count - 0.5, color='red', linestyle='--', alpha=0.5,
               label=f'Top {selected_count}')
    ax.set_xlabel('Channel index (sorted by priority)', fontsize=8)
    ax.set_ylabel('Avg DTW cost (lower = better)', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=7)

    # Annotate top K channel names
    for i in range(min(selected_count, 8)):
        ax.annotate(f'E{channels[i]}', (i, scores[i]),
                    textcoords='offset points', xytext=(0, 3),
                    ha='center', fontsize=5, rotation=45)


def _draw_cost_matrix(ax, cost_matrix, ranking, selected_count,
                      title='DTW Cost Matrix (sorted by priority)'):
    """Heatmap of 128×128 cost matrix with top-K block outlined.

    Args:
        ax: matplotlib axis
        cost_matrix: [128, 128] array
        ranking: list of dicts sorted by priority
        selected_count: number of top channels to outline
    """
    # Reorder channels by priority
    order = [r['channel'] - 1 for r in ranking]
    reordered = cost_matrix[order][:, order]

    im = ax.imshow(reordered, cmap='viridis', aspect='auto',
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    ax.set_xlabel('Channel (sorted by priority)', fontsize=8)
    ax.set_ylabel('Channel (sorted by priority)', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=6)

    # Outline top-K block
    rect = plt.Rectangle((-0.5, -0.5), selected_count, selected_count,
                         fill=False, edgecolor='red', lw=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(selected_count / 2 - 0.5, -3, f'Top {selected_count}',
            ha='center', fontsize=8, color='red', fontweight='bold')


def _draw_selection_diagram(ax, selected_indices, all_names,
                            title='Selected Channels'):
    """Simple horizontal bar showing which indices are selected."""
    n = len(all_names)
    y = np.zeros(n)
    y[selected_indices] = 1
    ax.bar(range(n), y, color='#e74c3c', width=0.8)
    ax.set_xlabel('Channel index (0-based)', fontsize=8)
    ax.set_ylabel('Selected', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['no', 'yes'], fontsize=7)
    ax.tick_params(labelsize=6)
    # Label first few selected
    for i in selected_indices[:10]:
        ax.annotate(all_names[i], (i, 1.02), ha='center', fontsize=5,
                    rotation=45)


def _draw_comparison_table(ax, ranking):
    """Add text table showing paper's claimed vs actual top channels."""
    top4 = [r['channel'] for r in ranking[:4]]
    top8 = [r['channel'] for r in ranking[:8]]
    paper_top4 = [67, 68, 93, 94]
    paper_top8 = [48, 66, 67, 68, 82, 84, 93, 94]

    rows = [
        ['', 'Paper (Esmi)', 'Ours (MODMA)', 'Match'],
        ['Top 4', str(paper_top4), str(top4),
         '✓' if set(top4) == set(paper_top4) else '✗'],
        ['Top 8', str(paper_top8), str(top8),
         '✓' if set(top8) == set(paper_top8) else '✗'],
    ]
    ax.axis('off')
    table = ax.table(cellText=rows, loc='center', cellLoc='center',
                     colWidths=[0.1, 0.3, 0.3, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
    ax.set_title('Channel Selection vs Paper', fontsize=11, fontweight='bold')


# ── Main dispatchers -----------------------------------------------------

def _plot_ftsm(channel_arg, save_path):
    """3-panel plot for FTSM subsets."""
    k, title, selected = _resolve_selection(channel_arg)
    names, pos_3d = _load_electrode_positions()
    pos_2d = _pos_2d(pos_3d)

    with open(FTSM_RANKING_PATH) as f:
        ranking_data = json.load(f)
    ranking = ranking_data['ranking']
    scores = np.array([r['score'] for r in ranking])
    # Map ranking order back to original channel order
    scores_orig = np.full(N_CHANS, np.nan)
    for r in ranking:
        scores_orig[r['channel'] - 1] = r['score']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f'{title}  —  {k} channels selected from {N_CHANS}',
                 fontsize=13, fontweight='bold', y=1.02)

    _draw_head(axes[0], pos_2d, selected, names, scores=scores_orig,
               title=f'Head Map — Top {k}')
    _draw_score_bars(axes[1], ranking, k)
    cost_matrix = np.load(FTSM_COST_PATH)
    _draw_cost_matrix(axes[2], cost_matrix, ranking, k)

    plt.tight_layout()
    _output(fig, save_path)


def _plot_static(channel_arg, save_path):
    """2-panel plot for static selections (19, 64, 128)."""
    n_ch, title, selected = _resolve_selection(channel_arg)
    names, pos_3d = _load_electrode_positions()
    pos_2d = _pos_2d(pos_3d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{title}  —  {n_ch} of {N_CHANS} channels',
                 fontsize=13, fontweight='bold', y=1.02)

    _draw_head(axes[0], pos_2d, selected, names,
               title=f'Head Map — {title}')
    _draw_selection_diagram(axes[1], selected, names,
                            title=f'Selected Channels (n={n_ch})')

    plt.tight_layout()
    _output(fig, save_path)


def _plot_all_ftsm(save_path):
    """2×3 grid of head maps for all 6 FTSM subsets."""
    names, pos_3d = _load_electrode_positions()
    pos_2d = _pos_2d(pos_3d)

    with open(FTSM_RANKING_PATH) as f:
        ranking_data = json.load(f)
    ranking = ranking_data['ranking']
    scores_orig = np.full(N_CHANS, np.nan)
    for r in ranking:
        scores_orig[r['channel'] - 1] = r['score']

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle('FTSM Channel Subsets Comparison  (score-colored: lower DTW cost = higher priority)',
                 fontsize=13, fontweight='bold', y=1.01)

    subsets = [4, 8, 16, 32, 64, 128]
    for ax, k in zip(axes.flat, subsets):
        selected = [c - 1 for c in ranking_data['nested_subsets'][str(k)]]
        _draw_head(ax, pos_2d, selected, names, scores=scores_orig,
                   title=f'Top {k}', show_labels=(k <= 16))

    plt.tight_layout()
    _output(fig, save_path)


def _output(fig, save_path):
    """Save to file or show interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    else:
        plt.show()
    plt.close(fig)


# ── Entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot EEG channel selections (head map + analysis panels)')
    parser.add_argument('--channels', type=str, default='ftsm4',
                        help='Which selection to plot: '
                             '64, 19, 128, ftsm4/8/16/32/64, or "all" for all FTSM subsets')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save figure (default: show window)')
    args = parser.parse_args()

    v = args.channels.lower()

    if v == 'all':
        if not os.path.exists(FTSM_RANKING_PATH):
            print('ERROR: run ftsm_chselector.py first (no ranking found)')
            sys.exit(1)
        _plot_all_ftsm(args.save_plot)
    elif v.startswith('ftsm'):
        if not os.path.exists(FTSM_RANKING_PATH):
            print('ERROR: run ftsm_chselector.py first (no ranking found)')
            sys.exit(1)
        _plot_ftsm(args.channels, args.save_plot)
    elif v in ('19', '64', '128'):
        _plot_static(args.channels, args.save_plot)
    else:
        print(f'ERROR: unknown --channels "{args.channels}"')
        sys.exit(1)


if __name__ == '__main__':
    main()

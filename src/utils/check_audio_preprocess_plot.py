"""Audio preprocessing diagnostic plots for MODMA mel-spectrogram cache.

Generates a 3x2 publication-ready figure from audio_mel_cache.npz.

Usage:
  python src/utils/check_audio_preprocess_plot.py
  python src/utils/check_audio_preprocess_plot.py --save
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg' if not os.environ.get('DISPLAY') and os.name != 'nt' else 'TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CACHE_PATH = 'data/processed/audio_mel_cache.npz'
FIG_DIR = 'outputs/figures'

MDD_C, HC_C = '#c0392b', '#2980b9'
MDD_L, HC_L = '#f5b7b1', '#aed6f1'


def main():
    parser = argparse.ArgumentParser(description='Audio cache diagnostics')
    parser.add_argument('--save', action='store_true', help='Save to outputs/figures/')
    args = parser.parse_args()

    print('Loading audio_mel_cache.npz ...')
    data = np.load(CACHE_PATH, allow_pickle=True)
    windows = list(data['windows'])
    labels = data['labels'].astype(int)

    n_subj = len(windows)
    n_mdd = int(labels.sum())
    n_hc = n_subj - n_mdd
    n_total = sum(w.shape[0] for w in windows)

    # ── Compute derived arrays (no concatenation, per-subject stats) ─
    mdd_idxs = [i for i in range(n_subj) if labels[i] == 1]
    hc_idxs  = [i for i in range(n_subj) if labels[i] == 0]

    mdd_mean = np.mean([windows[i].mean(axis=0) for i in mdd_idxs], axis=0)  # [64, 200]
    hc_mean  = np.mean([windows[i].mean(axis=0) for i in hc_idxs], axis=0)
    diff = mdd_mean - hc_mean           # [64, 200]
    vmax = max(abs(diff.min()), abs(diff.max()))

    # Mel band energy profile (avg over time axis)
    mdd_energy = mdd_mean.mean(axis=1)   # [64]
    hc_energy = hc_mean.mean(axis=1)
    bands = np.arange(64)

    # Per-subject variance (inter-subject, not pooled window variance)
    mdd_sub_means = np.array([windows[i].mean(axis=(0, 2)) for i in mdd_idxs])  # [n_mdd, 64]
    hc_sub_means  = np.array([windows[i].mean(axis=(0, 2)) for i in hc_idxs])
    mdd_band_std = mdd_sub_means.std(axis=0)  # [64]
    hc_band_std  = hc_sub_means.std(axis=0)

    wins_per_subj = np.array([w.shape[0] for w in windows])
    mdd_counts = wins_per_subj[labels == 1]
    hc_counts = wins_per_subj[labels == 0]

    # ── Figure: 3×2 ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    (ax1, ax2), (ax3, ax4), (ax5, ax6) = axes
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    # ===== Panel 1: Mean Mel MDD ===================================
    im1 = ax1.imshow(mdd_mean, aspect='auto', origin='lower', cmap='magma')
    ax1.set_title(f'Mean Mel-Spectrogram — MDD (n={n_mdd})',
                  fontsize=12, fontweight='bold', color=MDD_C)
    ax1.set_ylabel('Mel band', fontsize=10)
    ax1.set_xlabel('Time frame (200 ≈ 2s)', fontsize=10)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label('dB', fontsize=9)

    # ===== Panel 2: Mean Mel HC ====================================
    im2 = ax2.imshow(hc_mean, aspect='auto', origin='lower', cmap='magma')
    ax2.set_title(f'Mean Mel-Spectrogram — HC (n={n_hc})',
                  fontsize=12, fontweight='bold', color=HC_C)
    ax2.set_ylabel('Mel band', fontsize=10)
    ax2.set_xlabel('Time frame (200 ≈ 2s)', fontsize=10)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label('dB', fontsize=9)

    # ===== Panel 3: Difference MDD − HC ============================
    im3 = ax3.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax)
    ax3.set_title(f'Δ Mel-Spectrogram: MDD − HC\n(max |Δ| = {vmax:.1f} dB)',
                  fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mel band', fontsize=10)
    ax3.set_xlabel('Time frame (200 ≈ 2s)', fontsize=10)
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)
    cbar3.set_label('Δ dB', fontsize=9)

    # ===== Panel 4: Mel Band Energy Profile =========================
    ax4.plot(bands, mdd_energy, color=MDD_C, linewidth=2, label='MDD')
    ax4.fill_between(bands, mdd_energy - mdd_band_std,
                     mdd_energy + mdd_band_std, color=MDD_C, alpha=0.12)
    ax4.plot(bands, hc_energy, color=HC_C, linewidth=2, label='HC')
    ax4.fill_between(bands, hc_energy - hc_band_std,
                     hc_energy + hc_band_std, color=HC_C, alpha=0.12)
    ax4.set_title('Mel Band Energy Profile (±1σ)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mel band index', fontsize=10)
    ax4.set_ylabel('Mean power (dB)', fontsize=10)
    ax4.legend(fontsize=10, loc='upper right')
    ax4.set_xlim(0, 63)
    ax4.grid(True, alpha=0.3)

    # ===== Panel 5: Windows per Subject =============================
    positions = [1, 2]
    bp = ax5.boxplot([mdd_counts, hc_counts], positions=positions,
                     widths=0.4, patch_artist=True, medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor(MDD_L)
    bp['boxes'][1].set_facecolor(HC_L)
    bp['boxes'][0].set_edgecolor(MDD_C)
    bp['boxes'][1].set_edgecolor(HC_C)
    bp['boxes'][0].set_linewidth(1.5)
    bp['boxes'][1].set_linewidth(1.5)

    jitter_m = 0.08
    ax5.scatter(np.ones(len(mdd_counts)) + np.random.uniform(-jitter_m, jitter_m, len(mdd_counts)),
                mdd_counts, color=MDD_C, alpha=0.5, s=20, zorder=3)
    ax5.scatter(np.full(len(hc_counts), 2) + np.random.uniform(-jitter_m, jitter_m, len(hc_counts)),
                hc_counts, color=HC_C, alpha=0.5, s=20, zorder=3)

    ax5.set_title('Windows per Subject', fontsize=12, fontweight='bold')
    ax5.set_xticks([1, 2])
    ax5.set_xticklabels([f'MDD\n(n={n_mdd})', f'HC\n(n={n_hc})'], fontsize=11)
    ax5.set_ylabel('Number of windows', fontsize=10)
    ax5.set_ylim(0, max(wins_per_subj) + 20)
    ax5.axhline(y=200, color='green', linestyle='--', linewidth=1.2, alpha=0.6, label='Cap = 200')
    ax5.legend(fontsize=8, loc='lower right')
    ax5.grid(True, axis='y', alpha=0.3)

    # ===== Panel 6: Per-Band Variance Ratio =========================
    eps = 1e-8
    var_ratio = np.log2((mdd_band_std + eps) / (hc_band_std + eps))
    colors_ratio = [MDD_C if v > 0 else HC_C for v in var_ratio]
    ax6.bar(bands, var_ratio, color=colors_ratio, width=0.8, edgecolor='none', alpha=0.85)
    ax6.axhline(y=0, color='black', linewidth=0.8)
    ax6.axhline(y=-1, color='gray', linestyle=':', linewidth=0.7)
    ax6.axhline(y=+1, color='gray', linestyle=':', linewidth=0.7)
    ax6.set_title('Per-Band Variance Ratio (log₂ MDD/HC)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Mel band index', fontsize=10)
    ax6.set_ylabel('log₂(σ²_MDD / σ²_HC)', fontsize=10)
    ax6.set_xlim(0, 63)
    ax6.grid(True, alpha=0.3, axis='y')

    legend_var = [Patch(facecolor=MDD_C, label='MDD more variable'),
                  Patch(facecolor=HC_C, label='HC more variable')]
    ax6.legend(handles=legend_var, fontsize=8, loc='upper right')

    # ── Suptitle with summary stats ────────────────────────────────
    fig.suptitle(f'Audio Preprocessing Diagnostics — MODMA\n'
                 f'{n_subj} subjects ({n_mdd} MDD, {n_hc} HC)  |  '
                 f'{n_total:,} total windows  |  shape = (64 mel × 200 frames ≈ 2s)',
                 fontsize=14, fontweight='bold', y=1.005)

    if args.save:
        os.makedirs(FIG_DIR, exist_ok=True)
        fname = os.path.join(FIG_DIR, 'audio_preprocess_check.png')
        fig.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'\nSaved: {fname}')
    else:
        plt.show()


if __name__ == '__main__':
    main()

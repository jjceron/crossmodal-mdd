#!/usr/bin/env python3
"""Visualization utility for inspecting processed multimodal data (EEG + audio).

Reads pre-computed .npz files from ``data/processed/`` and generates
publication-quality figures for papers / debugging.

Usage examples
--------------
    python src/utils/plot_processed_data.py --modality eeg --figure signal
    python src/utils/plot_processed_data.py --modality eeg --figure heatmap \\
        --subject 3 --window 12
    python src/utils/plot_processed_data.py --modality audio --figure spectrogram \\
        --save audio_example.png --dpi 300
    python src/utils/plot_processed_data.py --modality both --save combined.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────
#  Data loaders
# ────────────────────────────────────────────────────────────


def _validate_range(name: str, value: int, lo: int, hi: int):
    if not (lo <= value < hi):
        raise IndexError(
            f"{name} {value} out of range [{lo}, {hi})"
        )


def load_eeg(processed_dir: Path, filename: str = "eeg_preprocessed_64ch.npz"):
    """Return ``(windows, subject_ids, labels)`` for EEG.

    *windows* is an object array of shape ``(n_subjects,)``; each element
    is a float array ``(n_windows, n_channels, 500)``.
    """
    path = processed_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["windows"], data["subject_ids"], data["labels"]


def load_audio(processed_dir: Path):
    """Return ``(windows, subject_ids, labels)`` for audio.

    *windows* is a float32 array ``(n_subjects, 200, 64, 200)``.
    """
    path = processed_dir / "audio_mel_cache.npz"
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["windows"], data["subject_ids"], data["labels"]


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────


def _setup_ax(ax, hide_axis: bool, xlabel="", ylabel=""):
    if hide_axis:
        ax.set_axis_off()
        return
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=7)


def _scale_to_db(x: np.ndarray) -> np.ndarray:
    """Convert Mel spectrogram (amplitude) to dB-scaled values."""
    mx = np.maximum(x, 1e-10)
    return 10.0 * np.log10(mx)


# ────────────────────────────────────────────────────────────
#  EEG figures
# ────────────────────────────────────────────────────────────


def plot_eeg_signal(
    windows: np.ndarray,
    subject: int,
    window: int,
    *,
    ax: plt.Axes | None = None,
    hide_axis: bool = False,
    n_channels: int = 12,
    **kwargs,
):
    """Overlaid channel traces in different colours (paper-style)."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    sig = windows[subject][window]  # (64, 500)
    n_ch = min(n_channels, sig.shape[0])
    t = np.arange(sig.shape[1])
    for ch in range(n_ch):
        ax.plot(t, sig[ch], linewidth=0.3, label=f"ch{ch}")
    ax.set_xlabel("Samples", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.tick_params(labelsize=7)
    if not kwargs.get('no_legend', False):
        ax.legend(fontsize=5, ncol=min(4, n_ch), loc="upper right",
                  frameon=False)
    if hide_axis:
        ax.set_axis_off()
    plt.tight_layout()


def plot_eeg_heatmap(
    windows: np.ndarray,
    subject: int,
    window: int,
    *,
    ax: plt.Axes | None = None,
    hide_axis: bool = False,
    no_colorbar: bool = False,
    cmap: str = "viridis",
):
    """64 channels × 500 samples heatmap."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    sig = windows[subject][window]
    im = ax.imshow(sig, aspect="auto", cmap=cmap, interpolation="nearest")
    _setup_ax(ax, hide_axis, xlabel="Samples", ylabel="Channels")
    if not no_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()


def plot_eeg_image(
    windows: np.ndarray,
    subject: int,
    window: int,
    *,
    ax: plt.Axes | None = None,
    hide_axis: bool = True,
    no_colorbar: bool = True,
    cmap: str = "viridis",
):
    """Bare image representation — no axes, ready for paper insertion."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    sig = windows[subject][window]
    ax.imshow(sig, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_axis_off()
    plt.tight_layout()


# ────────────────────────────────────────────────────────────
#  Audio figures
# ────────────────────────────────────────────────────────────


def plot_audio_spectrogram(
    windows: np.ndarray,
    subject: int,
    window: int,
    *,
    ax: plt.Axes | None = None,
    hide_axis: bool = False,
    no_colorbar: bool = False,
    cmap: str = "viridis",
):
    """Mel spectrogram (dB) — 64 bands × 200 time frames."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    spec = _scale_to_db(windows[subject, window])  # (64, 200)
    im = ax.imshow(spec, aspect="auto", cmap=cmap, interpolation="nearest")
    _setup_ax(ax, hide_axis, xlabel="Time frames", ylabel="Mel bands")
    if not no_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.7, label="dB")
    plt.tight_layout()


def plot_audio_heatmap(
    windows: np.ndarray,
    subject: int,
    window: int,
    *,
    ax: plt.Axes | None = None,
    hide_axis: bool = False,
    no_colorbar: bool = False,
    cmap: str = "magma",
):
    """Raw mel amplitude as heatmap — different colour scale."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    im = ax.imshow(windows[subject, window], aspect="auto", cmap=cmap,
                   interpolation="nearest")
    _setup_ax(ax, hide_axis, xlabel="Time frames", ylabel="Mel bands")
    if not no_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()


def plot_audio_wavelike(
    windows: np.ndarray,
    subject: int,
    window: int,
    *,
    ax: plt.Axes | None = None,
    hide_axis: bool = False,
    n_bands: int = 10,
    **kwargs,
):
    """Alternative: overlay several Mel-band envelopes as curves."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    spec = windows[subject, window]  # (64, 200)
    n_b = min(n_bands, spec.shape[0])
    t = np.arange(spec.shape[1])
    offset = np.ptp(spec[:n_b]) * 0.6
    for b in range(n_b):
        ax.plot(t, spec[b] + b * offset, linewidth=0.5, label=f"band {b}")
    _setup_ax(ax, hide_axis, xlabel="Time frames", ylabel="Amplitude (offset)")
    ax.set_yticks([])
    if not kwargs.get('no_legend', False):
        ax.legend(fontsize=6, loc="upper right", ncol=2)
    if hide_axis:
        ax.set_axis_off()
    plt.tight_layout()


# ────────────────────────────────────────────────────────────
#  Combined (both modalities, same subject & window)
# ────────────────────────────────────────────────────────────


def plot_both(
    eeg_windows: np.ndarray,
    audio_windows: np.ndarray,
    subject: int,
    window: int,
    *,
    hide_axis: bool = False,
    no_colorbar: bool = False,
    cmap: str = "viridis",
):
    """Two-panel figure: EEG on top, audio spectrogram on bottom."""
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 5))
    plot_eeg_heatmap(eeg_windows, subject, window,
                      ax=ax0, hide_axis=hide_axis,
                      no_colorbar=no_colorbar, cmap=cmap)
    plot_audio_spectrogram(audio_windows, subject, window,
                           ax=ax1, hide_axis=hide_axis,
                           no_colorbar=no_colorbar, cmap=cmap)
    plt.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
#  Figure dispatch
# ────────────────────────────────────────────────────────────

EEG_FIGURES = {
    "signal": plot_eeg_signal,
    "heatmap": plot_eeg_heatmap,
    "image": plot_eeg_image,
}

AUDIO_FIGURES = {
    "spectrogram": plot_audio_spectrogram,
    "heatmap": plot_audio_heatmap,
    "wavelike": plot_audio_wavelike,
}

# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualise processed EEG & audio data for papers / inspection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--modality", choices=["eeg", "audio", "both"],
        default="eeg", help="Which modality to plot (default: eeg).",
    )
    p.add_argument(
        "--figure",
        choices=[*EEG_FIGURES, *AUDIO_FIGURES, "both"],
        default="signal",
        help=(
            "Figure style.  EEG: signal / heatmap / image.  "
            "Audio: spectrogram / heatmap / wavelike.  both: both."
        ),
    )
    p.add_argument("--subject", type=int, default=0,
                    help="Subject index (0-based, default: 0).")
    p.add_argument("--window", type=int, default=0,
                    help="Window index (0-based, default: 0).")
    p.add_argument("--channels", type=int, default=12,
                    help="Number of channels to show in signal figure (default: 12).")
    p.add_argument("--save", type=str, default=None,
                    help="Save to this path (default: interactive display).")
    p.add_argument("--dpi", type=int, default=300,
                    help="Figure resolution (default: 300).")
    p.add_argument("--transparent", action="store_true",
                    help="Transparent figure background.")
    p.add_argument("--hide-axis", action="store_true",
                    help="Remove all axes / ticks.")
    p.add_argument("--no-colorbar", action="store_true",
                    help="Omit colour bar.")
    p.add_argument("--no-legend", action="store_true",
                    help="Omit legend.")
    p.add_argument("--cmap", type=str, default="viridis",
                    help="Matplotlib colormap (default: viridis).")
    p.add_argument(
        "--processed-dir", type=str, default="data/processed",
        help="Directory containing the .npz files (default: data/processed).",
    )
    p.add_argument(
        "--eeg-cache", type=str, default="eeg_preprocessed_64ch.npz",
        help="EEG cache filename (default: eeg_preprocessed_64ch.npz).",
    )
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_dir():
        parser.error(f"processed-dir not found: {processed_dir}")

    # ── Load required files ──
    eeg_windows = eeg_ids = eeg_labels = None
    audio_windows = audio_ids = audio_labels = None

    if args.modality in ("eeg", "both"):
        try:
            eeg_windows, eeg_ids, eeg_labels = load_eeg(processed_dir, args.eeg_cache)
        except FileNotFoundError as exc:
            parser.error(str(exc))

    if args.modality in ("audio", "both"):
        try:
            audio_windows, audio_ids, audio_labels = load_audio(processed_dir)
        except FileNotFoundError as exc:
            parser.error(str(exc))

    # ── Validate indices ──
    if eeg_windows is not None:
        n_eeg = len(eeg_windows)
        _validate_range("subject", args.subject, 0, n_eeg)
        n_win = eeg_windows[args.subject].shape[0]
        _validate_range("window", args.window, 0, n_win)

    if audio_windows is not None:
        n_aud = len(audio_windows)
        _validate_range("subject (audio)", args.subject, 0, n_aud)
        _validate_range("window (audio)", args.window, 0, audio_windows.shape[1])

    # ── Dispatch ──
    kw = dict(hide_axis=args.hide_axis, no_colorbar=args.no_colorbar,
              no_legend=args.no_legend, cmap=args.cmap)

    if args.modality == "both":
        fig = plot_both(eeg_windows, audio_windows, args.subject, args.window,
                        **kw)
    elif args.modality == "eeg":
        fn = EEG_FIGURES.get(args.figure)
        if fn is None:
            parser.error(f"Figure '{args.figure}' not available for EEG. "
                         f"Choose from: {', '.join(EEG_FIGURES)}")
        fig, ax = plt.subplots(1, 1)
        if args.figure == "signal":
            fn(eeg_windows, args.subject, args.window, ax=ax,
               n_channels=args.channels, **kw)
        else:
            fn(eeg_windows, args.subject, args.window, ax=ax, **kw)
    else:  # audio
        fn = AUDIO_FIGURES.get(args.figure)
        if fn is None:
            parser.error(f"Figure '{args.figure}' not available for audio. "
                         f"Choose from: {', '.join(AUDIO_FIGURES)}")
        fig, ax = plt.subplots(1, 1)
        fn(audio_windows, args.subject, args.window, ax=ax, **kw)

    # ── Save or display ──
    if args.save:
        fig.savefig(args.save, dpi=args.dpi, transparent=args.transparent,
                    bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()

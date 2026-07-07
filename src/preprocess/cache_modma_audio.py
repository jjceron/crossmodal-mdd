"""Cache audio mel-spectrogram windows for fast DL training — v1.

Preprocessing pipeline:
  1. Read WAV (44.1kHz, mono) → normalize int32 to float
  2. Resample 44.1kHz → 16kHz
  3. Mel-spectrogram: 64 bands, n_fft=1024, hop=160, f_min=20, f_max=8000
  4. Amplitude to dB (top_db=80)
  5. Sliding windows: 200 frames (~2s), 50% overlap
  6. Concatenate across 29 WAVs per subject, cap at 200 windows

Run once:
  py src/preprocess/cache_modma_audio.py

Output:
  data/processed/audio_mel_cache.npz
"""
import sys, os, glob, numpy as np, pandas as pd, warnings
import scipy.io.wavfile as wav
from scipy.io.wavfile import WavFileWarning
import torch, torchaudio
 
warnings.filterwarnings('ignore', category=WavFileWarning)
sys.path.insert(0, '.')
AUDIO_DIR = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015'
AUDIO_XLSX = 'data/raw/modma/854301_EEG_3Channels_Resting_Lanzhou_2015/854301_Audio_Lanzhou_2015/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx'

SR_TARGET = 16000; N_MELS = 64; N_FFT = 1024; HOP = 160; N_FRAMES = 200
OVERLAP = 0.5; N_WINS = 200; RANDOM_STATE = 42


def compute_mel(wav_path):
    try:
        sr, audio = wav.read(wav_path)
    except Exception:
        return None
    if len(audio.shape) > 1:
        audio = audio.mean(axis=-1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    wave_t = torch.from_numpy(audio).float().unsqueeze(0)
    if sr != SR_TARGET:
        wave_t = torchaudio.transforms.Resample(sr, SR_TARGET)(wave_t)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR_TARGET, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, power=2.0, f_min=20, f_max=8000)(wave_t[0])
    return torchaudio.transforms.AmplitudeToDB(top_db=80)(mel).numpy().astype(np.float32)


def main():
    df = pd.read_excel(AUDIO_XLSX).sort_values('subject id')
    y_bin = [1 if lbl == 'MDD' else 0 for lbl in df['type'].tolist()]
    sub_dirs = sorted(glob.glob(os.path.join(AUDIO_DIR, '020*')))

    all_wins_list = []
    all_ids = []
    all_labels = []

    for (sd, lbl) in zip(sub_dirs, y_bin):
        sid = os.path.basename(sd)
        wavs = sorted(glob.glob(os.path.join(sd, '*.wav')))
        all_win = []
        for wf in wavs:
            mel = compute_mel(wf)
            if mel is None:
                continue
            if mel.shape[1] < N_FRAMES:
                continue
            stride = int(N_FRAMES * (1 - OVERLAP))
            n_w = (mel.shape[1] - N_FRAMES) // stride + 1
            if n_w < 1:
                continue
            win = np.lib.stride_tricks.sliding_window_view(mel, N_FRAMES, axis=1)
            win = win[:, ::stride].transpose(1, 0, 2)[:n_w].astype(np.float32)
            all_win.append(win)
        if not all_win:
            continue
        all_win = np.concatenate(all_win, axis=0)
        if all_win.shape[0] > N_WINS:
            rng = np.random.RandomState(RANDOM_STATE)
            idx = rng.choice(all_win.shape[0], N_WINS, replace=False)
            all_win = all_win[idx]
        all_wins_list.append(all_win)
        all_ids.append(sid)
        all_labels.append(lbl)
        print(f"  {sid}: {all_win.shape[0]} windows, label={lbl}")

    os.makedirs('data/processed', exist_ok=True)
    out = {
        'windows': np.array(all_wins_list),
        'subject_ids': np.array(all_ids),
        'labels': np.array(all_labels, dtype=np.int32),
    }
    np.savez('data/processed/audio_mel_cache.npz', **out, allow_pickle=True)
    n_mdd = sum(all_labels)
    print(f"\nCached {len(all_ids)} subjects ({n_mdd} MDD, {len(all_ids)-n_mdd} HC)")
    print(f"Window shape: {all_wins_list[0].shape[1:]}")


if __name__ == '__main__':
    main()



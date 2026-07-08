# Best Unimodal Models — Reference

Methodologically clean baselines (nested 5-fold CV, no leakage).

---

## EEG — DeepConvNet

### Preprocessing

| Step | Parameter | Value |
|------|-----------|-------|
| Raw sampling rate | — | 250 Hz |
| Bandpass filter | — | 0.5 – 60 Hz |
| Notch filter | — | 50 Hz |
| Channels | first 64 of 128 (EGI GSN HydroCel) | 64 |
| Reference | average | — |
| Window length | 2.0 s | 500 samples |
| Overlap | 50% | stride = 250 samples |
| Windows per subject | all retained | ~300 avg (53–335 range) |
| Normalization | z-score per window | (x - μ) / (σ + 1e-8) |
| Subjects | 53 (24 MDD, 29 HC) | — |
| Cache file | `data/processed/eeg_preprocessed_64ch.npz` | — |

### Architecture: DeepConvNet (reduced)

Schirrmeister et al. 2017, reduced for 6GB GPU.

```
Input:  [B, 1, 64, 500]

Block1: Conv2d(1→8, (1,10)) + BatchNorm + ELU + MaxPool((1,3)) + Dropout(0.25)
Block2: Conv2d(8→16, (64,1)) + BatchNorm + ELU + MaxPool((1,3)) + Dropout(0.25)
Block3: Conv2d(16→32, (1,10)) + BatchNorm + ELU + MaxPool((1,3)) + Dropout(0.5)
Block4: Conv2d(32→64, (1,10)) + BatchNorm + ELU + MaxPool((1,3)) + Dropout(0.5)

Output features: [B, 128]
Classifier: Linear(128 → 1)
```

Total params: ~34K

### Training

| Hyperparameter | Value |
|----------------|-------|
| Validation | Nested 5-fold (outer) + 3-fold (inner) |
| Optimizer | AdamW (foreach=False) |
| Learning rate | 5e-4 |
| Weight decay | 1e-3 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 15 (val bacc) |
| Scheduler | ReduceLROnPlateau(factor=0.5, patience=5) |
| Seed | 42 |
| Label smoothing | y * 0.95 + 0.025 |
| Gradient clipping | 1.0 |

### Result

**bacc = 0.685 ± 0.073**

| Fold | bacc |
|------|------|
| 1 | 0.700 |
| 2 | 0.583 |
| 3 | 0.708 |
| 4 | 0.633 |
| 5 | 0.800 |

### Files

```
outputs/results/classical_dl/trained_eeg/deepconvnet_64ch/
├── config.json
├── fold_1.pt
├── fold_2.pt
├── fold_3.pt
├── fold_4.pt
└── fold_5.pt
```

---

## Audio — ShallowConvNet

### Preprocessing

| Step | Parameter | Value |
|------|-----------|-------|
| Resampling | — | 16 kHz |
| Mel spectrogram | n_fft=1024, hop=160, f_min=20, f_max=8000 | 64 bands |
| Window length | 200 frames | ~2.0 s |
| Overlap | 50% | stride = 100 frames |
| Windows per subject | capped at 200 | 200 avg |
| Normalization | z-score per window | (x - μ) / (σ + 1e-8) |
| Subjects | 52 (23 MDD, 29 HC) | — |
| Cache file | `data/processed/audio_mel_cache.npz` | — |

### Architecture: ShallowConvNet (v2018)

```
Input:  [B, 1, 64, 200]

TemporalConv: Conv2d(1→24, (1,13), padding='same')
SpatialConv:  Conv2d(24→24, (64,1))
BatchNorm + Square
AvgPool: ((1,35), stride=(1,7))
Log + Dropout(0.5)

Output features: [B, 576]
Classifier: Linear(576 → 1)
```

### Training

Same hyperparameters as EEG DeepConvNet (nested 5-fold CV).

### Result

**bacc = 0.687 ± 0.068**

### Files

```
outputs/results/classical_dl/trained_audio/shallowconvnet_64mel/
├── config.json
├── fold_1.pt
├── fold_2.pt
├── fold_3.pt
├── fold_4.pt
└── fold_5.pt
```

# Orchestrated Cross-modal Attention for Depression Assessment Using Multimodal Physiological Signals

[![CI](https://github.com/jjceron/crossmodal-mdd/actions/workflows/ci.yml/badge.svg)](https://github.com/jjceron/crossmodal-mdd/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ruff](https://img.shields.io/badge/Lint-Ruff-D7FF64?logo=ruff)](https://docs.astral.sh/ruff/)
[![CML](https://img.shields.io/badge/MLOps-CML-5A67D8)](https://cml.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Motivation

Major depressive disorder (MDD) is among the leading causes of disability worldwide, yet objective diagnostic tools remain scarce. Electroencephalography (EEG) captures neural dynamics associated with affective regulation, while speech recordings encode paralinguistic biomarkers of depression. Together, these modalities provide complementary information about the same underlying disorder.

Most existing multimodal approaches combine EEG and speech only after each modality has already been independently compressed into a single embedding or prediction. Such late fusion strategies discard potentially informative interactions between cortical activity and acoustic representations.

This repository investigates an **orchestrated cross-modal attention** architecture operating directly at the node level, where every EEG channel attends to every Mel-frequency band and vice versa before subject-level aggregation. All experiments are evaluated under a strict nested cross-validation protocol designed to eliminate subject leakage.

## Method Overview

- **Stage 1 — Preprocessing:** raw EEG (64 channels, 250 Hz) is bandpass filtered (0.5–45 Hz), notch filtered (50 Hz), average referenced, segmented into 2-second windows with 50% overlap, and normalized per window. Audio recordings (44.1 kHz) are resampled to 16 kHz and converted into log-Mel spectrograms (64 Mel bands, `n_fft=1024`, `hop=160`) using the same temporal segmentation.

- **Stage 2 — Channel-preserving encoders:** DeepConvNet (EEG) and ShallowConvNet (audio) preserve the spatial dimension of each modality while learning latent representations for every EEG channel and Mel band independently.

- **Stage 3 — Strict nested cross-validation:** a subject-wise Stratified Group K-Fold split prevents information leakage between train and test subjects. Inner folds are used exclusively for early stopping and hyperparameter selection.

- **Stage 4 — Backbone pre-training:** unimodal encoders are trained using all available subjects of their corresponding modality (paired and unpaired), then frozen before multimodal training.

- **Stage 5 — Cross-modal fusion:** projected node embeddings interact through bidirectional multi-head cross-attention, optionally followed by self-attention layers. Window representations are aggregated and classified through a lightweight multilayer perceptron.

- **Stage 6 — Subject-level inference:** window predictions are averaged to produce a single probability per subject, matching the clinical diagnosis task.

## Evaluation Protocol

Experiments follow a strict **zero-information leakage** protocol.

The paired subset (38 subjects: 17 MDD and 21 Healthy Controls) defines the outer 5-fold Stratified Group K-Fold evaluation. The remaining EEG-only (15) and audio-only (14) participants are used exclusively during unimodal backbone pre-training.

For every outer fold:

- unimodal backbones are trained on approximately 60 subjects;
- backbone weights are frozen;
- only paired training subjects are used to train the fusion module;
- early stopping is performed using inner validation folds.

Performance is reported using:

- Balanced Accuracy (primary metric)
- Accuracy
- F1-score
- Sensitivity
- Specificity
- ROC-AUC

The main comparison is against unimodal DeepConvNet (EEG) and ShallowConvNet (speech) trained under the exact same evaluation protocol.

## Key Results

*Work in progress.*

## Experiment Reports

Selected experiments generate standardized reports through `scripts/generate_report.py`.

Each report includes:

- experiment configuration
- hyperparameters
- validation and test metrics
- confusion matrix
- ROC curve
- learning curves
- execution metadata (date, runtime and Git commit)

Reports are stored inside the project's `results/` directory and can be automatically published as CML comments during Pull Requests.

## Repository Structure

```text
crossmodal-mdd/
├── src/
│   ├── models/          — Neural network architectures
│   ├── training/        — Training pipelines
│   └── utils/           — Evaluation and utility functions
├── scripts/             — Training, analysis and report generation
├── configs/             — Hyperparameter configurations
├── data/                — Local datasets (not tracked)
├── results/             — Published experiment reports and figures
├── docs/                — Paper source and documentation
├── tests/               — Unit and integration tests
├── .github/             — GitHub Actions workflows (CI, CML)
└── requirements.txt     — Project dependencies
```

## Data

This repository uses the **Multi-modal Open Dataset for Mental-disorder Analysis (MODMA)**, a publicly available multimodal dataset developed for depression research using physiological, behavioral and clinical data ([Hu, 2022](#ref-hu2022); [Cai et al., 2020](#ref-cai2020)).

The complete MODMA collection includes multiple partially overlapping modalities:

- 128-channel resting-state EEG
- 128-channel task EEG
- 3-channel portable EEG
- Clinical interview speech recordings
- Psychometric and demographic assessments

This project focuses on the paired subset containing **resting-state EEG and speech recordings**, following a strict subject-level evaluation protocol to prevent information leakage. Unpaired EEG-only and speech-only subjects are used exclusively during unimodal backbone pre-training and are never included in multimodal evaluation.

All data are obtained from the official MODMA repository under its corresponding data-use agreement. Participants provided informed consent, the study received institutional ethical approval, and personally identifiable information was removed before public release. Users of this repository are responsible for complying with the MODMA license and usage conditions.

## Requirements

Python 3.11 or newer is required.

Core dependencies include:

- PyTorch 2.x
- NumPy
- SciPy
- scikit-learn
- matplotlib
- librosa
- soundfile

Complete installation instructions are available in `requirements.txt`.

## Authors and Affiliation

This project is part of **ACEMATE**, a research program of the Master's in Electrical Engineering at **Universidad Tecnológica de Panamá**.

- **Cerón-Ordoñez, J. J.** <a href="https://orcid.org/0009-0009-7320-4809"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" width="16" alt="ORCID"/></a>

- **Cárdenas-Peña, D. A.** <a href="https://orcid.org/0000-0002-0522-8683"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" width="16" alt="ORCID"/></a>

## Bibliography

<a id="ref-hu2022"></a>

### Hu (2022) — MODMA Dataset

```bibtex
@misc{Hu2022MODMA,
  author       = {Hu, Bin},
  title        = {{Multi-modal Open Dataset for Mental-disorder Analysis, Experimental Data 2014--2016}},
  year         = {2022},
  publisher    = {UK Data Service},
  address      = {Colchester, Essex},
  howpublished = {Data Collection},
  doi          = {10.5255/UKDA-SN-854301},
  url          = {https://doi.org/10.5255/UKDA-SN-854301},
  note         = {Alternative title: Multi-modal Open Dataset for Mental-disorder Analysis (MODMA)}
}
```

Official dataset: https://reshare.ukdataservice.ac.uk/854301/

<a id="ref-cai2020"></a>

### Cai et al. (2020) — Dataset Description

```bibtex
@article{Cai2020MODMA,
  author  = {Hanshu Cai and Yiwen Gao and Shuting Sun and Na Li and Fuze Tian and Bin Hu},
  title   = {MODMA Dataset: A Multi-modal Open Dataset for Mental-disorder Analysis},
  journal = {arXiv preprint},
  volume  = {arXiv:2002.09283},
  year    = {2020},
  url     = {https://arxiv.org/abs/2002.09283}
}
```
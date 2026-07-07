from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from io import StringIO
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

mne.set_log_level("WARNING")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "data/raw/modma/MODMA_EEG_BIDS_format"
    / "EEG_LZU_2015_2_resting state"
)

LABEL_MAP = {"MDD": 1, "NC": 0, "HC": 0}


class MODMADataset(Dataset):
    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        lowcut: float | None = 0.5,
        highcut: float | None = 60.0,
        notch: float | None = 50.0,
        target_fs: float | None = None,
        duration_sec: float | None = None,
    ):
        self.root = Path(root)
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch = notch
        self.target_fs = None if target_fs is None else float(target_fs)
        self.duration_sec = duration_sec

        if not self.root.exists():
            raise FileNotFoundError(f"MODMA root does not exist: {self.root}")

        self.participants = self._load_participants()
        self.records = self._discover_records()

        if len(self.records) == 0:
            raise ValueError(f"No EDF records found in {self.root}")

        self.channel_names = self._resolve_channel_names()
        self.samples = []
        self._load_all()
        self._to_tensor()

    def _load_participants(self) -> pd.DataFrame:
        path = self.root / "participants.tsv"

        if not path.exists():
            raise FileNotFoundError(f"participants.tsv not found at {path}")

        raw = path.read_text(encoding="utf-8")
        raw = re.sub(r"\t+", "\t", raw)
        df = pd.read_csv(StringIO(raw), sep="\t")
        df = df.dropna(subset=["group"])
        df["label"] = df["group"].map(LABEL_MAP)

        missing = df["label"].isna()

        if missing.any():
            bad = df.loc[missing, "group"].unique().tolist()
            raise ValueError(f"Unknown group(s) in participants.tsv: {bad}. "
                             f"Expected one of {list(LABEL_MAP.keys())}")

        df["label"] = df["label"].astype(int)
        return df

    def _discover_records(self) -> list[dict]:
        records = []

        for _, row in self.participants.iterrows():
            pid = row["participant_id"]
            label = int(row["label"])
            edf_path = self.root / pid / "eeg" / f"{pid}_task-Resting-state_eeg.EDF"
            json_path = self.root / pid / "eeg" / f"{pid}_task-Resting-state_eeg.json"

            if not edf_path.exists():
                print(f"Warning: EDF not found for {pid}, skipping")
                continue

            records.append({
                "participant_id": pid,
                "label": label,
                "edf_path": edf_path,
                "json_path": json_path if json_path.exists() else None,
            })

        return records

    def _resolve_channel_names(self) -> list[str]:
        first_raw = self._read_header(self.records[0]["edf_path"])
        ref_channels = get_eeg_channel_names(first_raw)
        common = set(ref_channels)

        for rec in self.records[1:]:
            raw = self._read_header(rec["edf_path"])
            common &= set(get_eeg_channel_names(raw))

        common_list = [ch for ch in ref_channels if ch in common]

        if len(common_list) == 0:
            raise ValueError("No common EEG channels found across all subjects")

        return common_list

    def _load_all(self) -> None:
        for rec in self.records:
            eeg = self._process_edf(rec["edf_path"])

            if eeg is None:
                print(f"Warning: failed to load {rec['participant_id']}, skipping")
                continue

            self.samples.append({
                "participant_id": rec["participant_id"],
                "eeg": torch.tensor(eeg, dtype=torch.float32),
                "label": torch.tensor(rec["label"], dtype=torch.long),
            })

    def _process_edf(self, path: Path) -> np.ndarray | None:
        with mne.utils.use_log_level("WARNING"):
            raw = mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")

        missing = [ch for ch in self.channel_names if ch not in raw.ch_names]

        if missing:
            print(f"  Missing channels in {path.name}: {missing}")
            return None

        raw.pick(self.channel_names)
        raw.reorder_channels(self.channel_names)
        raw.set_eeg_reference("average", verbose=False)

        fs = float(raw.info["sfreq"])

        if self.notch is not None and self.notch < fs / 2:
            raw.notch_filter([self.notch], verbose=False)

        highcut = self.highcut

        if highcut is not None:
            highcut = min(highcut, fs / 2 - 1e-3)

            if self.lowcut is not None and highcut <= self.lowcut:
                highcut = None

        if self.lowcut is not None or highcut is not None:
            raw.filter(
                l_freq=self.lowcut,
                h_freq=highcut,
                fir_design="firwin",
                verbose=False,
            )

        if self.target_fs is not None and not np.isclose(fs, self.target_fs):
            raw.resample(self.target_fs, npad="auto", verbose=False)

        eeg = raw.get_data().astype(np.float32)
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

        if self.duration_sec is not None:
            final_fs = float(raw.info["sfreq"])
            n_samples = int(float(self.duration_sec) * final_fs)
            eeg = eeg[:, :n_samples]

        return eeg

    def _to_tensor(self) -> None:
        min_t = min(s["eeg"].shape[1] for s in self.samples)

        for s in self.samples:
            s["eeg"] = s["eeg"][:, :min_t].contiguous()

    @staticmethod
    def _read_header(path: Path):
        with mne.utils.use_log_level("WARNING"):
            return mne.io.read_raw_edf(path.as_posix(), preload=False, verbose="ERROR")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s["participant_id"], s["eeg"], s["label"]


def get_eeg_channel_names(raw) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    if len(picks) == 0:
        return list(raw.ch_names)

    return [raw.ch_names[i] for i in picks]


def split_into_windows(eeg: torch.Tensor, window_len: int, stride: int) -> list[torch.Tensor]:
    C, T = eeg.shape
    windows = []
    for start in range(0, T - window_len + 1, stride):
        windows.append(eeg[:, start:start + window_len].contiguous())
    return windows


class WindowedSubsetDataset(Dataset):
    def __init__(self, eeg_list, label_list, subject_names, window_len, stride):
        self.windows: list[torch.Tensor] = []
        self.window_labels: list[int] = []
        self.window_subjects: list[str] = []
        for eeg, lbl, name in zip(eeg_list, label_list, subject_names):
            for w in split_into_windows(eeg, window_len, stride):
                self.windows.append(w)
                self.window_labels.append(lbl)
                self.window_subjects.append(name)
        self.labels = torch.tensor(self.window_labels, dtype=torch.long)
        self.names = self.window_subjects

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.window_subjects[idx], self.windows[idx], self.labels[idx]


def create_windowed_dataloaders(
    dataset: MODMADataset,
    k_folder: int = 5,
    batch_size: int = 16,
    shuffle: bool = True,
    split_seed: int = 42,
    inner_split: int = 5,
    window_samples: int = 500,
    stride: int = 500,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> list[tuple[DataLoader, DataLoader, DataLoader]]:
    subjects, labels, eeg_data = [], [], []
    for s in dataset.samples:
        subjects.append(s["participant_id"])
        labels.append(int(s["label"].item()))
        eeg_data.append(s["eeg"])

    outer_gkf = StratifiedGroupKFold(
        n_splits=k_folder, shuffle=shuffle, random_state=split_seed,
    )
    folds: list[tuple[DataLoader, DataLoader, DataLoader]] = []

    for train_val_idx, test_idx in outer_gkf.split(eeg_data, labels, groups=subjects):
        inner_gkf = StratifiedGroupKFold(
            n_splits=inner_split, shuffle=shuffle, random_state=split_seed,
        )
        train_idx, val_idx = next(inner_gkf.split(
            [eeg_data[i] for i in train_val_idx],
            [labels[i] for i in train_val_idx],
            groups=[subjects[i] for i in train_val_idx],
        ))
        train_subjects = [train_val_idx[i] for i in train_idx]
        val_subjects = [train_val_idx[i] for i in val_idx]

        train_ds = WindowedSubsetDataset(
            [eeg_data[i] for i in train_subjects],
            [labels[i] for i in train_subjects],
            [subjects[i] for i in train_subjects],
            window_samples, stride,
        )
        val_ds = WindowedSubsetDataset(
            [eeg_data[i] for i in val_subjects],
            [labels[i] for i in val_subjects],
            [subjects[i] for i in val_subjects],
            window_samples, stride,
        )
        test_ds = WindowedSubsetDataset(
            [eeg_data[i] for i in test_idx],
            [labels[i] for i in test_idx],
            [subjects[i] for i in test_idx],
            window_samples, stride,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)
        folds.append((train_loader, val_loader, test_loader))

    return folds


def create_dataloaders(
    dataset: MODMADataset,
    k_folder: int = 5,
    batch_size: int = 16,
    shuffle: bool = True,
    split_seed: int = 42,
    inner_split: int = 5,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> list[tuple[DataLoader, DataLoader, DataLoader]]:
    subjects, labels, eeg_data = [], [], []

    for s in dataset.samples:
        subjects.append(s["participant_id"])
        labels.append(int(s["label"].item()))
        eeg_data.append(s["eeg"])

    outer_gkf = StratifiedGroupKFold(
        n_splits=k_folder,
        shuffle=shuffle,
        random_state=split_seed,
    )

    folds: list[tuple[DataLoader, DataLoader, DataLoader]] = []

    class FoldDataset(Dataset):
        def __init__(self, eeg_list, label_list, subject_indices):
            self.X, self.y, self.names = [], [], []

            for idx in subject_indices:
                self.X.append(eeg_list[idx])
                self.y.append(label_list[idx])
                self.names.append(subjects[idx])

            self.y = torch.tensor(self.y, dtype=torch.long)
            self.X = torch.stack(self.X)

        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            return self.names[idx], self.X[idx], self.y[idx]

    for train_val_idx, test_idx in outer_gkf.split(
        eeg_data,
        labels,
        groups=subjects,
    ):
        inner_gkf = StratifiedGroupKFold(
            n_splits=inner_split,
            shuffle=shuffle,
            random_state=split_seed,
        )

        train_idx, val_idx = next(
            inner_gkf.split(
                [eeg_data[i] for i in train_val_idx],
                [labels[i] for i in train_val_idx],
                groups=[subjects[i] for i in train_val_idx],
            )
        )

        train_subjects = [train_val_idx[i] for i in train_idx]
        val_subjects = [train_val_idx[i] for i in val_idx]

        train_dataset = FoldDataset(eeg_data, labels, train_subjects)
        val_dataset = FoldDataset(eeg_data, labels, val_subjects)
        test_dataset = FoldDataset(eeg_data, labels, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=60.0)
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--target-fs", type=float, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = MODMADataset(
        root=args.root,
        lowcut=args.lowcut,
        highcut=args.highcut,
        notch=args.notch,
        target_fs=args.target_fs,
        duration_sec=args.duration_sec,
    )

    print(f"\nMODMA Dataset loaded:")
    print(f"  Subjects: {len(dataset)}")
    print(f"  Channels: {len(dataset.channel_names)}")
    print(f"  Tensor shape: {dataset.samples[0]['eeg'].shape}")

    labels = [int(s["label"].item()) for s in dataset.samples]
    counts = Counter(labels)
    print(f"  HC (0): {counts.get(0, 0)}")
    print(f"  MDD (1): {counts.get(1, 0)}")

    folds = create_dataloaders(
        dataset=dataset,
        k_folder=args.k,
        batch_size=args.batch_size,
        inner_split=args.inner_splits,
    )

    print(f"\nCreated {len(folds)} folds")

    for fold_id, (train_loader, val_loader, test_loader) in enumerate(folds):
        train_y = train_loader.dataset.y.tolist()
        train_counts = Counter(train_y)
        val_y = val_loader.dataset.y.tolist()
        val_counts = Counter(val_y)
        test_y = test_loader.dataset.y.tolist()
        test_counts = Counter(test_y)

        print(
            f"  Fold {fold_id:02d} | "
            f"Train: HC={train_counts.get(0, 0):2d} MDD={train_counts.get(1, 0):2d} | "
            f"Val:   HC={val_counts.get(0, 0):2d} MDD={val_counts.get(1, 0):2d} | "
            f"Test:  HC={test_counts.get(0, 0):2d} MDD={test_counts.get(1, 0):2d}"
        )


if __name__ == "__main__":
    main()

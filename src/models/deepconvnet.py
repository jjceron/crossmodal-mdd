"""DeepConvNet (Schirrmeister et al., 2017) — compact for small-sample EEG.

Original: 16→32→64→128 filters, ~136K params
v4:        8→16→32→64  filters,  ~34K params
v5:        4→8→16→32   filters,  ~8.5K params
v6:        4→8→16→64   filters,  ~14K params  (128-d features, balanced with audio)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DeepConvNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 1,
        n_samples: int = 500,
        dropout: float = 0.5,
    ) -> None: 
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 4, (1, 10)),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout2d(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(4, 8, (n_channels, 1)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout2d(dropout),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(8, 16, (1, 10)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout2d(dropout),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(16, 64, (1, 10)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout2d(dropout),
        )

        dummy = torch.randn(1, 1, n_channels, n_samples)
        with torch.no_grad():
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
        self.fc_features = int(x.numel())
        self.classifier = nn.Linear(self.fc_features, n_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))



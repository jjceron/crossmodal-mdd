from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 56,
        n_classes: int = 3,
        n_samples: int = 385,
        temporal_filters: int = 25,
        temporal_kernel: int = 25,
        depth_multiplier: int = 1,
        pool_size: int = 40,
        pool_stride: int = 20,
        dropout: float = 0.5,
        lstm_units_1: int = 10,
        lstm_units_2: int = 10,
    ) -> None:
        super().__init__()

        temporal_pad = temporal_kernel // 2
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=temporal_filters,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_pad),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(temporal_filters)

        self.spatial_conv = nn.Conv2d(
            in_channels=temporal_filters,
            out_channels=temporal_filters * depth_multiplier,
            kernel_size=(n_channels, 1),
            groups=temporal_filters,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(temporal_filters * depth_multiplier)

        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(
            input_size=temporal_filters * depth_multiplier,
            hidden_size=lstm_units_1,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm_units_1,
            hidden_size=lstm_units_2,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Linear(lstm_units_2, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.pool(x)
        x = self.dropout(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        x = self.classifier(x)
        return x

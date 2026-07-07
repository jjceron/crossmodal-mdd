from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 2,
        n_samples: int = 256,
        dropout: float = 0.5,
        version: str = "2018",
    ) -> None:
        super().__init__()

        if version == "2017":
            bias_spatial = False
            pool = (1, 75)
            stride = (1, 15)
            kern = 25
        else:
            bias_spatial = True
            pool = (1, 35)
            stride = (1, 7)
            kern = 13

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=24,
            kernel_size=(1, kern),
            padding="same",
            bias=True,
        )
        self.spatial_conv = nn.Conv2d(
            in_channels=24,
            out_channels=24,
            kernel_size=(n_channels, 1),
            bias=bias_spatial,
        )
        self.bn = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1)
        self.pool = nn.AvgPool2d(kernel_size=pool, stride=stride)
        self.dropout = nn.Dropout(dropout)

        t_out = (n_samples - pool[1]) // stride[1] + 1
        in_features = 24 * t_out
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

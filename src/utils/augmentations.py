from __future__ import annotations

import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, snr: float = 20.0) -> None:
        super().__init__()
        self.snr = snr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        signal_power = x.pow(2).mean(dim=(1, 2), keepdim=True)
        noise_power = signal_power / (10.0 ** (self.snr / 10.0))
        noise = torch.randn_like(x) * noise_power.sqrt()
        return x + noise


class ChannelDropout(nn.Module):
    def __init__(self, p: float = 0.15) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = torch.bernoulli(
            (1.0 - self.p) * torch.ones(x.size(0), x.size(1), 1, device=x.device)
        )
        return x * mask


class TimeMasking(nn.Module):
    def __init__(self, max_mask_ratio: float = 0.15) -> None:
        super().__init__()
        self.max_mask_ratio = max_mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, C, T = x.shape
        mask_len = max(1, int(T * self.max_mask_ratio * torch.rand(1).item()))
        start = torch.randint(0, max(1, T - mask_len + 1), (1,)).item()
        x[:, :, start : start + mask_len] = 0.0
        return x


class Mixup:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.lam = None
        self.shuffled_idxs = None

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.alpha <= 0:
            return x, y, y
        B = x.size(0)
        self.lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        self.shuffled_idxs = torch.randperm(B, device=x.device)
        mixed_x = self.lam * x + (1.0 - self.lam) * x[self.shuffled_idxs]
        return mixed_x, y, y[self.shuffled_idxs]


def mixup_criterion(
    criterion: nn.Module,
    preds: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(preds, y_a) + (1.0 - lam) * criterion(preds, y_b)

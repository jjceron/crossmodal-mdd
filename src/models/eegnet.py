from __future__ import annotations

import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """EEGNet baseline.

    Modes
    -----
    pp_as="tensor":
        input:
            x: Tensor[B, C, T]

        if aggregate=True:
            output:
                logits: Tensor[B, L]
                logits_time: Tensor[B, T', L]

        if aggregate=False:
            output:
                logits: Tensor[B, T', L]
                logits_time: Tensor[B, T', L]

    pp_as="list":
        input:
            x: list[Tensor[C, T_i]]

        if aggregate=True:
            output:
                logits: Tensor[B, L]
                logits_time: list[Tensor[T'_i, L]]

        if aggregate=False:
            output:
                logits: list[Tensor[T'_i, L]]
                logits_time: list[Tensor[T'_i, L]]
    """

    def __init__(
        self,
        n_channels: int = 24,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        temporal_kern: int = 63,
        separable_kern: int = 15,
        pool1: int = 8,
        pool2: int = 8,
        dropout: float = 0.5,
        meanmax_alpha: float = 0.5,
        pp_as: str = "tensor",
        aggregate: bool = True,
        norm: str = "auto",
    ) -> None:
        super().__init__()

        if not 0.0 <= meanmax_alpha <= 1.0:
            raise ValueError(f"meanmax_alpha must be in [0, 1], got {meanmax_alpha}")

        if pp_as not in {"tensor", "list"}:
            raise ValueError("pp_as must be 'tensor' or 'list'.")

        if norm == "auto":
            norm = "group" if pp_as == "list" else "batch"

        if norm not in {"batch", "group"}:
            raise ValueError("norm must be 'auto', 'batch' or 'group'.")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.temporal_kern = temporal_kern
        self.separable_kern = separable_kern
        self.pool1 = pool1
        self.pool2 = pool2
        self.dropout = float(dropout)
        self.meanmax_alpha = float(meanmax_alpha)
        self.total_pool = pool1 * pool2
        self.pp_as = pp_as
        self.aggregate = bool(aggregate)
        self.norm = norm

        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, temporal_kern),
                padding=(0, temporal_kern // 2),
                bias=False,
            ),
            _make_norm2d(norm, F1),
        )

        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            _make_norm2d(norm, F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout2d(dropout),
        )

        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, separable_kern),
                padding=(0, separable_kern // 2),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            _make_norm2d(norm, F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Conv2d(
            in_channels=F2,
            out_channels=n_classes,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor | list[torch.Tensor],
        alpha: float | None = None,
    ) -> tuple[
        torch.Tensor | list[torch.Tensor],
        torch.Tensor | list[torch.Tensor],
    ]:
        if self.pp_as == "tensor":
            if not isinstance(x, torch.Tensor):
                raise TypeError("Expected x as torch.Tensor when pp_as='tensor'.")

            logits_time = self._forward_tensor(x)
            logits = self.agg_meanmax(logits_time, alpha=alpha) if self.aggregate else logits_time

            return logits, logits_time

        if not isinstance(x, list):
            raise TypeError("Expected x as list[torch.Tensor] when pp_as='list'.")

        logits_time = self._forward_list(x)

        if self.aggregate:
            logits = torch.cat(
                [
                    self.agg_meanmax(logits_i.unsqueeze(0), alpha=alpha)
                    for logits_i in logits_time
                ],
                dim=0,
            )
        else:
            logits = logits_time

        return logits, logits_time

    def _forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for tensor input."""

        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (N, C, T), got {tuple(x.shape)}."
            )

        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected input with {self.n_channels} channels, got {x.shape[1]}."
            )

        x = x.unsqueeze(1)

        z = self.temporal_block(x)
        z = self.spatial_block(z)
        z = self.separable_block(z)

        logits = self.classifier(z)
        logits = logits.squeeze(2)

        return logits.permute(0, 2, 1)

    def _forward_list(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass for list input."""

        for i, x_i in enumerate(x):
            if not isinstance(x_i, torch.Tensor):
                raise TypeError(f"Expected x[{i}] as torch.Tensor.")

            if x_i.ndim != 2:
                raise ValueError(
                    f"Expected x[{i}] with shape (C, T), got {tuple(x_i.shape)}."
                )

            if x_i.shape[0] != self.n_channels:
                raise ValueError(
                    f"Expected x[{i}] with {self.n_channels} channels, got {x_i.shape[0]}."
                )

        lengths = [x_i.shape[1] for x_i in x]

        if len(set(lengths)) == 1:
            logits_batch = self._forward_tensor(torch.stack(x, dim=0))
            return [logits_batch[i] for i in range(logits_batch.shape[0])]

        logits_time = []

        for x_i in x:
            logits_i = self._forward_tensor(x_i.unsqueeze(0))
            logits_time.append(logits_i.squeeze(0))

        return logits_time

    def agg_meanmax(
        self,
        logits_time: torch.Tensor,
        alpha: float | None = None,
    ) -> torch.Tensor:
        """Aggregate temporal logits using mean-max pooling."""

        if logits_time.ndim != 3:
            raise ValueError(
                f"Expected logits_time with shape (N, T', L), got {tuple(logits_time.shape)}."
            )

        if alpha is None:
            alpha = self.meanmax_alpha

        alpha = float(alpha)

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        mean_logits = logits_time.mean(dim=1)
        max_logits = logits_time.max(dim=1).values

        return (1.0 - alpha) * mean_logits + alpha * max_logits

def _make_norm2d(norm: str, num_channels: int, max_groups: int = 4) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)

    if norm == "group":
        groups = min(max_groups, num_channels)

        while num_channels % groups != 0:
            groups -= 1

        return nn.GroupNorm(groups, num_channels)

    raise ValueError(f"Unknown norm: {norm}")


if __name__ == "__main__":
    x = torch.randn(4, 24, 1280)

    print("Tensor mode with aggregation")

    model = EEGNet(
        n_channels=24,
        n_classes=3,
        pp_as="tensor",
        aggregate=True,
        meanmax_alpha=0.5,
    )

    logits, logits_time = model(x)

    print("norm:", model.norm)
    print("logits shape:", logits.shape)
    print("logits_time shape:", logits_time.shape)

    print("\nTensor mode without aggregation")

    model = EEGNet(
        n_channels=24,
        n_classes=3,
        pp_as="tensor",
        aggregate=False,
    )

    logits, logits_time = model(x)

    print("norm:", model.norm)
    print("logits shape:", logits.shape)
    print("logits_time shape:", logits_time.shape)

    print("\nList mode with aggregation")

    model = EEGNet(
        n_channels=24,
        n_classes=3,
        pp_as="list",
        aggregate=True,
        meanmax_alpha=0.5,
    )

    x_list = [
        torch.randn(24, 1280),
        torch.randn(24, 1600),
        torch.randn(24, 2048),
    ]

    logits, logits_time = model(x_list)

    print("norm:", model.norm)
    print("logits shape:", logits.shape)
    print("logits_time shapes:", [z.shape for z in logits_time])

    print("\nList mode without aggregation")

    model = EEGNet(
        n_channels=24,
        n_classes=3,
        pp_as="list",
        aggregate=False,
    )

    logits, logits_time = model(x_list)

    print("norm:", model.norm)
    print("logits shapes:", [z.shape for z in logits])
    print("logits_time shapes:", [z.shape for z in logits_time])
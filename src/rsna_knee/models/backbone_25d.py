"""EfficientNet-B0 multi-plane 2.5D backbone for study-level multi-label prediction."""

from __future__ import annotations

from dataclasses import dataclass

from rsna_knee.labels.targets import N_TARGETS


@dataclass
class BackboneConfig:
    name: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2
    in_channels: int = 3
    num_classes: int = N_TARGETS


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch + torchvision required for EfficientNetB0MultiPlane25D"
        ) from exc
    return torch, nn, models


class EfficientNetB0MultiPlane25D:
    """Factory returning a torch.nn.Module.

    Input: (B, 3, H, W) multi-plane mid-slices (Sag/Cor/Ax).
    Output: (B, 12) logits. Vision-only — no report pathway.
    """

    @staticmethod
    def build(cfg: BackboneConfig | None = None):
        torch, nn, models = _require_torch()
        cfg = cfg or BackboneConfig()

        weights = models.EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        net = models.efficientnet_b0(weights=weights)

        # Adapt first conv if channel count differs (should be 3 for multi-plane).
        if cfg.in_channels != 3:
            old = net.features[0][0]
            new = nn.Conv2d(
                cfg.in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if cfg.pretrained and cfg.in_channels > 0:
                with torch.no_grad():
                    new.weight.zero_()
                    for c in range(cfg.in_channels):
                        new.weight[:, c] = old.weight[:, c % 3] / (
                            cfg.in_channels / 3.0
                        )
            net.features[0][0] = new

        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout, inplace=True),
            nn.Linear(in_features, cfg.num_classes),
        )
        return net


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))

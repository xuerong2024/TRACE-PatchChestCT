"""Swin3D-T patch classifier aligned with PatchChestCT grounding training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.video import Swin3D_T_Weights, swin3d_t


class Swin3DTPatchClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 18,
        output_shape: tuple[int, int, int] = (6, 12, 12),
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        backbone = swin3d_t(weights=weights)

        # PatchChestCT uses single-channel CT volumes and the official baseline
        # changes Swin3D-T to a 16 x 2 x 2 tubelet embedding.
        backbone.patch_embed.proj = nn.Conv3d(
            1,
            96,
            kernel_size=(16, 2, 2),
            stride=(16, 2, 2),
        )

        self.patch_embed = backbone.patch_embed
        self.pos_drop = backbone.pos_drop
        self.features = backbone.features
        self.norm = nn.Identity()
        self.classifier = nn.Linear(768, num_classes)
        self.output_shape = output_shape

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch features with official shape (B, D, H, W, C)."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        if tuple(x.shape[1:4]) != self.output_shape:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = F.interpolate(x, size=self.output_shape, mode="trilinear", align_corners=False)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits.permute(0, 4, 1, 2, 3).contiguous()

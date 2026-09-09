"""R3D-18 patch classifier aligned with PatchChestCT grounding training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torchvision.models.video import R3D_18_Weights, r3d_18
except ModuleNotFoundError:
    class R3D_18_Weights:  # type: ignore[no-redef]
        DEFAULT = "torchvision-required"

    class _BasicStem(nn.Sequential):
        def __init__(self) -> None:
            super().__init__(
                nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
            )

    class _BasicBlock(nn.Module):
        expansion = 1

        def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: nn.Module | None = None,
        ) -> None:
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(planes),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(planes),
            )
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            return self.relu(out)

    class _FallbackR3D18(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inplanes = 64
            self.stem = _BasicStem()
            self.layer1 = self._make_layer(64, 2, stride=1)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=2)
            self.layer4 = self._make_layer(512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(512, 400)

        def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
            downsample = None
            if stride != 1 or self.inplanes != planes * _BasicBlock.expansion:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * _BasicBlock.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * _BasicBlock.expansion),
                )

            layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * _BasicBlock.expansion
            for _ in range(1, blocks):
                layers.append(_BasicBlock(self.inplanes, planes))
            return nn.Sequential(*layers)

    def r3d_18(weights: object | None = None) -> nn.Module:  # type: ignore[no-redef]
        if weights is not None:
            raise ModuleNotFoundError("torchvision is required to initialize R3D-18 pretrained weights")
        return _FallbackR3D18()


class R3D18PatchClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 18,
        output_shape: tuple[int, int, int] = (6, 12, 12),
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        backbone = r3d_18(weights=weights)

        backbone.stem[0] = nn.Conv3d(
            1,
            64,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=(0, 0, 0),
            bias=False,
        )

        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.classifier = nn.Linear(512, num_classes)
        self.output_shape = output_shape

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch features with official shape (B, D, H, W, C)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if tuple(x.shape[-3:]) != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode="trilinear", align_corners=False)
        return x.permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits.permute(0, 4, 1, 2, 3).contiguous()

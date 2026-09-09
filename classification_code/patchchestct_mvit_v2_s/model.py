"""MViT-v2-S patch classifier aligned with PatchChestCT grounding training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.video.mvit import MSBlockConfig, MViT


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    return tuple(int(item) for item in value)


class _DeterministicMaxPool3dFunction(torch.autograd.Function):
    """CUDA max-pool forward with a deterministic CPU scatter backward."""

    @staticmethod
    def forward(
        ctx: object,
        x: torch.Tensor,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
        ceil_mode: bool,
    ) -> torch.Tensor:
        output, indices = F.max_pool3d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
        ctx.save_for_backward(indices)  # type: ignore[attr-defined]
        ctx.input_shape = tuple(x.shape)  # type: ignore[attr-defined]
        ctx.input_device = x.device  # type: ignore[attr-defined]
        return output

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        (indices,) = ctx.saved_tensors  # type: ignore[attr-defined]
        input_shape = ctx.input_shape  # type: ignore[attr-defined]
        input_device = ctx.input_device  # type: ignore[attr-defined]
        batch_size, channels = input_shape[:2]

        # CUDA's overlapping max-pool backward uses atomic additions and has no
        # deterministic implementation in PyTorch 2.5. CPU scatter_add is
        # deterministic and is mathematically the same argmax gradient scatter.
        grad_output_cpu = grad_output.detach().to(device="cpu").contiguous()
        indices_cpu = indices.detach().to(device="cpu").contiguous()
        grad_input_cpu = torch.zeros(input_shape, dtype=grad_output.dtype, device="cpu")
        grad_input_cpu.view(batch_size, channels, -1).scatter_add_(
            2,
            indices_cpu.view(batch_size, channels, -1),
            grad_output_cpu.view(batch_size, channels, -1),
        )
        grad_input = grad_input_cpu.to(device=input_device)
        return grad_input, None, None, None, None, None


class DeterministicMaxPool3d(nn.Module):
    def __init__(self, pool: nn.MaxPool3d) -> None:
        super().__init__()
        if pool.stride is None:
            stride = pool.kernel_size
        else:
            stride = pool.stride
        self.kernel_size = _triple(pool.kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(pool.padding)
        self.dilation = _triple(pool.dilation)
        self.ceil_mode = bool(pool.ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DeterministicMaxPool3dFunction.apply(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )


def _mvit_v2_s_block_setting() -> list[MSBlockConfig]:
    num_heads = [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8]
    input_ch = [64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512]
    output_ch = [64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512]
    kernel_q = [[3, 3, 3]] * 16
    kernel_kv = [[3, 3, 3]] * 16
    stride_q = [
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
    ]
    stride_kv = [
        [1, 8, 8],
        [1, 4, 4],
        [1, 4, 4],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
    ]
    return [
        MSBlockConfig(
            num_heads=num_heads[i],
            input_channels=input_ch[i],
            output_channels=output_ch[i],
            kernel_q=kernel_q[i],
            kernel_kv=kernel_kv[i],
            stride_q=stride_q[i] if i < len(stride_q) else [1, 1, 1],
            stride_kv=stride_kv[i],
        )
        for i in range(16)
    ]


class MViTV2SPatchClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 18,
        input_shape: tuple[int, int, int] = (96, 192, 192),
        output_shape: tuple[int, int, int] = (6, 12, 12),
        pretrained: bool = False,
        deterministic_max_pool: bool = False,
    ) -> None:
        super().__init__()
        if pretrained:
            print("WARNING: PatchChestCT's MViT-v2-S baseline defines a custom MViT and does not load pretrained weights.")

        temporal_size, height, width = input_shape
        self.backbone = MViT(
            spatial_size=(height, width),
            temporal_size=temporal_size,
            block_setting=_mvit_v2_s_block_setting(),
            residual_pool=True,
            residual_with_cls_embed=False,
            rel_pos_embed=True,
            proj_after_attn=True,
            stochastic_depth_prob=0.2,
            num_classes=400,
            patch_embed_kernel=(16, 2, 2),
            patch_embed_stride=(16, 2, 2),
            patch_embed_padding=(0, 0, 0),
        )
        self.backbone.conv_proj = nn.Conv3d(
            1,
            64,
            kernel_size=(16, 2, 2),
            stride=(16, 2, 2),
            padding=(0, 0, 0),
        )
        if deterministic_max_pool:
            for block in self.backbone.blocks:
                pool_skip = getattr(block, "pool_skip", None)
                pool = getattr(pool_skip, "pool", None)
                if isinstance(pool, nn.MaxPool3d):
                    pool_skip.pool = DeterministicMaxPool3d(pool)
        self.classifier = nn.Linear(512, num_classes)
        self.output_shape = output_shape

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch features with official shape (B, D, H, W, C)."""
        x = self.backbone.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.backbone.pos_encoding(x)

        thw = (self.backbone.pos_encoding.temporal_size,) + self.backbone.pos_encoding.spatial_size
        for block in self.backbone.blocks:
            x, thw = block(x, thw)
        x = self.backbone.norm(x)

        batch_size, _, channels = x.shape
        x = x[:, 1:, :].reshape(batch_size, thw[0], thw[1], thw[2], channels)
        if tuple(x.shape[1:4]) != self.output_shape:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = F.interpolate(x, size=self.output_shape, mode="trilinear", align_corners=False)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits.permute(0, 4, 1, 2, 3).contiguous()

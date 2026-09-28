"""Dense official-MIL adapters for PyTorchVideo I3D-R50 and Slow-R50.

Both adapters consume the same one-channel CT tensor as the existing official
case models, ``(B, 1, 96, 192, 192)``, and return dense logits with shape
``(B, num_classes, 6, 12, 12)``.  Consequently, each selected disease is
aggregated over the same 864 MIL instances as R3D-18, Swin3D-T, MViT-v2-S,
and V-JEPA2.1-B.

Geometry
--------
The official loader supplies one-channel CT values in ``[0, 1]``.  Before the
trunk, the adapter applies the PyTorchVideo Kinetics normalization
``(x - 0.45) / 0.225``.  The original Kinetics stem is then deterministically
adapted to a one-channel, non-overlapping ``16 x 7 x 7`` convolution with
temporal stride 16:

* all 96 input slices are used exactly once by the stem's six depth bins;
* RGB weights are summed, which is equivalent to applying the RGB stem to a
  grayscale volume repeated over three channels;
* the pretrained temporal kernel (5 for I3D, 1 for Slow) is linearly resized
  to 16 and scaled by ``old_kernel / 16`` to preserve its response scale.

I3D's later temporal-only max pool is removed because the stem already
produces six depth bins.  The final residual stage's spatial stride is changed
from two to one in both backbones.  The resulting native trunk feature grid is
therefore exactly ``6 x 12 x 12``; no interpolation or adaptive pooling is
used.  All remaining MaxPool3d operations are replaced by the project's
strict deterministic implementation.

Pretrained checkpoints are loaded only from the local Torch Hub checkpoint
cache.  The adapter never downloads weights silently: a missing checkpoint
raises FileNotFoundError with the expected path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from classification_code.deterministic_ops import replace_max_pool3d


OFFICIAL_INPUT_SHAPE = (96, 192, 192)
OFFICIAL_LOGITS_GRID = (6, 12, 12)
OFFICIAL_BAG_SIZE = 6 * 12 * 12
STEM_TEMPORAL_KERNEL = 16
STEM_TEMPORAL_STRIDE = 16
KINETICS_MEAN = 0.45
KINETICS_STD = 0.225


class VideoConvOfficialMILSpec(NamedTuple):
    """Static architecture and checkpoint metadata for one adapter."""

    key: str
    display_name: str
    builder_name: str
    checkpoint_filename: str
    checkpoint_source: str
    pretrained_stem_temporal_kernel: int
    removed_temporal_pool_block: int | None
    feature_channels: int = 2048
    input_shape: tuple[int, int, int] = OFFICIAL_INPUT_SHAPE
    output_grid: tuple[int, int, int] = OFFICIAL_LOGITS_GRID
    bag_size: int = OFFICIAL_BAG_SIZE
    input_channels: int = 1
    pretraining: str = "PyTorchVideo Kinetics-400 8x8 model-zoo checkpoint"

    def to_dict(self) -> dict[str, Any]:
        return dict(self._asdict())


MODEL_SPECS: dict[str, VideoConvOfficialMILSpec] = {
    "i3d_r50": VideoConvOfficialMILSpec(
        key="i3d_r50",
        display_name="I3D-R50",
        builder_name="i3d_r50",
        checkpoint_filename="I3D_8x8_R50.pyth",
        checkpoint_source=(
            "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/"
            "kinetics/I3D_8x8_R50.pyth"
        ),
        pretrained_stem_temporal_kernel=5,
        removed_temporal_pool_block=2,
    ),
    "slow_r50": VideoConvOfficialMILSpec(
        key="slow_r50",
        display_name="Slow-R50",
        builder_name="slow_r50",
        checkpoint_filename="SLOW_8x8_R50.pyth",
        checkpoint_source=(
            "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/"
            "kinetics/SLOW_8x8_R50.pyth"
        ),
        pretrained_stem_temporal_kernel=1,
        removed_temporal_pool_block=None,
    ),
}


def normalize_model_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "i3d": "i3d_r50",
        "i3dr50": "i3d_r50",
        "slow": "slow_r50",
        "slowr50": "slow_r50",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODEL_SPECS:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown official video-conv model {name!r}; choose one of: {valid}")
    return normalized


def get_model_spec(name: str) -> VideoConvOfficialMILSpec:
    """Return immutable metadata for ``name``."""

    return MODEL_SPECS[normalize_model_name(name)]


def _load_offline_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required pretrained checkpoint is not cached: {path}. "
            "The official-MIL adapter intentionally does not download weights."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError(f"Unexpected PyTorchVideo checkpoint format in {path}")
    return checkpoint


def _resize_stem_temporal_kernel(
    weight: torch.Tensor,
    target_kernel: int,
) -> torch.Tensor:
    """Resize only the temporal axis and retain the pretrained response scale."""

    if weight.ndim != 5:
        raise ValueError(f"Expected a 5D Conv3d weight tensor, got {tuple(weight.shape)}")
    out_channels, in_channels, old_kernel, height, width = weight.shape
    temporal_vectors = (
        weight.permute(0, 1, 3, 4, 2)
        .contiguous()
        .reshape(out_channels * in_channels * height * width, 1, old_kernel)
    )
    resized = F.interpolate(
        temporal_vectors,
        size=target_kernel,
        mode="linear",
        align_corners=False,
    )
    resized = resized * (float(old_kernel) / float(target_kernel))
    return (
        resized.reshape(out_channels, in_channels, height, width, target_kernel)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
    )


def _adapt_stem_to_official_ct(
    backbone: nn.Module,
    spec: VideoConvOfficialMILSpec,
) -> dict[str, Any]:
    stem = backbone.blocks[0]
    old_conv = stem.conv
    if not isinstance(old_conv, nn.Conv3d):
        raise TypeError(f"Expected PyTorchVideo Conv3d stem, got {type(old_conv).__name__}")
    if old_conv.in_channels != 3:
        raise RuntimeError(f"Expected a three-channel pretrained stem, got {old_conv.in_channels}")
    if tuple(old_conv.kernel_size[1:]) != (7, 7):
        raise RuntimeError(f"Unexpected stem spatial kernel: {old_conv.kernel_size}")
    if int(old_conv.kernel_size[0]) != spec.pretrained_stem_temporal_kernel:
        raise RuntimeError(
            f"{spec.display_name} stem kernel {old_conv.kernel_size[0]} does not match "
            f"the documented pretrained kernel {spec.pretrained_stem_temporal_kernel}"
        )

    new_conv = nn.Conv3d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=(STEM_TEMPORAL_KERNEL, *old_conv.kernel_size[1:]),
        stride=(STEM_TEMPORAL_STRIDE, *old_conv.stride[1:]),
        padding=(0, *old_conv.padding[1:]),
        dilation=(1, *old_conv.dilation[1:]),
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )
    with torch.no_grad():
        resized_weight = _resize_stem_temporal_kernel(
            old_conv.weight.detach(),
            STEM_TEMPORAL_KERNEL,
        )
        # Summing RGB kernels is exactly equivalent to repeating a grayscale
        # input over three channels before the convolution.
        new_conv.weight.copy_(resized_weight.sum(dim=1, keepdim=True))
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias.detach())
    stem.conv = new_conv
    return {
        "old_kernel": tuple(int(v) for v in old_conv.kernel_size),
        "old_stride": tuple(int(v) for v in old_conv.stride),
        "new_kernel": tuple(int(v) for v in new_conv.kernel_size),
        "new_stride": tuple(int(v) for v in new_conv.stride),
        "channel_adaptation": "temporally_resized_rgb_weights_summed_to_one_channel",
        "temporal_weight_scale": (
            float(spec.pretrained_stem_temporal_kernel) / float(STEM_TEMPORAL_KERNEL)
        ),
    }


def _remove_i3d_temporal_pool(
    backbone: nn.Module,
    block_index: int | None,
) -> dict[str, Any] | None:
    if block_index is None:
        return None
    pool = backbone.blocks[block_index]
    if not isinstance(pool, nn.MaxPool3d):
        raise TypeError(
            f"Expected temporal MaxPool3d at block {block_index}, got {type(pool).__name__}"
        )
    if tuple(int(v) for v in pool.kernel_size) != (2, 1, 1):
        raise RuntimeError(f"Unexpected temporal pool kernel at block {block_index}: {pool.kernel_size}")
    metadata = {
        "block_index": int(block_index),
        "old_kernel": tuple(int(v) for v in pool.kernel_size),
        "old_stride": tuple(int(v) for v in pool.stride),
        "replacement": "Identity",
    }
    backbone.blocks[block_index] = nn.Identity()
    return metadata


def _retain_final_spatial_resolution(backbone: nn.Module) -> dict[str, Any]:
    """Change only the final residual stage's spatial stride from two to one."""

    final_stage = backbone.blocks[-2]
    first_block = final_stage.res_blocks[0]
    branch1 = first_block.branch1_conv
    branch2 = first_block.branch2.conv_b
    for label, conv in (("branch1_conv", branch1), ("branch2.conv_b", branch2)):
        if not isinstance(conv, nn.Conv3d):
            raise TypeError(f"Expected Conv3d at final-stage {label}, got {type(conv).__name__}")
        if tuple(int(v) for v in conv.stride) != (1, 2, 2):
            raise RuntimeError(f"Unexpected final-stage {label} stride: {conv.stride}")
        conv.stride = (1, 1, 1)
    return {
        "stage": "final_residual_stage",
        "branches": ("branch1_conv", "branch2.conv_b"),
        "old_stride": (1, 2, 2),
        "new_stride": (1, 1, 1),
    }


class PyTorchVideoConvOfficialMIL(nn.Module):
    """Shared dense-logit adapter used by I3D-R50 and Slow-R50."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 18,
        output_shape: tuple[int, int, int] = OFFICIAL_LOGITS_GRID,
        pretrained: bool = True,
        dropout: float | None = None,
        deterministic_max_pool: bool = True,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.spec = get_model_spec(model_name)
        self.num_classes = int(num_classes)
        self.output_shape = tuple(int(v) for v in output_shape)
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if self.output_shape != OFFICIAL_LOGITS_GRID:
            raise ValueError(
                "Official MIL comparison requires output_shape=(6, 12, 12), "
                f"got {self.output_shape}"
            )
        if not deterministic_max_pool:
            raise ValueError(
                "Official video-conv adapters require deterministic_max_pool=True "
                "for strict reproducibility"
            )

        try:
            from pytorchvideo.models import hub as pytorchvideo_hub
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "I3D-R50/Slow-R50 official adapters require pytorchvideo in the pcct environment"
            ) from error

        builder = getattr(pytorchvideo_hub, self.spec.builder_name)
        backbone = builder(pretrained=False)

        if pretrained:
            checkpoint_root = (
                Path(checkpoint_dir)
                if checkpoint_dir is not None
                else Path(torch.hub.get_dir()) / "checkpoints"
            )
            checkpoint_path = checkpoint_root / self.spec.checkpoint_filename
            checkpoint = _load_offline_checkpoint(checkpoint_path)
            incompatible = backbone.load_state_dict(checkpoint["model_state"], strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Strict checkpoint load failed: missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        else:
            checkpoint_path = None

        original_head = backbone.blocks[-1]
        original_dropout = getattr(original_head, "dropout", None)
        original_dropout_p = (
            float(original_dropout.p) if isinstance(original_dropout, nn.Dropout) else 0.0
        )
        effective_dropout = original_dropout_p if dropout is None else float(dropout)
        if not 0.0 <= effective_dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {effective_dropout}")

        stem_adaptation = _adapt_stem_to_official_ct(backbone, self.spec)
        temporal_pool_adaptation = _remove_i3d_temporal_pool(
            backbone,
            self.spec.removed_temporal_pool_block,
        )
        spatial_stride_adaptation = _retain_final_spatial_resolution(backbone)

        # Remove global average pool, Kinetics dropout/projection, and final
        # adaptive pool.  The full residual trunk remains active.
        backbone.blocks[-1] = nn.Identity()
        deterministic_pool_count = replace_max_pool3d(backbone)

        self.backbone = backbone
        self.dropout = nn.Dropout(p=effective_dropout)
        self.classifier = nn.Conv3d(
            self.spec.feature_channels,
            self.num_classes,
            kernel_size=1,
            bias=True,
        )
        self._deterministic_replacements = {
            "max_pool3d": int(deterministic_pool_count),
        }
        self._official_mil_metadata: dict[str, Any] = {
            **self.spec.to_dict(),
            "num_classes": self.num_classes,
            "pretrained": bool(pretrained),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "stem_adaptation": stem_adaptation,
            "temporal_pool_adaptation": temporal_pool_adaptation,
            "spatial_stride_adaptation": spatial_stride_adaptation,
            "removed_global_head": type(original_head).__name__,
            "dense_classifier": "Conv3d(2048, num_classes, kernel_size=1)",
            "dropout": effective_dropout,
            "deterministic_replacements": dict(self._deterministic_replacements),
            "uses_all_96_input_slices": True,
            "native_dense_feature_grid": OFFICIAL_LOGITS_GRID,
            "interpolation_or_adaptive_pool_after_trunk": False,
            "input_value_range": "[0, 1] from official loader",
            "internal_input_normalization": {
                "formula": "(x - mean) / std",
                "mean": KINETICS_MEAN,
                "std": KINETICS_STD,
                "semantics": (
                    "equivalent to grayscale replication followed by the "
                    "three identical PyTorchVideo Kinetics channel transforms"
                ),
            },
        }

    def metadata(self) -> dict[str, Any]:
        """Return a copy of runtime architecture/provenance metadata."""

        return dict(self._official_mil_metadata)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input (B, 1, D, H, W), got {tuple(x.shape)}")
        if int(x.shape[1]) != 1:
            raise ValueError(f"Expected one CT channel, got {x.shape[1]}")
        if tuple(int(v) for v in x.shape[-3:]) != OFFICIAL_INPUT_SHAPE:
            raise ValueError(
                f"Official input must have shape {OFFICIAL_INPUT_SHAPE}, "
                f"got {tuple(int(v) for v in x.shape[-3:])}"
            )
        normalized = (x - KINETICS_MEAN) / KINETICS_STD
        features = self.backbone(normalized)
        expected = (self.spec.feature_channels, *self.output_shape)
        if tuple(int(v) for v in features.shape[1:]) != expected:
            raise RuntimeError(
                f"{self.spec.display_name} produced feature shape {tuple(features.shape)}, "
                f"expected (B, {expected})"
            )
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(self.dropout(features))
        expected = (x.shape[0], self.num_classes, *self.output_shape)
        if tuple(logits.shape) != expected:
            raise RuntimeError(f"Dense logits have shape {tuple(logits.shape)}, expected {expected}")
        return logits


class I3DR50OfficialPatchClassifier(PyTorchVideoConvOfficialMIL):
    """I3D-R50 Kinetics trunk adapted to the official 864-instance CT grid."""

    def __init__(
        self,
        num_classes: int = 18,
        output_shape: tuple[int, int, int] = OFFICIAL_LOGITS_GRID,
        pretrained: bool = True,
        dropout: float | None = None,
        deterministic_max_pool: bool = True,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            "i3d_r50",
            num_classes=num_classes,
            output_shape=output_shape,
            pretrained=pretrained,
            dropout=dropout,
            deterministic_max_pool=deterministic_max_pool,
            checkpoint_dir=checkpoint_dir,
        )


class SlowR50OfficialPatchClassifier(PyTorchVideoConvOfficialMIL):
    """Slow-R50 Kinetics trunk adapted to the official 864-instance CT grid."""

    def __init__(
        self,
        num_classes: int = 18,
        output_shape: tuple[int, int, int] = OFFICIAL_LOGITS_GRID,
        pretrained: bool = True,
        dropout: float | None = None,
        deterministic_max_pool: bool = True,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            "slow_r50",
            num_classes=num_classes,
            output_shape=output_shape,
            pretrained=pretrained,
            dropout=dropout,
            deterministic_max_pool=deterministic_max_pool,
            checkpoint_dir=checkpoint_dir,
        )


def build_model(
    name: str,
    *,
    num_classes: int = 18,
    output_shape: tuple[int, int, int] = OFFICIAL_LOGITS_GRID,
    pretrained: bool = True,
    dropout: float | None = None,
    deterministic_max_pool: bool = True,
    checkpoint_dir: str | Path | None = None,
) -> PyTorchVideoConvOfficialMIL:
    """Build an I3D-R50 or Slow-R50 dense official-MIL adapter."""

    key = normalize_model_name(name)
    model_class = (
        I3DR50OfficialPatchClassifier
        if key == "i3d_r50"
        else SlowR50OfficialPatchClassifier
    )
    return model_class(
        num_classes=num_classes,
        output_shape=output_shape,
        pretrained=pretrained,
        dropout=dropout,
        deterministic_max_pool=deterministic_max_pool,
        checkpoint_dir=checkpoint_dir,
    )


__all__ = [
    "I3DR50OfficialPatchClassifier",
    "KINETICS_MEAN",
    "KINETICS_STD",
    "MODEL_SPECS",
    "OFFICIAL_BAG_SIZE",
    "OFFICIAL_INPUT_SHAPE",
    "OFFICIAL_LOGITS_GRID",
    "PyTorchVideoConvOfficialMIL",
    "SlowR50OfficialPatchClassifier",
    "VideoConvOfficialMILSpec",
    "build_model",
    "get_model_spec",
    "normalize_model_name",
]

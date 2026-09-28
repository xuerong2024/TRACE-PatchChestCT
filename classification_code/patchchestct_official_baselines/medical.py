"""Official PatchChestCT dense-MIL adapters for medical 3D pretraining methods.

The original MedicalNet and Models Genesis baselines in this repository end in
global pooling and therefore emit one logit vector per patient.  The official
PatchChestCT case protocol instead needs one logit vector per 3D instance on a
``6 x 12 x 12`` grid.  This module keeps the pretrained encoders intact and
replaces only their global classification tails:

``(B, 1, 96, 192, 192) -> (B, C, 12, 24, 24)
                        -> deterministic non-overlapping 2x2x2 mean
                        -> Conv3d(C, num_classes, 1)
                        -> (B, num_classes, 6, 12, 12)``

The training code is expected to apply the same selected-class indexing and
NoisyOR aggregation as the other official case models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import torch
from torch import nn

from classification_code.patchchestct_medical_models.models import (
    MedicalModelSpec,
    MedicalNetResNetClassifier,
    ModelGenesisClassifier,
    build_model as build_native_case_model,
    get_model_spec,
)


SUPPORTED_MODELS: Final[tuple[str, ...]] = (
    "medicalnet_resnet18_23",
    "medicalnet_resnet50_23",
    "modelgenesis_chest_ct",
)
OFFICIAL_INPUT_SHAPE: Final[tuple[int, int, int]] = (96, 192, 192)
TRUNK_FEATURE_SHAPE: Final[tuple[int, int, int]] = (12, 24, 24)
OFFICIAL_OUTPUT_SHAPE: Final[tuple[int, int, int]] = (6, 12, 12)


@dataclass(frozen=True)
class OfficialMedicalMILMetadata:
    """Serializable protocol and architecture metadata saved with each run."""

    model_name: str
    family: str
    source: str
    input_channels: int
    input_shape: tuple[int, int, int]
    feature_channels: int
    trunk_feature_shape: tuple[int, int, int]
    output_classes: int
    output_shape: tuple[int, int, int]
    bag_size: int
    dense_head: str
    spatial_reduction: str
    aggregation: str
    internal_input_normalization: str
    pretrained_model_id: str | None
    genesis_batch_norm_eval_semantics: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _MedicalNetTrunk(nn.Module):
    """MedicalNet encoder through its dilated fourth residual stage."""

    def __init__(self, native_model: MedicalNetResNetClassifier) -> None:
        super().__init__()
        self.conv1 = native_model.conv1
        self.bn1 = native_model.bn1
        self.relu = native_model.relu
        # ``build_native_case_model(..., deterministic=True)`` replaces this
        # with DeterministicMaxPool3d while retaining the pretrained trunk.
        self.maxpool = native_model.maxpool
        self.layer1 = native_model.layer1
        self.layer2 = native_model.layer2
        self.layer3 = native_model.layer3
        self.layer4 = native_model.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class _ModelGenesisTrunk(nn.Module):
    """Models Genesis encoder through its deepest down transition."""

    def __init__(self, native_model: ModelGenesisClassifier) -> None:
        super().__init__()
        self.down_tr64 = native_model.down_tr64
        self.down_tr128 = native_model.down_tr128
        self.down_tr256 = native_model.down_tr256
        self.down_tr512 = native_model.down_tr512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out64, _ = self.down_tr64(x)
        out128, _ = self.down_tr128(out64)
        out256, _ = self.down_tr256(out128)
        out512, _ = self.down_tr512(out256)
        return out512


class _NonOverlappingMeanPool3d2(nn.Module):
    """Exact 2x2x2 block mean with a strict-deterministic CUDA backward.

    Unlike CUDA ``avg_pool3d_backward``, reshape/broadcast reductions are
    accepted by ``torch.use_deterministic_algorithms(True)``.  Every input
    belongs to exactly one output block, so the backward also needs no atomic
    accumulation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected B,C,D,H,W features, got {tuple(x.shape)}")
        batch, channels, depth, height, width = x.shape
        if any(size % 2 for size in (depth, height, width)):
            raise ValueError(
                "Non-overlapping 2x mean reduction requires even D,H,W, got "
                f"{(depth, height, width)}"
            )
        return x.reshape(
            batch,
            channels,
            depth // 2,
            2,
            height // 2,
            2,
            width // 2,
            2,
        ).mean(dim=(3, 5, 7))


class OfficialMedicalMILDenseClassifier(nn.Module):
    """Dense classifier compatible with the official PatchChestCT MIL trainer."""

    def __init__(
        self,
        *,
        trunk: nn.Module,
        metadata: OfficialMedicalMILMetadata,
        pretrained_load_report: dict[str, int] | None,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        # CUDA avg_pool3d_backward is rejected by strict deterministic mode.
        # An exact non-overlapping reshape+mean has no atomic accumulation and
        # avoids the generic pooling fallback's CPU transfer.
        self.reduction = _NonOverlappingMeanPool3d2()
        self.classifier = nn.Conv3d(
            metadata.feature_channels,
            metadata.output_classes,
            kernel_size=1,
            bias=True,
        )
        self.metadata = metadata
        self.metadata_dict = metadata.to_dict()
        self.pretrained_load_report = (
            None
            if pretrained_load_report is None
            else dict(pretrained_load_report)
        )

        # Keep the attribute names used by the existing native builders so
        # trainers can record these details without architecture-specific code.
        self._pretrained_load_report = self.pretrained_load_report
        self._official_mil_metadata = self.metadata_dict
        self._deterministic_replacements = {
            "native_trunk_max_pool3d": (
                3 if metadata.family == "modelgenesis" else 1
            ),
            "dense_grid_nonoverlapping_reshape_mean": 1,
        }

        if freeze_backbone:
            for parameter in self.trunk.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _check_input(x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                "Official medical MIL adapters require a 5D tensor "
                f"(B, 1, D, H, W), got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 1 or tuple(x.shape[-3:]) != OFFICIAL_INPUT_SHAPE:
            raise ValueError(
                "Official medical MIL input must have shape "
                f"(B, 1, {OFFICIAL_INPUT_SHAPE[0]}, "
                f"{OFFICIAL_INPUT_SHAPE[1]}, {OFFICIAL_INPUT_SHAPE[2]}), "
                f"got {tuple(x.shape)}"
            )

    def extract_trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return channel-first encoder features before the 2x reduction."""

        self._check_input(x)
        if self.metadata.family.startswith("medicalnet_resnet"):
            # MedicalNet's published/native PatchChestCT preprocessing is a
            # per-volume population z-score.  The outer official loader still
            # owns HU clipping, orientation, and the exact 96x192x192 crop.
            mean = x.mean(dim=(2, 3, 4), keepdim=True)
            variance = (x - mean).square().mean(dim=(2, 3, 4), keepdim=True)
            x = (x - mean) / variance.sqrt().clamp_min(1e-6)
        features = self.trunk(x)
        if tuple(features.shape[-3:]) != TRUNK_FEATURE_SHAPE:
            raise RuntimeError(
                f"{self.metadata.model_name} trunk produced spatial shape "
                f"{tuple(features.shape[-3:])}; expected {TRUNK_FEATURE_SHAPE}"
            )
        return features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return official grid features as ``(B, D, H, W, C)``."""

        features = self.reduction(self.extract_trunk_features(x))
        if tuple(features.shape[-3:]) != OFFICIAL_OUTPUT_SHAPE:
            raise RuntimeError(
                f"{self.metadata.model_name} reduction produced spatial shape "
                f"{tuple(features.shape[-3:])}; expected {OFFICIAL_OUTPUT_SHAPE}"
            )
        return features.permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.reduction(self.extract_trunk_features(x))
        logits = self.classifier(features)
        expected = (
            x.shape[0],
            self.metadata.output_classes,
            *OFFICIAL_OUTPUT_SHAPE,
        )
        if tuple(logits.shape) != expected:
            raise RuntimeError(
                f"{self.metadata.model_name} emitted {tuple(logits.shape)}; "
                f"expected {expected}"
            )
        return logits


def build_official_medical_mil_model(
    name: str | MedicalModelSpec,
    num_classes: int = 18,
    *,
    pretrained: bool = True,
    pretrained_path: str | None = None,
    freeze_backbone: bool = False,
) -> OfficialMedicalMILDenseClassifier:
    """Build a pretrained dense medical model for official case-level MIL.

    Args:
        name: One of :data:`SUPPORTED_MODELS`, or its ``MedicalModelSpec``.
        num_classes: Number of dense output channels.  The existing official
            trainer uses 18 and selects its nine target indices afterwards.
        pretrained: Load the original MedicalNet/Models Genesis encoder.
        pretrained_path: Optional local checkpoint override (or URL for
            Models Genesis, matching the native builder).
        freeze_backbone: Freeze only the pretrained encoder; the new 1x1x1
            dense classifier remains trainable.
    """

    spec = get_model_spec(name) if isinstance(name, str) else name
    if spec.name not in SUPPORTED_MODELS:
        valid = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"{spec.name!r} is not an official dense medical adapter. "
            f"Valid choices: {valid}"
        )
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    # Loading through the established native builder preserves its checkpoint
    # key filtering and load accounting.  Its global pool and native linear
    # head are deliberately discarded below.
    native_model = build_native_case_model(
        spec,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        freeze_backbone=False,
        deterministic=True,
    )
    load_report = getattr(native_model, "_pretrained_load_report", None)
    expected_load_reports = {
        "medicalnet_resnet18_23": {"loaded": 102, "missing": 2, "unexpected": 0},
        "medicalnet_resnet50_23": {"loaded": 318, "missing": 2, "unexpected": 0},
        "modelgenesis_chest_ct": {"loaded": 56, "missing": 4, "unexpected": 0},
    }
    if pretrained:
        expected_report = expected_load_reports[spec.name]
        if load_report != expected_report:
            raise RuntimeError(
                f"{spec.name} pretrained trunk coverage changed: "
                f"expected {expected_report}, got {load_report}"
            )

    if isinstance(native_model, MedicalNetResNetClassifier):
        trunk: nn.Module = _MedicalNetTrunk(native_model)
        feature_channels = (
            2048 if spec.family == "medicalnet_resnet50" else 512
        )
        genesis_bn_semantics = None
    elif isinstance(native_model, ModelGenesisClassifier):
        trunk = _ModelGenesisTrunk(native_model)
        feature_channels = 512
        # ContBatchNorm3d in the shared implementation passes
        # ``self.training`` to F.batch_norm.  Consequently model.eval() uses
        # checkpoint running statistics and never statistics from another
        # patient in the evaluation batch.
        genesis_bn_semantics = (
            "eval() uses stored running_mean/running_var "
            "(F.batch_norm training=self.training)"
        )
    else:  # pragma: no cover - guarded by SUPPORTED_MODELS/family above.
        raise TypeError(
            f"Unexpected native model type for {spec.name}: "
            f"{type(native_model).__name__}"
        )

    metadata = OfficialMedicalMILMetadata(
        model_name=spec.name,
        family=spec.family,
        source=spec.source,
        input_channels=1,
        input_shape=OFFICIAL_INPUT_SHAPE,
        feature_channels=feature_channels,
        trunk_feature_shape=TRUNK_FEATURE_SHAPE,
        output_classes=int(num_classes),
        output_shape=OFFICIAL_OUTPUT_SHAPE,
        bag_size=(
            OFFICIAL_OUTPUT_SHAPE[0]
            * OFFICIAL_OUTPUT_SHAPE[1]
            * OFFICIAL_OUTPUT_SHAPE[2]
        ),
        dense_head=f"Conv3d({feature_channels}, {num_classes}, kernel_size=1)",
        spatial_reduction=(
            "non-overlapping reshape+mean over exact 2x2x2 blocks"
        ),
        aggregation=(
            "external official NoisyOR over D,H,W after selecting target classes"
        ),
        internal_input_normalization=(
            "per-patient population z-score after official HU clipping/cropping"
            if spec.family.startswith("medicalnet_resnet")
            else "identity; official loader [0,1] values are used directly"
        ),
        pretrained_model_id=(
            pretrained_path
            if pretrained_path is not None
            else spec.default_model_id
        ),
        genesis_batch_norm_eval_semantics=genesis_bn_semantics,
    )
    official_model = OfficialMedicalMILDenseClassifier(
        trunk=trunk,
        metadata=metadata,
        pretrained_load_report=load_report,
        freeze_backbone=freeze_backbone,
    )
    expected_report = expected_load_reports[spec.name]
    official_model._pretrained_trunk_coverage = (
        {
            "pretrained": True,
            "trunk_loaded": expected_report["loaded"],
            "trunk_expected": expected_report["loaded"],
            "trunk_missing": 0,
            "discarded_native_head_tensors": expected_report["missing"],
            "unexpected": expected_report["unexpected"],
        }
        if pretrained
        else {
            "pretrained": False,
            "trunk_loaded": 0,
            "trunk_expected": expected_report["loaded"],
            "trunk_missing": expected_report["loaded"],
            "discarded_native_head_tensors": 0,
            "unexpected": 0,
        }
    )
    return official_model


# Short alias for trainers that conventionally import ``build_model``.
build_model = build_official_medical_mil_model

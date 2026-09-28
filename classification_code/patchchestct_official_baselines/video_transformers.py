"""Official-grid dense MIL adapters for VideoMAE and TimeSformer.

The public models in this module accept the canonical PatchChestCT tensor
``(B, 1, 96, 192, 192)`` in the official loader's ``[0, 1]`` intensity
range and return dense logits with shape ``(B, C, 6, 12, 12)``.

Both adapters use all 96 consecutive input slices and preserve the native
pretrained channel normalization:

* VideoMAE expands its pretrained 2-frame tubelet projection to a
  non-overlapping 16-slice projection.  Each of the two temporal kernel
  planes is repeated over eight adjacent slices and divided by eight.
* TimeSformer averages each non-overlapping group of 16 adjacent slices to
  one frame before its unchanged pretrained 2-D patch projection.

No trainable feature map is interpolated or adaptively pooled.  Position
embedding adaptation happens once on CPU during construction, so its
backward path cannot introduce a nondeterministic CUDA pooling kernel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from classification_code.patchchestct_video_models.models import (
    _load_hf_classification_model,
)


OFFICIAL_INPUT_SHAPE = (96, 192, 192)
OFFICIAL_OUTPUT_SHAPE = (6, 12, 12)
OFFICIAL_BAG_SIZE = math.prod(OFFICIAL_OUTPUT_SHAPE)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class OfficialVideoTransformerMetadata:
    """Serializable provenance for one official dense adapter."""

    name: str
    family: str
    pretrained_model_id: str
    external_input_shape: tuple[int, int, int]
    external_input_range: str
    output_grid: tuple[int, int, int]
    bag_size: int
    slice_strategy: str
    spatial_strategy: str
    native_hidden_token_order: str
    exported_token_order: str
    class_token_policy: str
    position_embedding_policy: str
    deterministic_feature_resampling: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OFFICIAL_VIDEO_TRANSFORMER_METADATA: dict[str, OfficialVideoTransformerMetadata] = {
    "videomae": OfficialVideoTransformerMetadata(
        name="videomae",
        family="hf_videomae",
        pretrained_model_id="MCG-NJU/videomae-base-finetuned-kinetics",
        external_input_shape=OFFICIAL_INPUT_SHAPE,
        external_input_range="[0, 1], passed directly to ImageNet normalization",
        output_grid=OFFICIAL_OUTPUT_SHAPE,
        bag_size=OFFICIAL_BAG_SIZE,
        slice_strategy=(
            "all 96 slices; pretrained temporal kernel 2 is expanded to kernel/stride 16 "
            "by repeat-interleaving each kernel plane 8 times and dividing by 8"
        ),
        spatial_strategy="unchanged pretrained 16x16 patch kernel/stride; 192/16=12",
        native_hidden_token_order="time-major (t, h, w), w fastest; no CLS token",
        exported_token_order="(B, C, t, h, w)",
        class_token_policy="VideoMAE has no CLS token; all 864 hidden tokens are retained",
        position_embedding_policy=(
            "config-generated fixed 8x14x14 sin-cos table interpolated once on CPU "
            "to 6x12x12"
        ),
        deterministic_feature_resampling=(
            "none; the adapted non-overlapping Conv3d directly produces 6x12x12"
        ),
    ),
    "timesformer": OfficialVideoTransformerMetadata(
        name="timesformer",
        family="hf_timesformer",
        pretrained_model_id="facebook/timesformer-base-finetuned-k400",
        external_input_shape=OFFICIAL_INPUT_SHAPE,
        external_input_range="[0, 1], passed directly to ImageNet normalization",
        output_grid=OFFICIAL_OUTPUT_SHAPE,
        bag_size=OFFICIAL_BAG_SIZE,
        slice_strategy="all 96 slices; six non-overlapping 16-slice groups are averaged",
        spatial_strategy="unchanged pretrained 16x16 2-D patch kernel/stride; 192/16=12",
        native_hidden_token_order="spatial-major (h, w, t), t fastest, after one CLS token",
        exported_token_order="CLS removed, then rearranged to (B, C, t, h, w)",
        class_token_policy="the single global CLS token is explicitly excluded from dense logits",
        position_embedding_policy=(
            "pretrained 14x14 spatial and 8-frame temporal tables use the Hugging Face "
            "nearest-neighbor policy once on CPU to become 12x12 and 6 frames"
        ),
        deterministic_feature_resampling=(
            "non-overlapping reshape+mean before the backbone; no feature interpolation/pooling"
        ),
    ),
}


def get_official_video_transformer_metadata(name: str) -> dict[str, Any]:
    """Return a copy of the formal adapter metadata."""

    normalized = _normalize_name(name)
    return OFFICIAL_VIDEO_TRANSFORMER_METADATA[normalized].to_dict()


def _normalize_name(name: str) -> str:
    normalized = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "videomae": "videomae",
        "timesformer": "timesformer",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(OFFICIAL_VIDEO_TRANSFORMER_METADATA))
        raise ValueError(f"Unknown official video transformer {name!r}; choose one of: {valid}")
    return aliases[normalized]


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    result = tuple(int(item) for item in value)
    if len(result) != 2:
        raise ValueError(f"Expected a scalar or pair, got {value!r}")
    return result


def _shape3(value: Sequence[int], label: str) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError(f"{label} must contain three positive integers, got {value!r}")
    return result


def _main_snapshot(cache_dir: Path) -> Path | None:
    ref = cache_dir / "refs" / "main"
    if ref.is_file():
        snapshot = cache_dir / "snapshots" / ref.read_text().strip()
        if snapshot.is_dir():
            return snapshot
    return None


def _cached_hf_model_dir(model_id: str) -> str:
    """Resolve a fully local HF checkpoint without attempting network access."""

    supplied = Path(model_id).expanduser()
    if supplied.is_dir():
        return str(supplied.resolve())

    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
    )
    candidates: list[Path] = []
    main_snapshot = _main_snapshot(cache_dir)
    if main_snapshot is not None:
        candidates.append(main_snapshot)
    snapshots = cache_dir / "snapshots"
    if snapshots.is_dir():
        candidates.extend(path for path in sorted(snapshots.iterdir()) if path.is_dir())

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        has_config = (candidate / "config.json").is_file()
        has_weights = any(
            (candidate / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        if has_config and has_weights:
            return str(candidate)
    raise FileNotFoundError(
        f"No complete offline Hugging Face snapshot found for {model_id!r} under {cache_dir}"
    )


def _offline_model_dir(name: str, model_id: str) -> str:
    if name == "timesformer":
        if model_id != "facebook/timesformer-base-finetuned-k400":
            return _cached_hf_model_dir(model_id)
        cache = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--facebook--timesformer-base-finetuned-k400"
        )
        config_candidates = sorted(cache.glob("snapshots/*/config.json"))
        weight_candidates = sorted(cache.glob("snapshots/*/model.safetensors"))
        if not config_candidates or not weight_candidates:
            raise FileNotFoundError(
                f"No complete cached TimeSformer config/safetensors pair under {cache}"
            )
        sources = {
            "config.json": config_candidates[0].resolve(),
            "model.safetensors": weight_candidates[0].resolve(),
        }
        local_dir = (
            Path("/tmp")
            / "patchchestct_hf_local"
            / "official-timesformer-base-finetuned-k400-safetensors"
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in sources.items():
            destination = local_dir / filename
            if destination.is_symlink():
                if destination.resolve() == source:
                    continue
                destination.unlink()
            elif destination.exists():
                raise RuntimeError(
                    f"Refusing to replace non-symlink TimeSformer staging file: {destination}"
                )
            destination.symlink_to(source)
        return str(local_dir)
    return _cached_hf_model_dir(model_id)


def _restore_videomae_checkpoint_biases(
    backbone: nn.Module,
    local_model_dir: str,
) -> dict[str, Any]:
    """Map legacy HF ``q_bias``/``v_bias`` keys to the current modules.

    The cached MCG-NJU checkpoint stores separate query/value biases and no
    key bias.  Transformers 5 names the parameters ``query.bias``,
    ``value.bias`` and ``key.bias`` and currently reports the legacy tensors
    as unexpected.  Explicitly applying this lossless mapping prevents those
    pretrained parameters from being silently replaced by random values.
    """

    weights_path = Path(local_model_dir) / "model.safetensors"
    if not weights_path.is_file():
        return {"applied": False, "reason": "no model.safetensors legacy table"}
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The cached VideoMAE checkpoint is safetensors; install safetensors to restore "
            "its pretrained attention biases"
        ) from error

    repaired_layers = 0
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
        layers = list(backbone.encoder.layer)
        legacy_present = any(
            f"videomae.encoder.layer.{index}.attention.attention.q_bias" in checkpoint_keys
            for index in range(len(layers))
        )
        if not legacy_present:
            return {"applied": False, "reason": "checkpoint uses current attention-bias names"}

        for index, layer in enumerate(layers):
            prefix = f"videomae.encoder.layer.{index}.attention.attention"
            query_key = f"{prefix}.q_bias"
            value_key = f"{prefix}.v_bias"
            if query_key not in checkpoint_keys or value_key not in checkpoint_keys:
                raise RuntimeError(
                    f"Legacy VideoMAE checkpoint is missing {query_key!r} or {value_key!r}"
                )
            attention = layer.attention.attention
            if (
                attention.query.bias is None
                or attention.key.bias is None
                or attention.value.bias is None
            ):
                raise RuntimeError("Current VideoMAE attention modules do not expose split biases")
            with torch.no_grad():
                attention.query.bias.copy_(
                    checkpoint.get_tensor(query_key).to(
                        device=attention.query.bias.device,
                        dtype=attention.query.bias.dtype,
                    )
                )
                attention.value.bias.copy_(
                    checkpoint.get_tensor(value_key).to(
                        device=attention.value.bias.device,
                        dtype=attention.value.bias.dtype,
                    )
                )
                # The released architecture deliberately has no key bias.
                attention.key.bias.zero_()
            repaired_layers += 1
    return {
        "applied": True,
        "layers": repaired_layers,
        "mapping": "q_bias->query.bias, v_bias->value.bias, absent key bias->zeros",
    }


def _load_pretrained_classification_model_with_info(
    class_name: str,
    local_model_dir: str,
    attn_implementation: str | None,
) -> tuple[nn.Module, dict[str, Any]]:
    try:
        import transformers
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "VideoMAE/TimeSformer official adapters require transformers"
        ) from error
    model_class = getattr(transformers, class_name)
    kwargs: dict[str, Any] = {
        "num_labels": 400,
        "ignore_mismatched_sizes": True,
        "problem_type": "multi_label_classification",
        "use_safetensors": True,
        "local_files_only": True,
        "output_loading_info": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    loaded = model_class.from_pretrained(local_model_dir, **kwargs)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError(
            f"{class_name}.from_pretrained did not return loading information"
        )
    model, raw_info = loaded
    info = {
        key: sorted(str(item) for item in value)
        for key, value in raw_info.items()
    }
    return model, info


def _load_pretrained_backbone(
    name: str,
    model_id: str,
    attn_implementation: str | None,
) -> nn.Module:
    local_model_dir = _offline_model_dir(name, model_id)
    if name == "videomae":
        classification_model, loading_info = _load_pretrained_classification_model_with_info(
            "VideoMAEForVideoClassification",
            local_model_dir,
            attn_implementation,
        )
        expected_missing = {
            f"videomae.encoder.layer.{index}.attention.attention.{projection}.bias"
            for index in range(12)
            for projection in ("query", "key", "value")
        }
        expected_unexpected = {
            f"videomae.encoder.layer.{index}.attention.attention.{bias_name}"
            for index in range(12)
            for bias_name in ("q_bias", "v_bias")
        }
        if (
            set(loading_info.get("missing_keys", ())) != expected_missing
            or set(loading_info.get("unexpected_keys", ())) != expected_unexpected
            or loading_info.get("mismatched_keys")
            or loading_info.get("error_msgs")
        ):
            raise RuntimeError(
                "VideoMAE pretrained checkpoint coverage changed unexpectedly: "
                f"{loading_info}"
            )
        backbone = classification_model.videomae
        # HF's released VideoMAE classifier applies fc_norm after token mean
        # pooling.  For dense MIL, preserve that pretrained LayerNorm and apply
        # it independently to every retained patch token before classification.
        object.__setattr__(
            backbone,
            "_official_dense_norm",
            classification_model.fc_norm,
        )
        object.__setattr__(backbone, "_official_pretrained_dir", local_model_dir)
        object.__setattr__(
            backbone,
            "_official_pretrained_loading_info",
            {
                "raw_missing": len(expected_missing),
                "raw_unexpected": len(expected_unexpected),
                "mismatched": 0,
                "errors": 0,
                "coverage_after_bias_repair": "all pretrained backbone tensors accounted for",
            },
        )
        setattr(
            backbone,
            "_official_checkpoint_repairs",
            _restore_videomae_checkpoint_biases(backbone, local_model_dir),
        )
        return backbone

    timesformer_attn = "eager" if attn_implementation == "sdpa" else attn_implementation
    classification_model, loading_info = _load_pretrained_classification_model_with_info(
        "TimesformerForVideoClassification",
        local_model_dir,
        timesformer_attn,
    )
    if any(loading_info.get(key) for key in loading_info):
        raise RuntimeError(
            "TimeSformer pretrained checkpoint did not load completely: "
            f"{loading_info}"
        )
    backbone = classification_model.timesformer
    object.__setattr__(backbone, "_official_pretrained_dir", local_model_dir)
    object.__setattr__(
        backbone,
        "_official_pretrained_loading_info",
        {
            "raw_missing": 0,
            "raw_unexpected": 0,
            "mismatched": 0,
            "errors": 0,
            "coverage": "complete",
        },
    )
    return backbone


def _load_random_backbone(name: str, attn_implementation: str | None) -> nn.Module:
    if name == "videomae":
        classification_model = _load_hf_classification_model(
            "VideoMAEForVideoClassification",
            OFFICIAL_VIDEO_TRANSFORMER_METADATA[name].pretrained_model_id,
            num_classes=400,
            pretrained=False,
            attn_implementation=attn_implementation,
        )
        backbone = classification_model.videomae
        object.__setattr__(backbone, "_official_dense_norm", classification_model.fc_norm)
        return backbone

    timesformer_attn = "eager" if attn_implementation == "sdpa" else attn_implementation
    classification_model = _load_hf_classification_model(
        "TimesformerForVideoClassification",
        OFFICIAL_VIDEO_TRANSFORMER_METADATA[name].pretrained_model_id,
        num_classes=400,
        pretrained=False,
        attn_implementation=timesformer_attn,
    )
    return classification_model.timesformer


def _resize_videomae_positions(
    position_embeddings: torch.Tensor,
    source_grid: tuple[int, int, int],
    target_grid: tuple[int, int, int],
) -> torch.Tensor:
    expected = math.prod(source_grid)
    if tuple(position_embeddings.shape[:2]) != (1, expected):
        raise RuntimeError(
            f"VideoMAE position table has shape {tuple(position_embeddings.shape)}; "
            f"expected one table with {expected} tokens for source grid {source_grid}"
        )
    dtype = position_embeddings.dtype
    channels = position_embeddings.shape[-1]
    positions = (
        position_embeddings.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(1, *source_grid, channels)
        .permute(0, 4, 1, 2, 3)
    )
    positions = F.interpolate(
        positions,
        size=target_grid,
        mode="trilinear",
        align_corners=False,
    )
    return (
        positions.permute(0, 2, 3, 4, 1)
        .reshape(1, math.prod(target_grid), channels)
        .to(dtype=dtype)
        .contiguous()
    )


def _resize_timesformer_spatial_positions(
    position_embeddings: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    expected = 1 + math.prod(source_grid)
    if tuple(position_embeddings.shape[:2]) != (1, expected):
        raise RuntimeError(
            f"TimeSformer position table has shape {tuple(position_embeddings.shape)}; "
            f"expected one CLS plus {math.prod(source_grid)} spatial tokens"
        )
    dtype = position_embeddings.dtype
    channels = position_embeddings.shape[-1]
    source = position_embeddings.detach().to(device="cpu", dtype=torch.float32)
    cls_position = source[:, :1]
    spatial = (
        source[:, 1:]
        .reshape(1, *source_grid, channels)
        .permute(0, 3, 1, 2)
    )
    # This is the same interpolation mode used by HF TimesformerEmbeddings.
    spatial = F.interpolate(spatial, size=target_grid, mode="nearest")
    spatial = spatial.permute(0, 2, 3, 1).reshape(1, math.prod(target_grid), channels)
    return torch.cat((cls_position, spatial), dim=1).to(dtype=dtype).contiguous()


def _resize_timesformer_time_positions(
    time_embeddings: torch.Tensor,
    target_frames: int,
) -> torch.Tensor:
    if time_embeddings.ndim != 3 or time_embeddings.shape[0] != 1:
        raise RuntimeError(f"Unexpected TimeSformer time table shape {tuple(time_embeddings.shape)}")
    dtype = time_embeddings.dtype
    positions = (
        time_embeddings.detach()
        .to(device="cpu", dtype=torch.float32)
        .transpose(1, 2)
    )
    # This is the same interpolation mode used by HF TimesformerEmbeddings.
    positions = F.interpolate(positions, size=target_frames, mode="nearest")
    return positions.transpose(1, 2).to(dtype=dtype).contiguous()


class _OfficialInputMixin:
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    input_mean: torch.Tensor
    input_std: torch.Tensor

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected input (B,1,D,H,W), got {tuple(x.shape)}")
        if tuple(x.shape[2:]) != self.input_shape:
            raise ValueError(
                f"Expected canonical spatial shape {self.input_shape}, got {tuple(x.shape[2:])}"
            )
        if not x.is_floating_point():
            raise TypeError(f"Expected floating-point CT input, got {x.dtype}")

    def _imagenet_normalized_rgb(self, x: torch.Tensor) -> torch.Tensor:
        # The official loader already maps clipped HU to [0, 1], which is the
        # value range expected before the pretrained video normalization.
        rgb = x.expand(-1, 3, -1, -1, -1)
        mean = self.input_mean.to(device=x.device, dtype=x.dtype)
        std = self.input_std.to(device=x.device, dtype=x.dtype)
        return (rgb - mean) / std


class VideoMAEOfficialMIL(_OfficialInputMixin, nn.Module):
    """VideoMAE dense classifier whose tokens exactly match the official bag."""

    def __init__(
        self,
        num_classes: int = 18,
        *,
        pretrained: bool = True,
        model_id: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        input_shape: Sequence[int] = OFFICIAL_INPUT_SHAPE,
        output_shape: Sequence[int] = OFFICIAL_OUTPUT_SHAPE,
        freeze_backbone: bool = False,
        attn_implementation: str | None = "sdpa",
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_shape = _shape3(input_shape, "input_shape")
        self.output_shape = _shape3(output_shape, "output_shape")
        if self.input_shape[0] % self.output_shape[0] != 0:
            raise ValueError("Input depth must divide evenly into output depth")
        target_tubelet = self.input_shape[0] // self.output_shape[0]

        self.backbone = (
            backbone
            if backbone is not None
            else (
                _load_pretrained_backbone("videomae", model_id, attn_implementation)
                if pretrained
                else _load_random_backbone("videomae", attn_implementation)
            )
        )
        embeddings = self.backbone.embeddings
        patch_embeddings = embeddings.patch_embeddings
        projection = patch_embeddings.projection
        if not isinstance(projection, nn.Conv3d):
            raise TypeError(f"Expected VideoMAE Conv3d patch projection, got {type(projection).__name__}")

        source_tubelet = int(patch_embeddings.tubelet_size)
        source_patch = _pair(patch_embeddings.patch_size)
        if source_patch[0] != source_patch[1]:
            raise ValueError(f"VideoMAE adapter requires square spatial patches, got {source_patch}")
        if target_tubelet % source_tubelet != 0:
            raise ValueError(
                f"Target tubelet {target_tubelet} must be a multiple of pretrained tubelet {source_tubelet}"
            )
        expected_spatial = (
            self.input_shape[1] // source_patch[0],
            self.input_shape[2] // source_patch[1],
        )
        if expected_spatial != self.output_shape[1:]:
            raise ValueError(
                f"Input/patch geometry produces {expected_spatial}, not requested {self.output_shape[1:]}"
            )
        if any(
            input_size % patch_size != 0
            for input_size, patch_size in zip(self.input_shape[1:], source_patch)
        ):
            raise ValueError("Official spatial input must divide evenly by the pretrained patch size")

        source_image = _pair(patch_embeddings.image_size)
        source_frames = int(self.backbone.config.num_frames)
        source_grid = (
            source_frames // source_tubelet,
            source_image[0] // source_patch[0],
            source_image[1] // source_patch[1],
        )
        resized_positions = _resize_videomae_positions(
            embeddings.position_embeddings,
            source_grid,
            self.output_shape,
        )

        expanded_projection = nn.Conv3d(
            in_channels=projection.in_channels,
            out_channels=projection.out_channels,
            kernel_size=(target_tubelet, *source_patch),
            stride=(target_tubelet, *source_patch),
            padding=0,
            dilation=1,
            groups=projection.groups,
            bias=projection.bias is not None,
            padding_mode=projection.padding_mode,
        ).to(device=projection.weight.device, dtype=projection.weight.dtype)
        temporal_repeat = target_tubelet // source_tubelet
        with torch.no_grad():
            expanded_weight = projection.weight.repeat_interleave(temporal_repeat, dim=2)
            expanded_weight = expanded_weight / float(temporal_repeat)
            expanded_projection.weight.copy_(expanded_weight)
            if projection.bias is not None and expanded_projection.bias is not None:
                expanded_projection.bias.copy_(projection.bias)

        patch_embeddings.projection = expanded_projection
        patch_embeddings.image_size = self.input_shape[1:]
        patch_embeddings.tubelet_size = target_tubelet
        patch_embeddings.num_patches = math.prod(self.output_shape)
        embeddings.num_patches = math.prod(self.output_shape)
        embeddings.position_embeddings = nn.Parameter(
            resized_positions.to(device=expanded_projection.weight.device),
            requires_grad=False,
        )
        self.backbone.config.image_size = self.input_shape[1]
        self.backbone.config.num_frames = self.input_shape[0]
        self.backbone.config.tubelet_size = target_tubelet

        hidden_size = int(self.backbone.config.hidden_size)
        dense_norm = getattr(self.backbone, "_official_dense_norm", None)
        if dense_norm is None:
            if backbone is None:
                raise RuntimeError(
                    "VideoMAE classification checkpoint did not expose its pretrained fc_norm"
                )
            dense_norm = nn.Identity()
        self.dense_norm = dense_norm
        self.classifier = nn.Linear(hidden_size, int(num_classes))
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.register_buffer(
            "input_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.adapter_metadata = {
            **get_official_video_transformer_metadata("videomae"),
            "num_classes": int(num_classes),
            "source_token_grid": list(source_grid),
            "adapted_temporal_kernel": target_tubelet,
            "temporal_kernel_repeat": temporal_repeat,
            "pretrained": bool(pretrained),
            "offline_model_id": model_id,
            "checkpoint_repairs": getattr(
                self.backbone,
                "_official_checkpoint_repairs",
                {"applied": False, "reason": "random or injected backbone"},
            ),
            "pretrained_loading_info": getattr(
                self.backbone,
                "_official_pretrained_loading_info",
                None,
            ),
            "pretrained_dense_norm": type(self.dense_norm).__name__,
            "dense_norm_policy": (
                "preserved HF classification fc_norm applied independently to every token"
            ),
            "resolved_pretrained_dir": getattr(
                self.backbone,
                "_official_pretrained_dir",
                None,
            ),
            "pretrained_loading_info": getattr(
                self.backbone,
                "_official_pretrained_loading_info",
                None,
            ),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 6, 12, 12, hidden)`` in explicit ``t,h,w`` order."""

        self._validate_input(x)
        pixel_values = self._imagenet_normalized_rgb(x).permute(0, 2, 1, 3, 4).contiguous()
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        expected_tokens = math.prod(self.output_shape)
        if hidden.ndim != 3 or hidden.shape[1] != expected_tokens:
            raise RuntimeError(
                f"VideoMAE returned hidden shape {tuple(hidden.shape)}; expected "
                f"(B,{expected_tokens},hidden) with no CLS token"
            )
        hidden = self.dense_norm(hidden)
        # Conv3d.flatten(2) orders tokens as t-major, then h, then w.
        return hidden.reshape(hidden.shape[0], *self.output_shape, hidden.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits.permute(0, 4, 1, 2, 3).contiguous()

    def export_metadata(self) -> dict[str, Any]:
        return dict(self.adapter_metadata)


class TimeSformerOfficialMIL(_OfficialInputMixin, nn.Module):
    """TimeSformer dense classifier using all slices via 16-slice groups."""

    def __init__(
        self,
        num_classes: int = 18,
        *,
        pretrained: bool = True,
        model_id: str = "facebook/timesformer-base-finetuned-k400",
        input_shape: Sequence[int] = OFFICIAL_INPUT_SHAPE,
        output_shape: Sequence[int] = OFFICIAL_OUTPUT_SHAPE,
        freeze_backbone: bool = False,
        attn_implementation: str | None = "eager",
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_shape = _shape3(input_shape, "input_shape")
        self.output_shape = _shape3(output_shape, "output_shape")
        if self.input_shape[0] % self.output_shape[0] != 0:
            raise ValueError("Input depth must divide evenly into output depth")
        self.slice_group_size = self.input_shape[0] // self.output_shape[0]

        self.backbone = (
            backbone
            if backbone is not None
            else (
                _load_pretrained_backbone("timesformer", model_id, attn_implementation)
                if pretrained
                else _load_random_backbone("timesformer", attn_implementation)
            )
        )
        embeddings = self.backbone.embeddings
        patch_embeddings = embeddings.patch_embeddings
        projection = patch_embeddings.projection
        if not isinstance(projection, nn.Conv2d):
            raise TypeError(f"Expected TimeSformer Conv2d patch projection, got {type(projection).__name__}")

        source_patch = _pair(patch_embeddings.patch_size)
        source_image = _pair(patch_embeddings.image_size)
        source_spatial_grid = (
            source_image[0] // source_patch[0],
            source_image[1] // source_patch[1],
        )
        target_spatial_grid = (
            self.input_shape[1] // source_patch[0],
            self.input_shape[2] // source_patch[1],
        )
        if target_spatial_grid != self.output_shape[1:]:
            raise ValueError(
                f"Input/patch geometry produces {target_spatial_grid}, not requested {self.output_shape[1:]}"
            )
        if any(
            input_size % patch_size != 0
            for input_size, patch_size in zip(self.input_shape[1:], source_patch)
        ):
            raise ValueError("Official spatial input must divide evenly by the pretrained patch size")

        resized_spatial_positions = _resize_timesformer_spatial_positions(
            embeddings.position_embeddings,
            source_spatial_grid,
            target_spatial_grid,
        )
        source_frames = int(embeddings.time_embeddings.shape[1])
        resized_time_positions = _resize_timesformer_time_positions(
            embeddings.time_embeddings,
            self.output_shape[0],
        )
        parameter_device = projection.weight.device
        embeddings.position_embeddings = nn.Parameter(
            resized_spatial_positions.to(device=parameter_device),
            requires_grad=embeddings.position_embeddings.requires_grad,
        )
        embeddings.time_embeddings = nn.Parameter(
            resized_time_positions.to(device=parameter_device),
            requires_grad=embeddings.time_embeddings.requires_grad,
        )
        patch_embeddings.image_size = self.input_shape[1:]
        patch_embeddings.num_patches = math.prod(target_spatial_grid)
        embeddings.num_patches = math.prod(target_spatial_grid)
        # TimeSformerLayer reads these config values to reshape every block.
        self.backbone.config.image_size = self.input_shape[1]
        self.backbone.config.num_frames = self.output_shape[0]

        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, int(num_classes))
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.register_buffer(
            "input_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.adapter_metadata = {
            **get_official_video_transformer_metadata("timesformer"),
            "num_classes": int(num_classes),
            "source_spatial_grid": list(source_spatial_grid),
            "source_frames": source_frames,
            "slice_group_size": self.slice_group_size,
            "pretrained": bool(pretrained),
            "offline_model_id": model_id,
            "resolved_pretrained_dir": getattr(
                self.backbone,
                "_official_pretrained_dir",
                None,
            ),
        }

    def _group_slices(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, height, width = x.shape
        grouped = x.reshape(
            batch,
            channels,
            self.output_shape[0],
            self.slice_group_size,
            height,
            width,
        )
        return grouped.mean(dim=3)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 6, 12, 12, hidden)`` after explicit CLS removal."""

        self._validate_input(x)
        grouped = self._group_slices(x)
        pixel_values = (
            self._imagenet_normalized_rgb(grouped)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        expected_tokens = math.prod(self.output_shape)
        if hidden.ndim != 3 or hidden.shape[1] != expected_tokens + 1:
            raise RuntimeError(
                f"TimeSformer returned hidden shape {tuple(hidden.shape)}; expected one CLS plus "
                f"{expected_tokens} dense tokens"
            )

        # HF TimeSformer stores divided-space-time tokens with t as the fastest
        # axis inside each flattened spatial patch: (h, w, t).  Remove the
        # global CLS and explicitly move time to the leading grid axis.
        patch_tokens = hidden[:, 1:]
        batch_size, _, channels = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(
            batch_size,
            self.output_shape[1],
            self.output_shape[2],
            self.output_shape[0],
            channels,
        )
        return patch_tokens.permute(0, 3, 1, 2, 4).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits.permute(0, 4, 1, 2, 3).contiguous()

    def export_metadata(self) -> dict[str, Any]:
        return dict(self.adapter_metadata)


# Descriptive aliases for callers that use the existing "PatchClassifier"
# naming convention.
VideoMAEOfficialPatchClassifier = VideoMAEOfficialMIL
TimeSformerOfficialPatchClassifier = TimeSformerOfficialMIL


def build_model(
    name: str,
    num_classes: int = 18,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    deterministic: bool = True,
    model_id: str | None = None,
    attn_implementation: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Build one formal official-grid transformer adapter.

    ``deterministic`` is accepted for compatibility with the shared runner.
    The adapter geometry is strict-safe in either mode; the flag is recorded
    in the returned model metadata while the runner controls global PyTorch
    deterministic settings.
    """

    normalized = _normalize_name(name)
    metadata = OFFICIAL_VIDEO_TRANSFORMER_METADATA[normalized]
    selected_model_id = metadata.pretrained_model_id if model_id is None else model_id
    if attn_implementation is None:
        attn_implementation = "eager" if deterministic else (
            "sdpa" if normalized == "videomae" else "eager"
        )
    elif deterministic and attn_implementation != "eager":
        raise ValueError(
            "Strict deterministic official runs require attn_implementation='eager'"
        )
    cls = VideoMAEOfficialMIL if normalized == "videomae" else TimeSformerOfficialMIL
    model = cls(
        num_classes=num_classes,
        pretrained=pretrained,
        model_id=selected_model_id,
        freeze_backbone=freeze_backbone,
        attn_implementation=attn_implementation,
        **kwargs,
    )
    model.adapter_metadata["strict_deterministic_requested"] = bool(deterministic)
    model.adapter_metadata["attention_implementation"] = attn_implementation
    return model


build_official_video_transformer = build_model


__all__ = [
    "OFFICIAL_BAG_SIZE",
    "OFFICIAL_INPUT_SHAPE",
    "OFFICIAL_OUTPUT_SHAPE",
    "OFFICIAL_VIDEO_TRANSFORMER_METADATA",
    "OfficialVideoTransformerMetadata",
    "TimeSformerOfficialMIL",
    "TimeSformerOfficialPatchClassifier",
    "VideoMAEOfficialMIL",
    "VideoMAEOfficialPatchClassifier",
    "build_model",
    "build_official_video_transformer",
    "get_official_video_transformer_metadata",
]

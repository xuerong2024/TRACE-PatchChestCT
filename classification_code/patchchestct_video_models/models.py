"""Model registry for PatchChestCT video fine-tuning."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from classification_code.deterministic_ops import (
    replace_adaptive_global_avg_pool3d,
    replace_avg_pool3d,
    replace_max_pool3d,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    default_model_id: str | None
    frames: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    source: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)
TORCHVISION_R3D_MEAN = (0.43216, 0.394666, 0.37645)
TORCHVISION_R3D_STD = (0.22803, 0.22145, 0.216989)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


MODEL_SPECS: dict[str, ModelSpec] = {
    "videomae": ModelSpec(
        name="videomae",
        family="hf_videomae",
        default_model_id="MCG-NJU/videomae-base-finetuned-kinetics",
        frames=16,
        image_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        source="Hugging Face Transformers / MCG-NJU VideoMAE",
        notes="Base VideoMAE fine-tuned on Kinetics; 16-frame default matches the released config.",
    ),
    "timesformer": ModelSpec(
        name="timesformer",
        family="hf_timesformer",
        default_model_id="facebook/timesformer-base-finetuned-k400",
        frames=8,
        image_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        source="Hugging Face Transformers / Facebook TimeSformer",
        notes="Base TimeSformer fine-tuned on Kinetics-400; default clip length is 8 frames.",
    ),
    "mvit_v2_s": ModelSpec(
        name="mvit_v2_s",
        family="torchvision_mvit_v2_s",
        default_model_id=None,
        frames=16,
        image_size=224,
        mean=KINETICS_MEAN,
        std=KINETICS_STD,
        source="torchvision.models.video.mvit_v2_s",
        notes="Torchvision MViT-v2-S Kinetics-400 weights.",
    ),
    "i3d_r50": ModelSpec(
        name="i3d_r50",
        family="pytorchvideo_i3d_r50",
        default_model_id="facebookresearch/pytorchvideo:i3d_r50",
        frames=8,
        image_size=224,
        mean=KINETICS_MEAN,
        std=KINETICS_STD,
        source="PyTorchVideo TorchHub",
        notes="I3D R50 Kinetics-400 model-zoo checkpoint.",
    ),
    "slow_r50": ModelSpec(
        name="slow_r50",
        family="pytorchvideo_slow_r50",
        default_model_id="facebookresearch/pytorchvideo:slow_r50",
        frames=8,
        image_size=224,
        mean=KINETICS_MEAN,
        std=KINETICS_STD,
        source="PyTorchVideo TorchHub",
        notes="Slow-only 3D ResNet R50 Kinetics-400 model-zoo checkpoint.",
    ),
    "vjepa2": ModelSpec(
        name="vjepa2",
        family="hf_vjepa2",
        default_model_id="facebook/vjepa2-vitl-fpc64-256",
        frames=64,
        image_size=256,
        mean=KINETICS_MEAN,
        std=KINETICS_STD,
        source="Hugging Face Transformers / Meta V-JEPA 2",
        notes="V-JEPA 2 ViT-L 64-frame 256px encoder; practical default for fine-tuning.",
    ),
    "vjepa2_g": ModelSpec(
        name="vjepa2_g",
        family="hf_vjepa2",
        default_model_id="facebook/vjepa2-vitg-fpc64-384",
        frames=64,
        image_size=384,
        mean=KINETICS_MEAN,
        std=KINETICS_STD,
        source="Hugging Face Transformers / Meta V-JEPA 2",
        notes="V-JEPA 2 ViT-G 64-frame 384px encoder; highest-memory official option.",
    ),
    "vjepa2_1_b": ModelSpec(
        name="vjepa2_1_b",
        family="torchhub_vjepa2_1",
        default_model_id="vjepa2_1_vit_base_384",
        frames=64,
        image_size=384,
        mean=KINETICS_MEAN,
        std=KINETICS_STD,
        source="facebookresearch/vjepa2 PyTorch Hub",
        notes="V-JEPA 2.1 ViT-B/16 384px encoder; smallest official 2.1 checkpoint.",
    ),
    "r3d18": ModelSpec(
        name="r3d18",
        family="torchvision_r3d18",
        default_model_id=None,
        frames=16,
        image_size=112,
        mean=TORCHVISION_R3D_MEAN,
        std=TORCHVISION_R3D_STD,
        source="torchvision.models.video.r3d_18",
        notes="Utility baseline for smoke tests and comparison, not part of the requested run list.",
    ),
}


def get_model_spec(
    name: str,
    *,
    hf_model_id: str | None = None,
    frames: int | None = None,
    image_size: int | None = None,
) -> ModelSpec:
    if name not in MODEL_SPECS:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model {name!r}. Valid choices: {valid}")
    spec = MODEL_SPECS[name]
    if hf_model_id is not None:
        spec = replace(spec, default_model_id=hf_model_id)
    if frames is not None:
        spec = replace(spec, frames=int(frames))
    if image_size is not None:
        spec = replace(spec, image_size=int(image_size))
    return spec


def _missing_optional_package(package: str, model_name: str, install_hint: str) -> RuntimeError:
    return RuntimeError(
        f"{model_name} requires optional package {package!r}. Install it in the pcct env first: {install_hint}"
    )


def _replace_last_classifier(module: nn.Module, num_classes: int) -> bool:
    candidate: tuple[nn.Module, str, nn.Module] | None = None
    for parent in module.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear) and child.out_features in {400, 600, 174, 1000}:
                candidate = (parent, child_name, child)
            elif isinstance(child, nn.Conv3d) and child.out_channels in {400, 600, 174, 1000}:
                candidate = (parent, child_name, child)
    if candidate is None:
        return False

    parent, child_name, child = candidate
    if isinstance(child, nn.Linear):
        replacement: nn.Module = nn.Linear(child.in_features, num_classes, bias=child.bias is not None)
    else:
        replacement = nn.Conv3d(
            child.in_channels,
            num_classes,
            kernel_size=child.kernel_size,
            stride=child.stride,
            padding=child.padding,
            dilation=child.dilation,
            groups=child.groups,
            bias=child.bias is not None,
        )
    setattr(parent, child_name, replacement)
    return True


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _unfreeze_classifier_heads(module: nn.Module, num_classes: int) -> None:
    for head_name in ("head", "patch_head"):
        head = getattr(module, head_name, None)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad = True
    for child in module.modules():
        if isinstance(child, nn.Linear) and child.out_features == num_classes:
            for parameter in child.parameters():
                parameter.requires_grad = True
        elif isinstance(child, nn.Conv3d) and child.out_channels == num_classes:
            for parameter in child.parameters():
                parameter.requires_grad = True


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def _local_timesformer_safetensors_dir(model_id: str) -> str | None:
    if model_id != "facebook/timesformer-base-finetuned-k400":
        return None

    cache_dir = Path.home() / ".cache/huggingface/hub/models--facebook--timesformer-base-finetuned-k400"
    if not cache_dir.exists():
        return None
    config_candidates = sorted(cache_dir.glob("snapshots/*/config.json"))
    weight_candidates = sorted(cache_dir.glob("snapshots/*/model.safetensors"))
    if not config_candidates or not weight_candidates:
        return None

    local_dir = Path("/tmp/patchchestct_hf_local/timesformer-base-finetuned-k400-safetensors")
    local_dir.mkdir(parents=True, exist_ok=True)
    _safe_symlink(config_candidates[0].resolve(), local_dir / "config.json")
    _safe_symlink(weight_candidates[0].resolve(), local_dir / "model.safetensors")
    return str(local_dir)


def _load_hf_classification_model(
    class_name: str,
    model_id: str,
    num_classes: int,
    pretrained: bool,
    attn_implementation: str | None,
    revision: str | None = None,
    use_safetensors: bool | None = None,
) -> nn.Module:
    try:
        import transformers
    except ModuleNotFoundError as e:
        raise _missing_optional_package(
            "transformers",
            class_name,
            "python -m pip install -U git+https://github.com/huggingface/transformers",
        ) from e

    cls = getattr(transformers, class_name)
    common_kwargs = {
        "num_labels": num_classes,
        "ignore_mismatched_sizes": True,
        "problem_type": "multi_label_classification",
    }
    if pretrained:
        load_kwargs = dict(common_kwargs)
        if revision is not None:
            load_kwargs["revision"] = revision
        if use_safetensors is not None:
            load_kwargs["use_safetensors"] = use_safetensors
        if attn_implementation:
            try:
                return cls.from_pretrained(model_id, attn_implementation=attn_implementation, **load_kwargs)
            except TypeError:
                pass
        return cls.from_pretrained(model_id, **load_kwargs)

    if class_name == "VideoMAEForVideoClassification":
        config = transformers.VideoMAEConfig()
    elif class_name == "TimesformerForVideoClassification":
        config = transformers.TimesformerConfig()
    else:
        config = transformers.AutoConfig.from_pretrained(model_id)
    config.num_labels = num_classes
    config.problem_type = "multi_label_classification"
    return cls(config)


class HFVideoClassificationWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.backbone(pixel_values=pixel_values).logits


class VJEPA2Classifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
        attn_implementation: str | None,
    ) -> None:
        super().__init__()
        try:
            import transformers
        except ModuleNotFoundError as e:
            raise _missing_optional_package(
                "transformers",
                "vjepa2",
                "python -m pip install -U git+https://github.com/huggingface/transformers",
            ) from e

        if pretrained:
            kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
            try:
                self.backbone = transformers.AutoModel.from_pretrained(model_id, **kwargs)
            except TypeError:
                self.backbone = transformers.AutoModel.from_pretrained(model_id)
        else:
            config = transformers.VJEPA2Config()
            self.backbone = transformers.VJEPA2Model(config)

        hidden_size = int(getattr(self.backbone.config, "hidden_size", 1024))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x: torch.Tensor, return_patch_logits: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        pixel_values = x.permute(0, 2, 1, 3, 4).contiguous()
        try:
            outputs = self.backbone(pixel_values_videos=pixel_values, skip_predictor=True)
        except TypeError:
            outputs = self.backbone(pixel_values_videos=pixel_values)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        if return_patch_logits:
            raise RuntimeError("Patch logits are only implemented for the V-JEPA 2.1 PyTorch Hub wrapper")
        return self.head(pooled)


VJEPA21_CHECKPOINTS = {
    "vjepa2_1_vit_base_384": {
        "url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        "checkpoint_key": "ema_encoder",
    },
    "vjepa2_1_vit_large_384": {
        "url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        "checkpoint_key": "ema_encoder",
    },
    "vjepa2_1_vit_giant_384": {
        "url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
        "checkpoint_key": "target_encoder",
    },
    "vjepa2_1_vit_gigantic_384": {
        "url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
        "checkpoint_key": "target_encoder",
    },
}


def _clean_vjepa_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        key = key.removeprefix("backbone.")
        cleaned[key] = value
    return cleaned


def _torchhub_repo_or_local(repo_slug: str) -> tuple[str, str]:
    owner, repo = repo_slug.split("/", maxsplit=1)
    local_repo = Path(torch.hub.get_dir()) / f"{owner}_{repo}_main"
    if local_repo.exists():
        return str(local_repo), "local"
    return repo_slug, "github"


class VJEPA21TorchHubClassifier(nn.Module):
    def __init__(
        self,
        hub_model_name: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
        patch_supervision: bool = False,
        mil_head: str = "basic",
        class_token_heads: int = 8,
        class_token_dropout: float = 0.1,
        class_token_coord_embedding: bool = False,
        class_token_coord_mode: str = "add",
        class_token_anatomical_prior: bool = False,
        class_token_prior_gamma: float = 1.0,
        class_token_prior_grid: Sequence[int] = (32, 24, 24),
        patch_output_shape: Sequence[int] | None = None,
        prototype_count: int = 4,
        prototype_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if hub_model_name not in VJEPA21_CHECKPOINTS:
            valid = ", ".join(sorted(VJEPA21_CHECKPOINTS))
            raise ValueError(f"Unknown V-JEPA 2.1 hub model {hub_model_name!r}. Valid choices: {valid}")

        # Build from the official repo, but load the checkpoint here because the
        # upstream hubconf may point at a local test URL for pretrained weights.
        repo_or_dir, hub_source = _torchhub_repo_or_local("facebookresearch/vjepa2")
        encoder, _predictor = torch.hub.load(
            repo_or_dir,
            hub_model_name,
            source=hub_source,
            pretrained=False,
            trust_repo=True,
            skip_validation=True,
        )
        del _predictor
        self.backbone = encoder

        if pretrained:
            checkpoint_info = VJEPA21_CHECKPOINTS[hub_model_name]
            checkpoint = torch.hub.load_state_dict_from_url(
                str(checkpoint_info["url"]),
                map_location="cpu",
                progress=True,
            )
            state_dict = _clean_vjepa_state_dict(checkpoint[str(checkpoint_info["checkpoint_key"])])
            self.backbone.load_state_dict(state_dict, strict=True)

        hidden_size = int(getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features", 768)))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        self.mil_head = mil_head
        self.patch_output_shape = (
            tuple(int(value) for value in patch_output_shape)
            if patch_output_shape is not None
            else None
        )
        if self.patch_output_shape is not None and (
            len(self.patch_output_shape) != 3 or any(value <= 0 for value in self.patch_output_shape)
        ):
            raise ValueError(f"patch_output_shape must contain three positive integers, got {patch_output_shape}")
        self.patch_head = (
            build_patch_mil_head(
                mil_head=mil_head,
                hidden_size=hidden_size,
                num_classes=num_classes,
                class_token_heads=class_token_heads,
                class_token_dropout=class_token_dropout,
                class_token_coord_embedding=class_token_coord_embedding,
                class_token_coord_mode=class_token_coord_mode,
                class_token_anatomical_prior=class_token_anatomical_prior,
                class_token_prior_gamma=class_token_prior_gamma,
                class_token_prior_grid=class_token_prior_grid,
                prototype_count=prototype_count,
                prototype_temperature=prototype_temperature,
            )
            if patch_supervision
            else None
        )

    def token_grid(self, x: torch.Tensor) -> tuple[int, int, int]:
        patch_size = int(getattr(self.backbone, "patch_size", 16))
        tubelet_size = int(getattr(self.backbone, "tubelet_size", 2))
        return int(x.shape[2] // tubelet_size), int(x.shape[3] // patch_size), int(x.shape[4] // patch_size)

    def forward(self, x: torch.Tensor, return_patch_logits: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        hidden = self.backbone(x)
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[-1]
        pooled = hidden.mean(dim=1)
        case_logits = self.head(pooled)
        if not return_patch_logits:
            return case_logits

        if self.patch_head is None:
            raise RuntimeError("This V-JEPA 2.1 classifier was built without a patch head")
        grid = self.token_grid(x)
        expected_tokens = grid[0] * grid[1] * grid[2]
        if hidden.shape[1] != expected_tokens:
            raise RuntimeError(
                f"V-JEPA token count {hidden.shape[1]} does not match input-derived grid {grid} "
                f"({expected_tokens} tokens)"
            )
        patch_hidden = hidden
        if self.patch_output_shape is not None and self.patch_output_shape != grid:
            feature_map = hidden.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[2], *grid)
            feature_map = F.adaptive_avg_pool3d(feature_map, self.patch_output_shape)
            patch_hidden = feature_map.flatten(2).transpose(1, 2).contiguous()
            grid = self.patch_output_shape
        if isinstance(self.patch_head, nn.Linear):
            patch_logits = self.patch_head(patch_hidden).transpose(1, 2).reshape(x.shape[0], -1, *grid)
            return {"logits": case_logits, "patch_logits": patch_logits}
        patch_logits, aux = self.patch_head(patch_hidden, grid)
        outputs = {"logits": case_logits, "patch_logits": patch_logits}
        outputs.update(aux)
        return outputs


class ClassTokenPatchMILHead(nn.Module):
    """Disease class tokens query V-JEPA patch tokens to make evidence maps.

    Decoder depth one retains the original parameter names and numerical path.
    Deeper decoders refine the disease tokens with independently trainable
    copies of the first cross-attention block.  Zero-initialized residual gates
    keep those refinements inert initially while their attention maps can still
    receive direct localization supervision.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.1,
        decoder_depth: int = 1,
        coord_embedding: bool = False,
        coord_mode: str = "add",
        anatomical_prior: bool = False,
        prior_gamma: float = 1.0,
        prior_grid: Sequence[int] = (32, 24, 24),
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"class-token heads {num_heads} must divide hidden_size {hidden_size}")
        if not 1 <= int(decoder_depth) <= 3:
            raise ValueError(f"class-token decoder depth must be in [1, 3], got {decoder_depth}")
        if coord_mode not in {"add", "concat"}:
            raise ValueError(f"coord_mode must be 'add' or 'concat', got {coord_mode!r}")
        prior_grid_tuple = tuple(int(v) for v in prior_grid)
        if len(prior_grid_tuple) != 3 or any(v <= 0 for v in prior_grid_tuple):
            raise ValueError(f"prior_grid must contain three positive integers, got {prior_grid}")
        self.num_classes = num_classes
        self.scale = hidden_size**-0.5
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.decoder_depth = int(decoder_depth)
        self.coord_embedding = bool(coord_embedding)
        self.coord_mode = coord_mode
        self.anatomical_prior = bool(anatomical_prior)
        self.prior_gamma = float(prior_gamma)
        self.prior_grid = prior_grid_tuple
        self.class_tokens = nn.Parameter(torch.empty(num_classes, hidden_size))
        self.query_norm = nn.LayerNorm(hidden_size)
        self.patch_norm = nn.LayerNorm(hidden_size)
        if self.coord_embedding:
            self.coord_mlp = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.coord_norm = nn.LayerNorm(hidden_size)
            self.coord_fusion = (
                nn.Sequential(
                    nn.LayerNorm(hidden_size * 2),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                if coord_mode == "concat"
                else None
            )
        else:
            self.coord_mlp = None
            self.coord_norm = None
            self.coord_fusion = None
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        # Deep-copying avoids advancing the global RNG, so all shared modules
        # retain identical initialization between depth-one and depth-three
        # ablations.  The copied parameters remain independent and trainable.
        self.extra_query_norms = nn.ModuleList(
            copy.deepcopy(self.query_norm) for _ in range(self.decoder_depth - 1)
        )
        self.extra_cross_attns = nn.ModuleList(
            copy.deepcopy(self.cross_attn) for _ in range(self.decoder_depth - 1)
        )
        self.extra_output_norms = nn.ModuleList(
            copy.deepcopy(self.output_norm) for _ in range(self.decoder_depth - 1)
        )
        if self.decoder_depth > 1:
            self.extra_residual_gates = nn.Parameter(torch.zeros(self.decoder_depth - 1))
        else:
            self.register_parameter("extra_residual_gates", None)
        if self.anatomical_prior:
            self.anatomical_prior_bias = nn.Parameter(torch.zeros(num_classes, *prior_grid_tuple))
        else:
            self.register_parameter("anatomical_prior_bias", None)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.trunc_normal_(self.class_tokens, std=0.02)

    @staticmethod
    def _normalized_coordinates(
        grid: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        axes = [
            torch.linspace(0.0, 1.0, steps=size, device=device, dtype=dtype) if size > 1 else torch.zeros(1, device=device, dtype=dtype)
            for size in grid
        ]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        return torch.stack((zz, yy, xx), dim=-1).reshape(-1, 3)

    def _add_coordinate_embedding(
        self,
        patch_tokens: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        if not self.coord_embedding or self.coord_mlp is None or self.coord_norm is None:
            return patch_tokens
        coords = self._normalized_coordinates(grid, patch_tokens.device, patch_tokens.dtype)
        coord_tokens = self.coord_mlp(coords).unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        if self.coord_mode == "add":
            return self.coord_norm(patch_tokens + coord_tokens)
        if self.coord_fusion is None:
            raise RuntimeError("coord_fusion is missing for concat coordinate mode")
        return self.coord_norm(self.coord_fusion(torch.cat((patch_tokens, coord_tokens), dim=-1)))

    def _interpolated_prior_bias(
        self,
        grid: tuple[int, int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.anatomical_prior or self.anatomical_prior_bias is None:
            return None
        prior = self.anatomical_prior_bias
        if tuple(prior.shape[1:]) != grid:
            prior = F.interpolate(
                prior[:, None],
                size=grid,
                mode="trilinear",
                align_corners=False,
            )[:, 0]
        return prior.reshape(self.num_classes, -1).to(dtype=dtype)

    def _cross_attention_with_bias(
        self,
        attention_module: nn.MultiheadAttention,
        query_input: torch.Tensor,
        patch_tokens: torch.Tensor,
        prior_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_classes, hidden_size = query_input.shape
        num_patches = patch_tokens.shape[1]
        head_dim = hidden_size // self.num_heads
        in_proj_bias = attention_module.in_proj_bias
        q_weight, k_weight, v_weight = attention_module.in_proj_weight.chunk(3, dim=0)
        if in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)

        q = F.linear(query_input, q_weight, q_bias)
        k = F.linear(patch_tokens, k_weight, k_bias)
        v = F.linear(patch_tokens, v_weight, v_bias)
        q = q.view(batch_size, num_classes, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, num_patches, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, num_patches, self.num_heads, head_dim).transpose(1, 2)

        attention_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attention_logits = attention_logits + self.prior_gamma * prior_bias.view(1, 1, num_classes, num_patches)
        attention = F.softmax(attention_logits, dim=-1)
        attention = F.dropout(attention, p=float(attention_module.dropout), training=self.training)
        attended = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, num_classes, hidden_size)
        attended = attention_module.out_proj(attended)
        evidence_attention = attention.mean(dim=1)
        return attended, evidence_attention

    def forward(self, hidden: torch.Tensor, grid: tuple[int, int, int]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden.shape[0]
        patch_tokens = self.patch_norm(hidden)
        patch_tokens = self._add_coordinate_embedding(patch_tokens, grid)
        queries = self.class_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        query_input = self.query_norm(queries)
        prior_bias = self._interpolated_prior_bias(grid, patch_tokens.dtype)
        if prior_bias is None:
            attended, attention = self.cross_attn(
                query_input,
                patch_tokens,
                patch_tokens,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attended, attention = self._cross_attention_with_bias(
                self.cross_attn,
                query_input,
                patch_tokens,
                prior_bias,
            )
        disease_tokens = self.output_norm(queries + attended)

        # Keep the depth-one path byte-for-byte equivalent in its tensor
        # operations and output contract.  In particular, do not stack and
        # average a singleton attention tensor here.
        if self.decoder_depth == 1:
            patch_logits = torch.einsum("bcd,bnd->bcn", disease_tokens, patch_tokens) * self.scale
            patch_logits = patch_logits + self.bias.view(1, -1, 1)
            patch_logits = patch_logits.reshape(batch_size, self.num_classes, *grid)
            class_patch_relation_scores = torch.einsum(
                "bcd,bnd->bcn",
                F.normalize(disease_tokens.float(), dim=-1),
                F.normalize(patch_tokens.float(), dim=-1),
            ).reshape(batch_size, self.num_classes, *grid)
            if attention.dim() == 4:
                evidence_attention = attention.mean(dim=1)
            else:
                evidence_attention = attention
            evidence_attention = evidence_attention.reshape(batch_size, self.num_classes, *grid)
            aux = {
                "evidence_attention": evidence_attention,
                "class_token_features": disease_tokens,
                "class_patch_relation_scores": class_patch_relation_scores,
                "patch_token_features": patch_tokens,
            }
            if prior_bias is not None:
                aux["anatomical_prior_bias"] = prior_bias.reshape(self.num_classes, *grid)
            return patch_logits, aux

        if self.extra_residual_gates is None:
            raise RuntimeError("Multi-level class-token decoder residual gates are missing")
        layer_features = [disease_tokens]
        layer_attentions = [attention.mean(dim=1) if attention.dim() == 4 else attention]
        for layer_index, (query_norm, cross_attn, output_norm) in enumerate(
            zip(self.extra_query_norms, self.extra_cross_attns, self.extra_output_norms)
        ):
            query_input = query_norm(disease_tokens)
            if prior_bias is None:
                attended, attention = cross_attn(
                    query_input,
                    patch_tokens,
                    patch_tokens,
                    need_weights=True,
                    average_attn_weights=False,
                )
            else:
                attended, attention = self._cross_attention_with_bias(
                    cross_attn,
                    query_input,
                    patch_tokens,
                    prior_bias,
                )
            candidate = output_norm(disease_tokens + attended)
            residual_gate = torch.tanh(self.extra_residual_gates[layer_index]).to(
                dtype=disease_tokens.dtype
            )
            disease_tokens = disease_tokens + residual_gate * (candidate - disease_tokens)
            layer_features.append(disease_tokens)
            layer_attentions.append(attention.mean(dim=1) if attention.dim() == 4 else attention)

        patch_logits = torch.einsum("bcd,bnd->bcn", disease_tokens, patch_tokens) * self.scale
        patch_logits = patch_logits + self.bias.view(1, -1, 1)
        patch_logits = patch_logits.reshape(batch_size, self.num_classes, *grid)
        class_patch_relation_scores = torch.einsum(
            "bcd,bnd->bcn",
            F.normalize(disease_tokens.float(), dim=-1),
            F.normalize(patch_tokens.float(), dim=-1),
        ).reshape(batch_size, self.num_classes, *grid)
        class_token_features_layers = torch.stack(layer_features, dim=1)
        evidence_attention_layers = torch.stack(layer_attentions, dim=1).reshape(
            batch_size,
            self.decoder_depth,
            self.num_classes,
            *grid,
        )
        evidence_attention = evidence_attention_layers.mean(dim=1)
        aux = {
            "evidence_attention": evidence_attention,
            "class_token_features": disease_tokens,
            "class_patch_relation_scores": class_patch_relation_scores,
            "patch_token_features": patch_tokens,
            "class_token_features_layers": class_token_features_layers,
            "evidence_attention_layers": evidence_attention_layers,
        }
        if prior_bias is not None:
            aux["anatomical_prior_bias"] = prior_bias.reshape(self.num_classes, *grid)
        return patch_logits, aux


class PrototypePatchMILHead(nn.Module):
    """Disease-specific prototypes score each V-JEPA patch token."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        *,
        prototype_count: int = 4,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if prototype_count <= 0:
            raise ValueError(f"prototype_count must be positive, got {prototype_count}")
        if temperature <= 0.0:
            raise ValueError(f"prototype_temperature must be positive, got {temperature}")
        self.num_classes = num_classes
        self.prototype_count = prototype_count
        self.temperature = float(temperature)
        self.patch_norm = nn.LayerNorm(hidden_size)
        self.prototypes = nn.Parameter(torch.empty(num_classes, prototype_count, hidden_size))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, hidden: torch.Tensor, grid: tuple[int, int, int]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden.shape[0]
        patch_tokens = F.normalize(self.patch_norm(hidden), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = torch.einsum("bnd,ckd->bcnk", patch_tokens, prototypes) / self.temperature
        patch_logits = torch.logsumexp(similarity, dim=-1) - torch.log(
            torch.as_tensor(float(self.prototype_count), device=hidden.device, dtype=hidden.dtype)
        )
        patch_logits = patch_logits + self.bias.view(1, -1, 1)
        patch_logits = patch_logits.reshape(batch_size, self.num_classes, *grid)
        prototype_similarity = similarity.amax(dim=-1).reshape(batch_size, self.num_classes, *grid)
        return patch_logits, {"prototype_similarity": prototype_similarity}


def build_patch_mil_head(
    *,
    mil_head: str,
    hidden_size: int,
    num_classes: int,
    class_token_heads: int,
    class_token_dropout: float,
    class_token_coord_embedding: bool = False,
    class_token_coord_mode: str = "add",
    class_token_anatomical_prior: bool = False,
    class_token_prior_gamma: float = 1.0,
    class_token_prior_grid: Sequence[int] = (32, 24, 24),
    prototype_count: int = 4,
    prototype_temperature: float = 0.1,
) -> nn.Module:
    if mil_head == "basic":
        return nn.Linear(hidden_size, num_classes)
    if mil_head == "class-token":
        return ClassTokenPatchMILHead(
            hidden_size,
            num_classes,
            num_heads=class_token_heads,
            dropout=class_token_dropout,
            coord_embedding=class_token_coord_embedding,
            coord_mode=class_token_coord_mode,
            anatomical_prior=class_token_anatomical_prior,
            prior_gamma=class_token_prior_gamma,
            prior_grid=class_token_prior_grid,
        )
    if mil_head == "prototype":
        return PrototypePatchMILHead(
            hidden_size,
            num_classes,
            prototype_count=prototype_count,
            temperature=prototype_temperature,
        )
    raise ValueError(f"Unsupported MIL head {mil_head!r}; choose basic, class-token, or prototype")


def _build_torchvision_model(spec: ModelSpec, num_classes: int, pretrained: bool) -> nn.Module:
    if spec.family == "torchvision_mvit_v2_s":
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        model = mvit_v2_s(weights=weights)
    elif spec.family == "torchvision_r3d18":
        from torchvision.models.video import R3D_18_Weights, r3d_18

        weights = R3D_18_Weights.DEFAULT if pretrained else None
        model = r3d_18(weights=weights)
    else:
        raise ValueError(f"Unsupported torchvision family {spec.family!r}")

    if not _replace_last_classifier(model, num_classes):
        raise RuntimeError(f"Could not locate the classifier head for {spec.name}")
    return model


def _load_pytorchvideo_from_package(builder_name: str) -> Callable[..., nn.Module]:
    try:
        from pytorchvideo.models import hub as pytorchvideo_hub
    except ModuleNotFoundError as e:
        raise _missing_optional_package(
            "pytorchvideo",
            builder_name,
            "python -m pip install -U git+https://github.com/facebookresearch/pytorchvideo",
        ) from e
    return getattr(pytorchvideo_hub, builder_name)


def _build_pytorchvideo_model(spec: ModelSpec, num_classes: int, pretrained: bool) -> nn.Module:
    builder_name = spec.family.removeprefix("pytorchvideo_")
    try:
        builder = _load_pytorchvideo_from_package(builder_name)
        model = builder(pretrained=pretrained)
    except RuntimeError:
        model = torch.hub.load("facebookresearch/pytorchvideo", builder_name, pretrained=pretrained)

    if not _replace_last_classifier(model, num_classes):
        raise RuntimeError(f"Could not locate the classifier head for {spec.name}")
    return model


def build_model(
    spec: ModelSpec,
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
    attn_implementation: str | None = "sdpa",
    patch_supervision: bool = False,
    mil_head: str = "basic",
    class_token_heads: int = 8,
    class_token_dropout: float = 0.1,
    class_token_coord_embedding: bool = False,
    class_token_coord_mode: str = "add",
    class_token_anatomical_prior: bool = False,
    class_token_prior_gamma: float = 1.0,
    class_token_prior_grid: Sequence[int] = (32, 24, 24),
    patch_output_shape: Sequence[int] | None = None,
    prototype_count: int = 4,
    prototype_temperature: float = 0.1,
    deterministic: bool = False,
) -> nn.Module:
    if spec.default_model_id is None and spec.family.startswith("hf_"):
        raise ValueError(f"{spec.name} requires a Hugging Face model id")

    if spec.family == "hf_videomae":
        model = HFVideoClassificationWrapper(
            _load_hf_classification_model(
                "VideoMAEForVideoClassification",
                str(spec.default_model_id),
                num_classes,
                pretrained,
                attn_implementation,
            )
        )
    elif spec.family == "hf_timesformer":
        timesformer_attn = "eager" if attn_implementation == "sdpa" else attn_implementation
        timesformer_model_id = str(spec.default_model_id)
        timesformer_revision: str | None = None
        timesformer_use_safetensors: bool | None = None
        if pretrained:
            local_safetensors_dir = _local_timesformer_safetensors_dir(timesformer_model_id)
            if local_safetensors_dir is not None:
                timesformer_model_id = local_safetensors_dir
                timesformer_use_safetensors = True
            elif timesformer_model_id == "facebook/timesformer-base-finetuned-k400":
                timesformer_revision = "refs/pr/5"
                timesformer_use_safetensors = True
        model = HFVideoClassificationWrapper(
            _load_hf_classification_model(
                "TimesformerForVideoClassification",
                timesformer_model_id,
                num_classes,
                pretrained,
                timesformer_attn,
                revision=timesformer_revision,
                use_safetensors=timesformer_use_safetensors,
            )
        )
    elif spec.family == "hf_vjepa2":
        model = VJEPA2Classifier(
            str(spec.default_model_id),
            num_classes,
            pretrained=pretrained,
            dropout=dropout,
            attn_implementation=attn_implementation,
        )
    elif spec.family == "torchhub_vjepa2_1":
        model = VJEPA21TorchHubClassifier(
            str(spec.default_model_id),
            num_classes,
            pretrained=pretrained,
            dropout=dropout,
            patch_supervision=patch_supervision,
            mil_head=mil_head,
            class_token_heads=class_token_heads,
            class_token_dropout=class_token_dropout,
            class_token_coord_embedding=class_token_coord_embedding,
            class_token_coord_mode=class_token_coord_mode,
            class_token_anatomical_prior=class_token_anatomical_prior,
            class_token_prior_gamma=class_token_prior_gamma,
            class_token_prior_grid=class_token_prior_grid,
            patch_output_shape=patch_output_shape,
            prototype_count=prototype_count,
            prototype_temperature=prototype_temperature,
        )
    elif spec.family.startswith("torchvision_"):
        model = _build_torchvision_model(spec, num_classes, pretrained)
    elif spec.family.startswith("pytorchvideo_"):
        model = _build_pytorchvideo_model(spec, num_classes, pretrained)
    else:
        raise ValueError(f"Unsupported model family {spec.family!r}")

    if deterministic:
        deterministic_replacements = {
            "max_pool3d": replace_max_pool3d(model),
            "avg_pool3d": replace_avg_pool3d(model),
            "adaptive_global_avg_pool3d": replace_adaptive_global_avg_pool3d(model),
        }
        setattr(model, "_deterministic_replacements", deterministic_replacements)
    if freeze_backbone:
        _freeze_module(model)
        _unfreeze_classifier_heads(model, num_classes)
    return model


def trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable

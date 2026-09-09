"""V-JEPA 2.1 dense patch classifier for the PatchChestCT official protocol."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification_code.patchchestct_video_models.models import (
    ClassTokenPatchMILHead,
    VJEPA21TorchHubClassifier,
)
from classification_code.patchchestct_grid import (
    ANATOMICAL_GRID_V2,
    LEGACY_OFFICIAL_GRID,
    PATCH_GRID_PROTOCOLS,
)
from classification_code.patchchestct_pooling import (
    physical_mean_pool3d,
    physical_center_linear_resample3d,
    smooth_logmeanexp_pool3d,
)


class _DeterministicAdaptiveAvgPool3dFunction(torch.autograd.Function):
    """CUDA adaptive-average forward with a deterministic CPU backward."""

    @staticmethod
    def forward(
        ctx: object,
        x: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        ctx.input_shape = tuple(x.shape)  # type: ignore[attr-defined]
        ctx.input_device = x.device  # type: ignore[attr-defined]
        ctx.input_dtype = x.dtype  # type: ignore[attr-defined]
        ctx.output_shape = output_shape  # type: ignore[attr-defined]
        return F.adaptive_avg_pool3d(x, output_shape)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        input_shape = ctx.input_shape  # type: ignore[attr-defined]
        input_device = ctx.input_device  # type: ignore[attr-defined]
        input_dtype = ctx.input_dtype  # type: ignore[attr-defined]
        output_shape = ctx.output_shape  # type: ignore[attr-defined]
        with torch.enable_grad():
            cpu_input = torch.zeros(
                input_shape,
                dtype=input_dtype,
                device="cpu",
                requires_grad=True,
            )
            cpu_output = F.adaptive_avg_pool3d(cpu_input, output_shape)
            (grad_input_cpu,) = torch.autograd.grad(
                cpu_output,
                cpu_input,
                grad_outputs=grad_output.detach().to(device="cpu"),
                create_graph=False,
            )
        return grad_input_cpu.to(device=input_device), None


class CropAwareAnatomicalEvidence(nn.Module):
    """Learn a smooth disease-specific spatial prior in absolute crop coordinates."""

    def __init__(
        self,
        num_classes: int,
        output_shape: Sequence[int],
        pad_shape: Sequence[int],
        crop_shape: Sequence[int],
        hidden_dim: int = 64,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("anatomical evidence hidden_dim must be positive")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("anatomical evidence gate_init must be strictly between 0 and 1")
        self.num_classes = int(num_classes)
        self.output_shape = tuple(int(value) for value in output_shape)
        self.pad_shape = tuple(float(value) for value in pad_shape)
        self.crop_shape = tuple(float(value) for value in crop_shape)
        if any(value <= 0 for value in (*self.output_shape, *self.pad_shape, *self.crop_shape)):
            raise ValueError("anatomical evidence shapes must contain positive values")

        self.coord_mlp = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.gate_logit = nn.Parameter(
            torch.full(
                (num_classes,),
                math.log(gate_init / (1.0 - gate_init)),
            )
        )
        # The residual starts at exactly zero, preserving the initialized baseline.
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)

    def _coordinate_basis(
        self,
        crop_starts: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = crop_starts.device
        axes = [
            (torch.arange(size, device=device, dtype=dtype) + 0.5)
            * (crop_size / float(size))
            for size, crop_size in zip(self.output_shape, self.crop_shape)
        ]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        local_centers = torch.stack((zz, yy, xx), dim=-1).reshape(1, -1, 3)
        absolute_centers = local_centers + crop_starts.to(dtype=dtype).unsqueeze(1)
        pad = torch.tensor(self.pad_shape, device=device, dtype=dtype).view(1, 1, 3)
        coords = 2.0 * absolute_centers / pad - 1.0
        z, y, x = coords.unbind(dim=-1)
        return torch.stack((z, y, x, z.square(), y.square(), x.square(), z * y, z * x, y * x), dim=-1)

    def forward(
        self,
        crop_starts: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis = self._coordinate_basis(crop_starts, dtype)
        prior_bias = self.coord_mlp(basis).transpose(1, 2)
        prior_bias = prior_bias - prior_bias.mean(dim=-1, keepdim=True)
        gate = self.gate_logit.sigmoid().view(1, -1, 1)
        residual = gate * prior_bias
        evidence_attention = F.softmax(prior_bias.float(), dim=-1).to(dtype=prior_bias.dtype)
        batch_size = crop_starts.shape[0]
        grid = self.output_shape
        return (
            residual.reshape(batch_size, self.num_classes, *grid),
            evidence_attention.reshape(batch_size, self.num_classes, *grid),
            gate.reshape(1, self.num_classes, 1, 1, 1),
        )


class VJEPA21OfficialPatchClassifier(nn.Module):
    """Adapt V-JEPA 2.1 dense tokens to the official 6x12x12 patch grid."""

    def __init__(
        self,
        num_classes: int = 18,
        pretrained: bool = True,
        output_shape: Sequence[int] = (6, 12, 12),
        native_input_shape: Sequence[int] = (64, 384, 384),
        mil_head: str = "basic",
        class_token_heads: int = 8,
        class_token_dropout: float = 0.1,
        class_token_decoder_depth: int = 1,
        class_token_coord_embedding: bool = False,
        class_token_coord_mode: str = "add",
        class_token_anatomical_prior: bool = False,
        class_token_prior_gamma: float = 1.0,
        class_token_prior_grid: Sequence[int] = (6, 12, 12),
        class_token_residual_init: float = 0.1,
        anatomical_evidence: bool = False,
        anatomical_evidence_hidden_dim: int = 64,
        anatomical_evidence_gate_init: float = 0.1,
        anatomical_pad_shape: Sequence[int] = (120, 240, 240),
        anatomical_crop_shape: Sequence[int] = (96, 192, 192),
        global_local_fusion: bool = False,
        local_case_pooling: str = "max",
        local_case_topk: int = 4,
        adaptive_pool_topks: Sequence[int] = (1, 4, 16, 0),
        adaptive_pool_init_weights: Sequence[float] = (0.05, 0.85, 0.08, 0.02),
        gwrp_decay: float = 0.996,
        fusion_local_init: float = 0.8,
        mct_attention_residual_max: float = 0.25,
        debug_breakpoints: bool = False,
        deterministic_adaptive_pool: bool = False,
        patch_grid_protocol: str = LEGACY_OFFICIAL_GRID,
        patch_token_pooling: str = "mean",
        smooth_or_temperature: float = 1.0,
        fine_annotation_supervision: bool = False,
        fine_annotation_shape: Sequence[int] = (24, 12, 12),
    ) -> None:
        super().__init__()
        base = VJEPA21TorchHubClassifier(
            "vjepa2_1_vit_base_384",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=0.0,
            patch_supervision=False,
        )
        self.backbone = base.backbone
        hidden_size = int(getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features", 768)))
        if mil_head not in {
            "basic",
            "class-token",
            "hybrid-class-token",
            "decoupled-class-token",
            "mct-decoupled",
        }:
            raise ValueError(f"Unsupported official V-JEPA MIL head: {mil_head!r}")
        if not 0.0 < class_token_residual_init < 1.0:
            raise ValueError("class_token_residual_init must be strictly between 0 and 1")
        self.mil_head = mil_head
        self.classifier = (
            nn.Linear(hidden_size, num_classes)
            if mil_head in {"basic", "hybrid-class-token", "decoupled-class-token", "mct-decoupled"}
            else None
        )
        self.patch_head = (
            ClassTokenPatchMILHead(
                hidden_size,
                num_classes,
                num_heads=class_token_heads,
                dropout=class_token_dropout,
                decoder_depth=class_token_decoder_depth,
                coord_embedding=class_token_coord_embedding,
                coord_mode=class_token_coord_mode,
                anatomical_prior=class_token_anatomical_prior,
                prior_gamma=class_token_prior_gamma,
                prior_grid=class_token_prior_grid,
            )
            if mil_head in {"class-token", "hybrid-class-token", "decoupled-class-token", "mct-decoupled"}
            else None
        )
        self.class_token_case_head = (
            nn.Linear(hidden_size, 1)
            if mil_head == "mct-decoupled"
            else None
        )
        if mil_head == "mct-decoupled" and self.patch_head is not None:
            # The MCT-decoupled path consumes disease-token features and
            # attention maps, not the legacy class-token map's constant bias.
            self.patch_head.bias.requires_grad_(False)
        self.mct_attention_residual_gate = (
            nn.Parameter(torch.zeros(num_classes))
            if mil_head == "mct-decoupled"
            else None
        )
        if mct_attention_residual_max <= 0.0:
            raise ValueError("mct_attention_residual_max must be positive")
        self.mct_attention_residual_max = float(mct_attention_residual_max)
        self.class_token_residual_logit = (
            nn.Parameter(
                torch.full(
                    (num_classes,),
                    math.log(class_token_residual_init / (1.0 - class_token_residual_init)),
                )
            )
            if mil_head == "hybrid-class-token"
            else None
        )
        self.output_shape = tuple(int(v) for v in output_shape)
        self.native_input_shape = tuple(int(v) for v in native_input_shape)
        self.deterministic_adaptive_pool = bool(deterministic_adaptive_pool)
        if patch_grid_protocol not in PATCH_GRID_PROTOCOLS:
            valid = ", ".join(PATCH_GRID_PROTOCOLS)
            raise ValueError(
                f"Unsupported patch_grid_protocol {patch_grid_protocol!r}; choose one of: {valid}"
            )
        if patch_token_pooling not in {"mean", "smooth-or"}:
            raise ValueError(
                f"Unsupported patch_token_pooling {patch_token_pooling!r}; "
                "choose 'mean' or 'smooth-or'"
            )
        if not math.isfinite(float(smooth_or_temperature)) or smooth_or_temperature <= 0.0:
            raise ValueError("smooth_or_temperature must be finite and positive")
        if patch_token_pooling == "smooth-or" and patch_grid_protocol != ANATOMICAL_GRID_V2:
            raise ValueError(
                "smooth-or pooling requires anatomical_grid_v2_6x12x12 so predictions and "
                "targets use the same mutually exclusive physical bins"
            )
        if patch_token_pooling == "smooth-or" and mil_head != "basic":
            raise ValueError("smooth-or pooling currently supports only the basic linear MIL head")
        if patch_token_pooling == "smooth-or" and global_local_fusion:
            raise ValueError(
                "smooth-or pooling keeps case prediction as max over the same patch map and "
                "therefore cannot be combined with global-local fusion"
            )
        fine_shape = tuple(int(value) for value in fine_annotation_shape)
        if len(fine_shape) != 3 or any(value <= 0 for value in fine_shape):
            raise ValueError("fine_annotation_shape must contain three positive integers")
        if fine_annotation_supervision and (
            patch_grid_protocol != ANATOMICAL_GRID_V2 or mil_head != "basic"
        ):
            raise ValueError(
                "fine annotation supervision requires anatomical-grid-v2 and the basic head"
            )
        self.patch_grid_protocol = patch_grid_protocol
        self.patch_token_pooling = patch_token_pooling
        self.smooth_or_temperature = float(smooth_or_temperature)
        self.fine_annotation_supervision = bool(fine_annotation_supervision)
        self.fine_annotation_shape = fine_shape
        if local_case_pooling not in {"max", "topk", "adaptive", "gwrp"}:
            raise ValueError(f"Unsupported local case pooling: {local_case_pooling!r}")
        if not 0.0 < gwrp_decay <= 1.0:
            raise ValueError("gwrp_decay must be in (0, 1]")
        if local_case_topk <= 0:
            raise ValueError("local_case_topk must be positive")
        adaptive_topks = tuple(int(value) for value in adaptive_pool_topks)
        adaptive_weights = tuple(float(value) for value in adaptive_pool_init_weights)
        if not adaptive_topks or any(value < 0 for value in adaptive_topks):
            raise ValueError("adaptive_pool_topks must contain non-negative values; zero denotes global mean")
        if len(adaptive_topks) != len(adaptive_weights):
            raise ValueError("adaptive pooling top-k and initial-weight lists must have equal length")
        if any(value <= 0.0 for value in adaptive_weights):
            raise ValueError("adaptive pooling initial weights must be positive")
        if not 0.0 < fusion_local_init < 1.0:
            raise ValueError("fusion_local_init must be strictly between 0 and 1")
        self.global_local_fusion = bool(global_local_fusion)
        self.local_case_pooling = local_case_pooling
        self.local_case_topk = int(local_case_topk)
        self.gwrp_decay = float(gwrp_decay)
        self.adaptive_pool_topks = adaptive_topks
        self.global_classifier = (
            nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, num_classes))
            if self.global_local_fusion and mil_head not in {"decoupled-class-token", "mct-decoupled"}
            else None
        )
        self.fusion_logit = (
            nn.Parameter(torch.full((num_classes,), math.log(fusion_local_init / (1.0 - fusion_local_init))))
            if self.global_local_fusion
            else None
        )
        normalized_pool_weights = torch.tensor(adaptive_weights, dtype=torch.float32)
        normalized_pool_weights = normalized_pool_weights / normalized_pool_weights.sum()
        self.adaptive_pool_logits = (
            nn.Parameter(normalized_pool_weights.log().repeat(num_classes, 1))
            if self.local_case_pooling == "adaptive"
            else None
        )
        self.debug_breakpoints = bool(debug_breakpoints)
        self._debug_breakpoints_hit: set[str] = set()
        # Initialize optional modules after shared heads so paired runs retain
        # identical shared-parameter initialization for the same random seed.
        self.crop_aware_anatomical_evidence = bool(anatomical_evidence)
        self.anatomical_evidence = (
            CropAwareAnatomicalEvidence(
                num_classes,
                self.output_shape,
                anatomical_pad_shape,
                anatomical_crop_shape,
                hidden_dim=anatomical_evidence_hidden_dim,
                gate_init=anatomical_evidence_gate_init,
            )
            if anatomical_evidence
            else None
        )
        self.register_buffer("input_mean", torch.tensor(0.45, dtype=torch.float32), persistent=False)
        self.register_buffer("input_std", torch.tensor(0.225, dtype=torch.float32), persistent=False)

    def _local_case_logits(self, patch_logits: torch.Tensor) -> torch.Tensor:
        flat = patch_logits.flatten(2)
        if self.local_case_pooling == "max":
            return flat.amax(dim=-1)
        if self.local_case_pooling == "topk":
            topk = min(self.local_case_topk, flat.shape[-1])
            return flat.topk(topk, dim=-1).values.mean(dim=-1)
        if self.local_case_pooling == "gwrp":
            ranked_logits = flat.sort(dim=-1, descending=True).values
            ranks = torch.arange(
                ranked_logits.shape[-1],
                device=ranked_logits.device,
                dtype=torch.float32,
            )
            weights = self.gwrp_decay**ranks
            weights = (weights / weights.sum()).to(dtype=ranked_logits.dtype)
            return (ranked_logits * weights.view(1, 1, -1)).sum(dim=-1)
        if self.adaptive_pool_logits is None:
            raise RuntimeError("Adaptive pooling logits are missing")
        experts: list[torch.Tensor] = []
        for topk in self.adaptive_pool_topks:
            if topk == 0:
                experts.append(flat.mean(dim=-1))
            else:
                count = min(topk, flat.shape[-1])
                experts.append(flat.topk(count, dim=-1).values.mean(dim=-1))
        expert_logits = torch.stack(experts, dim=-1)
        weights = F.softmax(self.adaptive_pool_logits, dim=-1).to(dtype=expert_logits.dtype)
        if self.debug_breakpoints and "adaptive_pool" not in self._debug_breakpoints_hit:
            self._debug_breakpoints_hit.add("adaptive_pool")
            print("[debug 2/4] Adaptive pooling experts and disease-specific weights", flush=True)
            breakpoint()
        return (expert_logits * weights.unsqueeze(0)).sum(dim=-1)

    def _add_global_local_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        patch_logits: torch.Tensor,
        global_case_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.global_local_fusion:
            return outputs
        if self.fusion_logit is None:
            raise RuntimeError("Global-local fusion logit is missing")
        if global_case_logits is None:
            if self.global_classifier is None:
                raise RuntimeError("Global classifier is missing")
            global_case_logits = self.global_classifier(hidden.mean(dim=1))
        local_case_logits = self._local_case_logits(patch_logits)
        local_weight = self.fusion_logit.sigmoid().view(1, -1)
        if self.debug_breakpoints and "dual_fusion" not in self._debug_breakpoints_hit:
            self._debug_breakpoints_hit.add("dual_fusion")
            print("[debug 3/4] Global/local logits and fusion weights", flush=True)
            breakpoint()
        outputs.update(
            {
                "case_logits": local_weight * local_case_logits + (1.0 - local_weight) * global_case_logits,
                "global_case_logits": global_case_logits,
                "local_case_logits": local_case_logits,
                "fusion_local_weight": local_weight.expand(hidden.shape[0], -1),
            }
        )
        if self.adaptive_pool_logits is not None:
            adaptive_weights = F.softmax(self.adaptive_pool_logits, dim=-1)
            outputs["adaptive_pool_weights"] = adaptive_weights.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        return outputs

    def _add_anatomical_evidence(
        self,
        outputs: dict[str, torch.Tensor],
        patch_logits: torch.Tensor,
        crop_starts: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if not self.crop_aware_anatomical_evidence:
            return outputs, patch_logits
        if self.anatomical_evidence is None:
            raise RuntimeError("Crop-aware anatomical evidence module is missing")
        if crop_starts is None:
            pad = torch.tensor(self.anatomical_evidence.pad_shape, device=patch_logits.device)
            crop = torch.tensor(self.anatomical_evidence.crop_shape, device=patch_logits.device)
            crop_starts = ((pad - crop) / 2.0).view(1, 3).expand(patch_logits.shape[0], -1)
        if crop_starts.ndim != 2 or tuple(crop_starts.shape) != (patch_logits.shape[0], 3):
            raise ValueError(
                f"Expected crop_starts shape ({patch_logits.shape[0]}, 3), got {tuple(crop_starts.shape)}"
            )
        residual, evidence_attention, gate = self.anatomical_evidence(
            crop_starts.to(device=patch_logits.device),
            patch_logits.dtype,
        )
        patch_logits = patch_logits + residual
        outputs.update(
            {
                "anatomical_evidence_bias": residual,
                "anatomical_evidence_attention": evidence_attention,
                "anatomical_evidence_gate": gate.expand(patch_logits.shape[0], -1, -1, -1, -1),
                "evidence_attention": evidence_attention,
            }
        )
        return outputs, patch_logits

    def forward(
        self,
        x: torch.Tensor,
        crop_starts: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim != 5 or x.shape[1] not in {1, 3}:
            raise ValueError(f"Expected input shape (B,1|3,D,H,W), got {tuple(x.shape)}")
        if self.debug_breakpoints and "multiwindow_input" not in self._debug_breakpoints_hit:
            self._debug_breakpoints_hit.add("multiwindow_input")
            print("[debug 1/4] Multi-window input before V-JEPA resizing", flush=True)
            breakpoint()

        x = F.interpolate(x, size=self.native_input_shape, mode="trilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        x = (x - self.input_mean.to(dtype=x.dtype)) / self.input_std.to(dtype=x.dtype)

        hidden = self.backbone(x)
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[-1]

        tubelet_size = int(getattr(self.backbone, "tubelet_size", 2))
        patch_size = int(getattr(self.backbone, "patch_size", 16))
        native_grid = (
            self.native_input_shape[0] // tubelet_size,
            self.native_input_shape[1] // patch_size,
            self.native_input_shape[2] // patch_size,
        )
        expected_tokens = native_grid[0] * native_grid[1] * native_grid[2]
        if hidden.ndim != 3 or hidden.shape[1] != expected_tokens:
            raise RuntimeError(
                f"V-JEPA returned hidden shape {tuple(hidden.shape)}; expected {expected_tokens} dense tokens"
            )

        dense_tokens: torch.Tensor | None = None
        linear_patch_logits: torch.Tensor | None = None
        fine_patch_logits: torch.Tensor | None = None
        if self.patch_grid_protocol == ANATOMICAL_GRID_V2 and self.mil_head == "basic":
            if self.classifier is None:
                raise RuntimeError("Basic V-JEPA classifier head is missing")
            # Apply the exact same Linear to every native token, then change
            # only the aggregation within mutually exclusive physical bins.
            # This makes v2 mean and Smooth-OR a controlled pair.
            native_linear_logits = self.classifier(hidden).transpose(1, 2).reshape(
                hidden.shape[0], -1, *native_grid
            )
            if self.patch_token_pooling == "mean":
                linear_patch_logits = physical_mean_pool3d(
                    native_linear_logits,
                    self.output_shape,
                )
            else:
                linear_patch_logits = smooth_logmeanexp_pool3d(
                    native_linear_logits,
                    self.output_shape,
                    self.smooth_or_temperature,
                )
            if self.fine_annotation_supervision:
                fine_patch_logits = physical_center_linear_resample3d(
                    native_linear_logits,
                    self.fine_annotation_shape,
                )
        else:
            features = hidden.transpose(1, 2).reshape(
                hidden.shape[0], hidden.shape[2], *native_grid
            )
            if self.patch_grid_protocol == LEGACY_OFFICIAL_GRID:
                # Preserve the legacy baseline path byte-for-byte: PyTorch
                # adaptive pooling uses overlapping depth windows for 32->6.
                if self.deterministic_adaptive_pool and features.is_cuda:
                    features = _DeterministicAdaptiveAvgPool3dFunction.apply(
                        features,
                        self.output_shape,
                    )
                else:
                    features = F.adaptive_avg_pool3d(features, self.output_shape)
            else:
                features = physical_mean_pool3d(features, self.output_shape)
            dense_tokens = features.flatten(2).transpose(1, 2).contiguous()
            if self.classifier is not None:
                linear_logits = self.classifier(dense_tokens).transpose(1, 2)
                linear_patch_logits = linear_logits.reshape(
                    hidden.shape[0], -1, *self.output_shape
                ).contiguous()
        if self.mil_head == "basic":
            if linear_patch_logits is None:
                raise RuntimeError("Basic V-JEPA classifier head is missing")
            outputs, linear_patch_logits = self._add_anatomical_evidence(
                {"patch_logits": linear_patch_logits},
                linear_patch_logits,
                crop_starts,
            )
            outputs["patch_logits"] = linear_patch_logits
            if fine_patch_logits is not None:
                outputs["fine_patch_logits"] = fine_patch_logits
            if (
                not self.global_local_fusion
                and not self.crop_aware_anatomical_evidence
                and fine_patch_logits is None
            ):
                return linear_patch_logits
            if not self.global_local_fusion:
                return outputs
            return self._add_global_local_outputs(
                outputs, hidden, linear_patch_logits
            )

        if self.patch_head is None:
            raise RuntimeError("Class-token V-JEPA patch head is missing")
        if dense_tokens is None:
            raise RuntimeError("Dense output-grid tokens are missing for the class-token head")
        class_token_patch_logits, aux = self.patch_head(dense_tokens, self.output_shape)
        if self.mil_head in {"decoupled-class-token", "mct-decoupled"}:
            if linear_patch_logits is None:
                raise RuntimeError("Decoupled class-token linear patch head is missing")
            if self.mil_head == "mct-decoupled":
                class_token_features = aux.get("class_token_features")
                if class_token_features is None or self.class_token_case_head is None:
                    raise RuntimeError("MCT-decoupled class-token case modules are missing")
                class_token_case_logits = self.class_token_case_head(class_token_features).squeeze(-1)
                evidence_attention = aux.get("evidence_attention")
                if evidence_attention is None or self.mct_attention_residual_gate is None:
                    raise RuntimeError("MCT-decoupled evidence modules are missing")
                num_patches = math.prod(self.output_shape)
                normalized_evidence = evidence_attention.float().clamp_min(1e-8)
                normalized_evidence = normalized_evidence / normalized_evidence.sum(
                    dim=(2, 3, 4),
                    keepdim=True,
                ).clamp_min(1e-8)
                log_evidence = torch.log(
                    normalized_evidence * float(num_patches)
                ).to(dtype=linear_patch_logits.dtype)
                residual_gate = (
                    self.mct_attention_residual_max
                    * self.mct_attention_residual_gate.tanh()
                ).view(1, -1, 1, 1, 1)
                patch_logits = linear_patch_logits + residual_gate * log_evidence
            else:
                class_token_case_logits = class_token_patch_logits.flatten(2).amax(dim=-1)
                log_evidence = None
                residual_gate = None
                patch_logits = linear_patch_logits
            outputs = {
                "patch_logits": patch_logits,
                "linear_patch_logits": linear_patch_logits,
                "class_token_patch_logits": class_token_patch_logits,
                "class_token_case_logits": class_token_case_logits,
            }
            if log_evidence is not None and residual_gate is not None:
                outputs.update(
                    {
                        "mct_log_evidence": log_evidence,
                        "mct_attention_residual_gate": residual_gate.expand(
                            hidden.shape[0], -1, 1, 1, 1
                        ),
                    }
                )
            outputs.update(aux)
            return self._add_global_local_outputs(
                outputs,
                hidden,
                patch_logits,
                global_case_logits=class_token_case_logits,
            )
        if self.mil_head == "hybrid-class-token":
            if linear_patch_logits is None or self.class_token_residual_logit is None:
                raise RuntimeError("Hybrid class-token residual modules are missing")
            residual_weight = self.class_token_residual_logit.sigmoid().view(1, -1, 1, 1, 1)
            patch_logits = linear_patch_logits + residual_weight * class_token_patch_logits
            aux.update(
                {
                    "linear_patch_logits": linear_patch_logits,
                    "class_token_patch_logits": class_token_patch_logits,
                    "class_token_residual_weight": residual_weight.expand(hidden.shape[0], -1, 1, 1, 1),
                }
            )
        else:
            patch_logits = class_token_patch_logits
        outputs = {"patch_logits": patch_logits}
        outputs.update(aux)
        outputs, patch_logits = self._add_anatomical_evidence(outputs, patch_logits, crop_starts)
        outputs["patch_logits"] = patch_logits
        return self._add_global_local_outputs(outputs, hidden, patch_logits)

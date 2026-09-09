"""VoCo-10K SwinUNETR /16 encoder for PatchChestCT patch supervision.

This adapter deliberately stops the MONAI SwinUNETR-v2 encoder at the /16
stage.  A 96 x 192 x 192 PatchChestCT crop therefore produces the native
6 x 12 x 12 grid used by the aligned Table-2 protocol.  The segmentation
decoder, feature pyramid, interpolation, and case-specific auxiliary heads
are intentionally absent; the only task-specific module is a linear
18-channel patch classifier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep
from torch import nn


VOCO10K_EXPECTED_SHA256 = "de4160dc52436e2e437a4c0293822e5fd644f4858738aa14995866a2617fedcd"
VOCO10K_EXPECTED_BYTES = 220_245_413
VOCO10K_MIRROR_REVISION = "825c1d474f429322a3fed7abebfc3978516006c2"
VOCO10K_MIRROR_FILE_COMMIT = "112b8d9a1c53b92c6882f017306334fc63567b2b"
VOCO10K_MIRROR_URL = (
    "https://huggingface.co/jethro682/pretrain_mri/resolve/"
    f"{VOCO10K_MIRROR_REVISION}/VoCo_10k.pt?download=true"
)
VOCO10K_HISTORICAL_OFFICIAL_URL = (
    "https://www.dropbox.com/scl/fi/35ldfszlvw1ke4vr7xr5h/VoCo_10k.pt"
    "?rlkey=iu3muui9420soyjwlui79njmq&dl=0"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_state_dict(raw: object) -> Mapping[str, torch.Tensor]:
    if isinstance(raw, Mapping) and raw and all(torch.is_tensor(value) for value in raw.values()):
        return raw  # official VoCo_10k.pt is a bare OrderedDict
    if isinstance(raw, Mapping):
        for key in ("state_dict", "network_weights", "net", "student", "model"):
            candidate = raw.get(key)
            if isinstance(candidate, Mapping) and candidate and all(
                torch.is_tensor(value) for value in candidate.values()
            ):
                return candidate
    raise RuntimeError("VoCo checkpoint does not contain a recognized tensor state_dict")


def _normalize_key(key: str) -> str:
    for prefix in ("module.", "student.", "backbone."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key.replace("swin_vit", "swinViT")


class VoCo10KSwinUNETRPatchClassifier(nn.Module):
    """Fine-tune the VoCo-10K SwinUNETR-v2 /16 encoder with a linear patch head."""

    output_shape = (6, 12, 12)
    feature_channels = 384

    def __init__(
        self,
        num_classes: int = 18,
        pretrained: bool = True,
        pretrained_checkpoint: str | Path | None = None,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if not pretrained:
            raise ValueError("This Table-2 adapter requires the verified VoCo-10K initialization")
        if pretrained_checkpoint is None:
            raise ValueError("pretrained_checkpoint is required for VoCo-10K")

        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.backbone = SwinTransformer(
            in_chans=1,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            downsample="merging",
            use_v2=True,
        )

        checkpoint_path = Path(pretrained_checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
        if checkpoint_bytes != VOCO10K_EXPECTED_BYTES:
            raise RuntimeError(
                f"Unexpected VoCo-10K checkpoint size: {checkpoint_bytes}; "
                f"expected {VOCO10K_EXPECTED_BYTES}"
            )
        checkpoint_sha256 = _sha256(checkpoint_path)
        if checkpoint_sha256 != VOCO10K_EXPECTED_SHA256:
            raise RuntimeError(
                f"Unexpected VoCo-10K SHA256: {checkpoint_sha256}; "
                f"expected {VOCO10K_EXPECTED_SHA256}"
            )

        raw = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        source_state = {
            _normalize_key(str(key)): value
            for key, value in _unwrap_state_dict(raw).items()
        }
        expected_state = self.backbone.state_dict()
        encoder_state: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        shape_mismatches: list[dict[str, Any]] = []
        for target_key, target_value in expected_state.items():
            source_key = f"swinViT.{target_key}"
            source_value = source_state.get(source_key)
            if source_value is None:
                missing.append(target_key)
                continue
            if tuple(source_value.shape) != tuple(target_value.shape):
                shape_mismatches.append(
                    {
                        "key": target_key,
                        "checkpoint": list(source_value.shape),
                        "model": list(target_value.shape),
                    }
                )
                continue
            encoder_state[target_key] = source_value
        if missing or shape_mismatches or len(encoder_state) != len(expected_state):
            raise RuntimeError(
                "VoCo-10K does not exactly cover the MONAI SwinUNETR encoder: "
                f"missing={missing}, shape_mismatches={shape_mismatches}, "
                f"loaded={len(encoder_state)}/{len(expected_state)}"
            )
        self.backbone.load_state_dict(encoder_state, strict=True)

        # The /32 stage is not part of the registered model because the aligned
        # PatchChestCT target is natively /16.  Removing it avoids counting or
        # optimizing parameters that cannot influence the prediction.
        del self.backbone.layers4
        del self.backbone.layers4c

        self.classifier = nn.Linear(self.feature_channels, num_classes)
        used_state = self.backbone.state_dict()
        self.pretraining_report: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_sha256,
            "safe_load": "torch.load(weights_only=True, mmap=True)",
            "source": "public mirror of authors' formerly released VoCo_10k.pt",
            "mirror_url": VOCO10K_MIRROR_URL,
            "mirror_revision": VOCO10K_MIRROR_REVISION,
            "mirror_file_commit": VOCO10K_MIRROR_FILE_COMMIT,
            "historical_official_url": VOCO10K_HISTORICAL_OFFICIAL_URL,
            "full_swinvit_loaded_tensors": len(encoder_state),
            "full_swinvit_expected_tensors": len(expected_state),
            "registered_used_encoder_tensors": len(used_state),
            "registered_used_encoder_numel": sum(value.numel() for value in used_state.values()),
            "checkpoint_tensor_keys_unused_by_patch_adapter": sorted(
                key for key in source_state if not key.startswith("swinViT.")
            ),
            "removed_after_verified_load": ["layers4", "layers4c"],
            "feature_stage": "SwinUNETR-v2 /16 (hidden state x3)",
            "feature_shape": [self.feature_channels, *self.output_shape],
        }
        del raw, source_state, encoder_state, expected_state, used_state

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return B x 384 x 6 x 12 x 12 normalized /16 encoder features."""
        x0 = self.backbone.pos_drop(self.backbone.patch_embed(x))
        x0 = self.backbone.layers1c[0](x0.contiguous())
        x1 = self.backbone.layers1[0](x0.contiguous())
        x1 = self.backbone.layers2c[0](x1.contiguous())
        x2 = self.backbone.layers2[0](x1.contiguous())
        x2 = self.backbone.layers3c[0](x2.contiguous())
        x3 = self.backbone.layers3[0](x2.contiguous())
        features = self.backbone.proj_out(x3, normalize=True)
        if tuple(features.shape[1:]) != (self.feature_channels, *self.output_shape):
            raise RuntimeError(
                "VoCo /16 feature grid changed: "
                f"got {tuple(features.shape[1:])}, "
                f"expected {(self.feature_channels, *self.output_shape)}"
            )
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        features = features.permute(0, 2, 3, 4, 1).contiguous()
        logits = self.classifier(features)
        return logits.permute(0, 4, 1, 2, 3).contiguous()

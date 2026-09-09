#!/usr/bin/env python3
"""Train PatchChestCT fold-0 patch-level grounding baselines.

This is the single-entry patch-level counterpart of
`train_patchchestct_official_case_fold0.py`.

It keeps PatchChestCT's official fully supervised grounding setup while using
our explicit train/val/test split:

- train: model fitting only
- val: best checkpoint selection and case-threshold selection
- test: final reporting only

Patch-DSC follows PatchChestCT's official `evaluate_model()` convention:
for each class, evaluate only test cases with a positive manual patch
annotation for that class, sweep thresholds in [0, 1], and report the
class-wise best Dice/F1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from types import ModuleType
from typing import Any

import numpy as np

# Configure CUDA BLAS before the first handle can be created. Strict
# deterministic runs require this workspace setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classification_code.patchchestct_video_models.train import (  # noqa: E402
    evidence_alignment_loss,
    token_consistency_loss,
)
from classification_code.patchchestct_grid import (  # noqa: E402
    ANATOMICAL_GRID_V2,
    LEGACY_OFFICIAL_GRID,
    PATCH_GRID_PROTOCOLS,
    reduce_patch_target_24_to_6,
)
from classification_code.patchchestct_pooling import (  # noqa: E402
    smooth_logmeanexp_pool3d,
)

PATCHCHESTCT_ROOT = REPO_ROOT / "nnunet_data" / "Bronchidata" / "PatchChestCT"
DEFAULT_SPLITS_DIR = PATCHCHESTCT_ROOT / "manifests" / "cv5_validtest_seed2026_spacing1p5_1p5_3p0"
DEFAULT_OUTPUT_ROOT = PATCHCHESTCT_ROOT / "patch_official_spacing1p5_1p5_3p0_fold0_runs"

SELECTED_IDX = [1, 3, 4, 5, 6, 8, 10, 15, 16]
CLASSES = [
    "arterial_wall_calcification",
    "pericardial_effusion",
    "coronary_wall_calcification",
    "hiatal_hernia",
    "lymphadenopathy",
    "atelectasis",
    "lung_opacity",
    "consolidation",
    "bronchiectasis",
]
VJEPA_MULTI_WINDOWS = (
    (-1000.0, 200.0),
    (-160.0, 240.0),
    (-100.0, 900.0),
)
PATCH_LOCALIZATION_MODES = (
    "linear",
    "mct-attention",
    "mct-patchcam",
    "mct-fused",
    "mct-affinity-fused",
    "mct-lam",
    "mct-lam-fused",
)


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    display_name: str
    model_file: Path
    class_name: str
    batch_size: int
    optimizer: str
    pretraining: str
    init_description: str


BACKBONES = {
    "r3d18": BackboneSpec(
        key="r3d18",
        display_name="R3D-18",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_r3d18" / "model.py",
        class_name="R3D18PatchClassifier",
        batch_size=10,
        optimizer="adamw",
        pretraining="Torchvision Kinetics-400",
        init_description=(
            "torchvision.models.video.r3d_18 with R3D_18_Weights.DEFAULT; "
            "first conv replaced for 1-channel CT; patch classifier newly initialized"
        ),
    ),
    "swin3d_t": BackboneSpec(
        key="swin3d_t",
        display_name="Swin3D-T",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_swin3d_t" / "model.py",
        class_name="Swin3DTPatchClassifier",
        batch_size=6,
        optimizer="sgd",
        pretraining="Torchvision Kinetics-400",
        init_description=(
            "torchvision.models.video.swin3d_t with Swin3D_T_Weights.DEFAULT; "
            "tubelet projection replaced for 1-channel CT; patch classifier newly initialized"
        ),
    ),
    "mvit_v2_s": BackboneSpec(
        key="mvit_v2_s",
        display_name="MViT-v2-S",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_mvit_v2_s" / "model.py",
        class_name="MViTV2SPatchClassifier",
        batch_size=6,
        optimizer="sgd",
        pretraining="None",
        init_description=(
            "PatchChestCT custom MViT-v2-S block setting; no explicit torchvision pretrained "
            "weights are loaded; 1-channel conv and patch classifier newly initialized"
        ),
    ),
    "vjepa2_1_b": BackboneSpec(
        key="vjepa2_1_b",
        display_name="V-JEPA 2.1 ViT-B/16",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_vjepa2_1" / "model.py",
        class_name="VJEPA21OfficialPatchClassifier",
        batch_size=2,
        optimizer="adamw",
        pretraining="V-JEPA 2.1 official ViT-B/16 384px",
        init_description=(
            "facebookresearch/vjepa2 vjepa2_1_vit_base_384; official CT crop resized to "
            "64x384x384; dense tokens pooled to the official 6x12x12 patch grid"
        ),
    ),
    "voco10k_swinunetr": BackboneSpec(
        key="voco10k_swinunetr",
        display_name="VoCo-10K SwinUNETR /16",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_voco10k" / "model.py",
        class_name="VoCo10KSwinUNETRPatchClassifier",
        batch_size=1,
        optimizer="adamw",
        pretraining="VoCo-10K self-supervised CT pretraining (CVPR 2024)",
        init_description=(
            "MONAI SwinUNETR-v2 feature_size=48 initialized from the verified "
            "VoCo_10k.pt mirror; /16 encoder stage gives a native 6x12x12 grid; "
            "linear patch classifier newly initialized; no decoder or FPN"
        ),
    ),
}


def normalize_backbone(name: str) -> str:
    return "mvit_v2_s" if name == "mvit" else name


def parse_shape(values: list[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("shape must contain D H W")
    return int(values[0]), int(values[1]), int(values[2])


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    base_path = base_dir / path
    if base_path.exists():
        return base_path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return cwd_path


def center_crop_or_pad_3d(arr: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    out = arr
    for axis, tgt in enumerate(target):
        cur = out.shape[axis]
        if cur >= tgt:
            start = (cur - tgt) // 2
            slices = [slice(None)] * out.ndim
            slices[axis] = slice(start, start + tgt)
            out = out[tuple(slices)]
        else:
            pad = [(0, 0)] * out.ndim
            before = (tgt - cur) // 2
            after = tgt - cur - before
            pad[axis] = (before, after)
            out = np.pad(out, pad, mode="constant", constant_values=0)
    return out


def center_crop_or_pad_cd_hw(arr: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    out = arr
    for spatial_axis, tgt in enumerate(target, start=1):
        cur = out.shape[spatial_axis]
        if cur >= tgt:
            start = (cur - tgt) // 2
            slices = [slice(None)] * out.ndim
            slices[spatial_axis] = slice(start, start + tgt)
            out = out[tuple(slices)]
        else:
            pad = [(0, 0)] * out.ndim
            before = (tgt - cur) // 2
            after = tgt - cur - before
            pad[spatial_axis] = (before, after)
            out = np.pad(out, pad, mode="constant", constant_values=0)
    return out


def crop_starts(source: tuple[int, int, int], target: tuple[int, int, int], random_crop: bool) -> tuple[int, int, int]:
    starts: list[int] = []
    for axis, (src, tgt) in enumerate(zip(source, target)):
        if src < tgt:
            raise ValueError(f"Cannot crop axis {axis} from {src} to {tgt}")
        if random_crop and src > tgt:
            starts.append(random.randint(0, src - tgt))
        else:
            starts.append((src - tgt) // 2)
    return int(starts[0]), int(starts[1]), int(starts[2])


def crop_3d(arr: np.ndarray, target: tuple[int, int, int], starts: tuple[int, int, int]) -> np.ndarray:
    d0, h0, w0 = starts
    d, h, w = target
    return arr[d0 : d0 + d, h0 : h0 + h, w0 : w0 + w]


def crop_cd_hw(arr: np.ndarray, target: tuple[int, int, int], starts: tuple[int, int, int]) -> np.ndarray:
    d0, h0, w0 = starts
    d, h, w = target
    return arr[:, d0 : d0 + d, h0 : h0 + h, w0 : w0 + w]


def load_ct_official_input(
    image_path: Path,
    pad_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    starts: tuple[int, int, int],
    clip_hu: tuple[float, float],
    input_mode: str = "grayscale",
) -> np.ndarray:
    with np.load(image_path) as data:
        arr = np.asarray(data["arr_0"], dtype=np.float32) * 1000.0
    if arr.ndim != 3:
        raise ValueError(f"{image_path} arr_0 has shape {arr.shape}; expected 3D")

    windows = (clip_hu,) if input_mode == "grayscale" else VJEPA_MULTI_WINDOWS
    channels: list[np.ndarray] = []
    for hu_min, hu_max in windows:
        channel = np.clip(arr, hu_min, hu_max)
        channel = (channel - hu_min) / (hu_max - hu_min)
        channel = center_crop_or_pad_3d(channel, pad_shape)
        channel = np.rot90(channel, k=-1, axes=(1, 2))
        channel = np.flip(channel, axis=2).copy()
        channel = crop_3d(channel, crop_shape, starts)
        channels.append(channel.astype(np.float32, copy=False))
    if input_mode == "grayscale":
        return channels[0]
    if input_mode == "multi-window":
        return np.stack(channels, axis=0)
    raise ValueError(f"Unsupported input mode {input_mode!r}")


def load_high_res_annotation_mask(annotation_dir: Path, classes: list[str]) -> np.ndarray:
    masks: list[np.ndarray] = []
    for class_name in classes:
        path = annotation_dir / f"{class_name}.npz"
        if path.exists():
            with np.load(path) as data:
                ann = np.asarray(data["arr_0"], dtype=np.float32)
        else:
            ann = np.zeros((24, 12, 12), dtype=np.float32)
        if ann.shape != (24, 12, 12):
            raise ValueError(f"{path} has shape {ann.shape}; expected (24, 12, 12)")
        ann = (ann > 0).astype(np.float32, copy=False)
        ann = np.repeat(ann, 4, axis=0)
        ann = np.repeat(ann, 16, axis=1)
        ann = np.repeat(ann, 16, axis=2)
        masks.append(ann)
    return np.stack(masks, axis=0)


def load_patch_target_24(
    annotation_dir: Path,
    classes: list[str],
    pad_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    starts: tuple[int, int, int],
) -> np.ndarray:
    mask = load_high_res_annotation_mask(annotation_dir, classes)
    mask = center_crop_or_pad_cd_hw(mask, pad_shape)
    mask = crop_cd_hw(mask, crop_shape, starts)
    if tuple(mask.shape[1:]) != crop_shape:
        raise ValueError(f"Annotation crop has shape {mask.shape[1:]}; expected {crop_shape}")
    return mask[:, 2::4, 8::16, 8::16].astype(np.float32, copy=False)


def official_reduce_patch_target_24_to_6(target24: torch.Tensor) -> torch.Tensor:
    """Backward-compatible entry point for the historical official grid."""

    return reduce_patch_target_24_to_6(target24, protocol=LEGACY_OFFICIAL_GRID)


class PatchChestCTPatchDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        classes: list[str],
        pad_shape: tuple[int, int, int],
        crop_shape: tuple[int, int, int],
        clip_hu: tuple[float, float],
        random_crop: bool,
        input_mode: str = "grayscale",
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.base_dir = self.manifest_csv.parent
        self.classes = classes
        self.pad_shape = pad_shape
        self.crop_shape = crop_shape
        self.clip_hu = clip_hu
        self.random_crop = random_crop
        self.input_mode = input_mode
        with self.manifest_csv.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise RuntimeError(f"No rows found in {self.manifest_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_path = resolve_path(row["image_path"], self.base_dir)
        annotation_dir = resolve_path(row["annotation_dir"], self.base_dir)
        starts = crop_starts(self.pad_shape, self.crop_shape, self.random_crop)
        labels = [float(row[f"{class_name}_label"]) for class_name in self.classes]
        image = load_ct_official_input(
            image_path=image_path,
            pad_shape=self.pad_shape,
            crop_shape=self.crop_shape,
            starts=starts,
            clip_hu=self.clip_hu,
            input_mode=self.input_mode,
        )
        patch_target24 = load_patch_target_24(
            annotation_dir=annotation_dir,
            classes=self.classes,
            pad_shape=self.pad_shape,
            crop_shape=self.crop_shape,
            starts=starts,
        )
        return {
            "image": torch.from_numpy(image[None] if image.ndim == 3 else image),
            "case_target": torch.tensor(labels, dtype=torch.float32),
            "patch_target_24": torch.from_numpy(patch_target24),
            "crop_starts": torch.tensor(starts, dtype=torch.float32),
            "volume_id": row["volume_id"],
        }


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(
    spec: BackboneSpec,
    crop_shape: tuple[int, int, int],
    num_output_classes: int,
    args: argparse.Namespace | None = None,
) -> nn.Module:
    module = load_module(spec.model_file, f"patchchestct_{spec.key}_model_for_official_patch")
    model_cls = getattr(module, spec.class_name)
    if spec.key == "mvit_v2_s":
        return model_cls(
            num_classes=num_output_classes,
            input_shape=crop_shape,
            pretrained=False,
            deterministic_max_pool=bool(args is not None and args.deterministic),
        )
    if spec.key == "vjepa2_1_b" and args is not None:
        return model_cls(
            num_classes=num_output_classes,
            pretrained=True,
            mil_head=args.mil_head,
            class_token_heads=args.class_token_heads,
            class_token_dropout=args.class_token_dropout,
            class_token_decoder_depth=args.class_token_decoder_depth,
            class_token_coord_embedding=args.class_token_coord_embedding,
            class_token_coord_mode=args.class_token_coord_mode,
            class_token_anatomical_prior=args.class_token_anatomical_prior,
            class_token_prior_gamma=args.class_token_prior_gamma,
            class_token_prior_grid=args.class_token_prior_grid,
            class_token_residual_init=args.class_token_residual_init,
            anatomical_evidence=args.anatomical_evidence,
            anatomical_evidence_hidden_dim=args.anatomical_evidence_hidden_dim,
            anatomical_evidence_gate_init=args.anatomical_evidence_gate_init,
            anatomical_pad_shape=args.pad_shape,
            anatomical_crop_shape=args.crop_shape,
            global_local_fusion=args.global_local_fusion,
            local_case_pooling=args.local_case_pooling,
            local_case_topk=args.local_case_topk,
            adaptive_pool_topks=args.adaptive_pool_topks,
            adaptive_pool_init_weights=args.adaptive_pool_init_weights,
            gwrp_decay=args.gwrp_decay,
            fusion_local_init=args.fusion_local_init,
            mct_attention_residual_max=args.mct_attention_residual_max,
            debug_breakpoints=args.debug_breakpoints,
            deterministic_adaptive_pool=args.deterministic,
            patch_grid_protocol=args.patch_grid_protocol,
            patch_token_pooling=args.patch_token_pooling,
            smooth_or_temperature=args.smooth_or_temperature,
            fine_annotation_supervision=args.fine_annotation_supervision_weight > 0.0,
            fine_annotation_shape=(24, 12, 12),
        )
    if spec.key == "voco10k_swinunetr" and args is not None:
        return model_cls(
            num_classes=num_output_classes,
            pretrained=True,
            pretrained_checkpoint=args.voco_pretrained_checkpoint,
            use_checkpoint=True,
        )
    return model_cls(num_classes=num_output_classes, pretrained=True)


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def deterministic_runtime_state() -> dict[str, Any]:
    return {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def source_provenance(spec: BackboneSpec) -> dict[str, Any]:
    video_model_root = REPO_ROOT / "classification_code" / "patchchestct_video_models"
    source_files = [
        Path(__file__).resolve(),
        spec.model_file.resolve(),
        (REPO_ROOT / "classification_code" / "patchchestct_grid.py").resolve(),
        (REPO_ROOT / "classification_code" / "patchchestct_pooling.py").resolve(),
        (video_model_root / "train.py").resolve(),
        (video_model_root / "dataset.py").resolve(),
        (video_model_root / "metrics.py").resolve(),
    ]
    if spec.key == "vjepa2_1_b":
        source_files.extend(
            (
                (video_model_root / "models.py").resolve(),
                (REPO_ROOT / "classification_code" / "deterministic_ops.py").resolve(),
            )
        )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def dice_loss(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(0, 2, 3, 4))
    union = preds.sum(dim=(0, 2, 3, 4)) + targets.sum(dim=(0, 2, 3, 4))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def asymmetric_focal_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    gamma_positive: float,
    gamma_negative: float,
) -> torch.Tensor:
    probs = probs.float().clamp(1e-6, 1.0 - 1e-6)
    targets = targets.float()
    positive = -targets * (1.0 - probs).pow(gamma_positive) * probs.log()
    negative = -(1.0 - targets) * probs.pow(gamma_negative) * torch.log1p(-probs)
    return (positive + negative).mean()


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    preds = [int(score >= threshold) for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2.0 * precision * sensitivity, precision + sensitivity)
    return {
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "precision": precision,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def roc_auc(labels: list[int], scores: list[float]) -> float:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: list[int], scores: list[float]) -> float:
    n_pos = sum(labels)
    if n_pos == 0:
        return math.nan
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for _, label in pairs[i:j]:
            if label:
                tp += 1
            else:
                fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def choose_threshold(labels: list[int], scores: list[float], objective: str) -> tuple[float, dict[str, float]]:
    if not scores or sum(labels) == 0 or sum(labels) == len(labels):
        return 0.5, binary_metrics(labels, scores, 0.5)
    eps = 1e-7
    unique_scores = sorted(set(scores))
    candidates = [unique_scores[0] - eps, *unique_scores, unique_scores[-1] + eps, 0.5]
    best_threshold = 0.5
    best_metrics = binary_metrics(labels, scores, best_threshold)
    best_key = (-1.0, -1.0, -abs(best_threshold - 0.5))
    for threshold in candidates:
        metrics = binary_metrics(labels, scores, threshold)
        if objective == "f1":
            primary = metrics["f1"]
        elif objective == "balanced_accuracy":
            primary = metrics["balanced_accuracy"]
        elif objective == "youden":
            primary = metrics["sensitivity"] + metrics["specificity"] - 1.0
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")
        key = (primary, metrics["balanced_accuracy"], -abs(threshold - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def best_dice(labels: list[int], scores: list[float]) -> tuple[float, float, dict[str, float]]:
    if not labels:
        return math.nan, math.nan, {"tp": math.nan, "fp": math.nan, "fn": math.nan, "tn": math.nan}
    best_score = -1.0
    best_threshold = 0.5
    best_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    for threshold in np.linspace(0.0, 1.0, 101):
        preds = [int(score > threshold) for score in scores]
        tp = sum(1 for y, p in zip(labels, preds) if y and p)
        fp = sum(1 for y, p in zip(labels, preds) if not y and p)
        fn = sum(1 for y, p in zip(labels, preds) if y and not p)
        tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
        dsc = safe_div(2.0 * tp, 2.0 * tp + fp + fn)
        if dsc > best_score:
            best_score = dsc
            best_threshold = float(threshold)
            best_counts = {"tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}
    return best_score, best_threshold, best_counts


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean_or_nan(values: list[float]) -> float:
    usable = finite(values)
    return mean(usable) if usable else math.nan


def mean_std_percent(values: list[float]) -> str:
    usable = finite(values)
    if not usable:
        return "NA"
    scaled = [value * 100.0 for value in usable]
    return f"{mean(scaled):.2f} \u00b1 {pstdev(scaled):.2f}"


def fmt_float(value: float, places: int = 6) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{places}f}"


def get_volume_ids(batch_volume_id: object) -> list[str]:
    if isinstance(batch_volume_id, str):
        return [batch_volume_id]
    return [str(value) for value in batch_volume_id]


@dataclass
class PredictionBundle:
    loss: float
    total_loss: float
    fine_loss: float
    consistency_loss: float
    volume_ids: list[str]
    case_probabilities: np.ndarray
    case_targets: np.ndarray
    patch_scores: list[list[float]]
    patch_targets: list[list[int]]
    patch_positive_cases: list[int]


@dataclass(frozen=True)
class EpochLosses:
    total: float
    coarse: float
    fine: float
    consistency: float


def patch_loss(
    logits: torch.Tensor,
    patch_target24: torch.Tensor,
    selected_idx: list[int],
    criterion: nn.Module,
    dice_weight: float,
    loss_name: str,
    focal_gamma_positive: float,
    focal_gamma_negative: float,
    patch_grid_protocol: str = LEGACY_OFFICIAL_GRID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = reduce_patch_target_24_to_6(
        patch_target24,
        protocol=patch_grid_protocol,
    )
    probs = logits[:, selected_idx].sigmoid()
    if loss_name == "bce":
        loss_ce = criterion(probs.float(), targets.float())
    elif loss_name == "asymmetric_focal":
        loss_ce = asymmetric_focal_loss(
            probs,
            targets,
            gamma_positive=focal_gamma_positive,
            gamma_negative=focal_gamma_negative,
        )
    else:
        raise ValueError(f"Unsupported patch loss {loss_name!r}")
    loss_dice = dice_loss(probs, targets)
    return loss_ce + dice_weight * loss_dice, probs, targets


def fine_annotation_losses(
    outputs: dict[str, torch.Tensor],
    coarse_logits: torch.Tensor,
    patch_target24: torch.Tensor,
    selected_idx: list[int],
    dice_weight: float,
    loss_name: str,
    focal_gamma_positive: float,
    focal_gamma_negative: float,
    smooth_or_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fine-grid grounding and fine-to-coarse probability consistency."""

    fine_logits = outputs.get("fine_patch_logits")
    if fine_logits is None:
        raise RuntimeError(
            "Fine annotation supervision requires model outputs['fine_patch_logits']"
        )
    expected_shape = (
        patch_target24.shape[0],
        coarse_logits.shape[1],
        *patch_target24.shape[2:],
    )
    if tuple(fine_logits.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"Fine logits have shape {tuple(fine_logits.shape)}; expected {expected_shape}"
        )
    selected_fine_logits = fine_logits[:, selected_idx].float()
    fine_targets = patch_target24.float()
    fine_probs = selected_fine_logits.sigmoid()
    if loss_name == "bce":
        fine_ce = F.binary_cross_entropy_with_logits(
            selected_fine_logits,
            fine_targets,
        )
    elif loss_name == "asymmetric_focal":
        fine_ce = asymmetric_focal_loss(
            fine_probs,
            fine_targets,
            gamma_positive=focal_gamma_positive,
            gamma_negative=focal_gamma_negative,
        )
    else:
        raise ValueError(f"Unsupported patch loss {loss_name!r}")
    fine_loss = fine_ce + dice_weight * dice_loss(fine_probs, fine_targets)

    fine_to_coarse_logits = smooth_logmeanexp_pool3d(
        selected_fine_logits,
        output_shape=(6, 12, 12),
        temperature=smooth_or_temperature,
    )
    coarse_teacher = coarse_logits[:, selected_idx].float().sigmoid().detach()
    consistency_loss = F.mse_loss(
        fine_to_coarse_logits.sigmoid(),
        coarse_teacher,
    )
    return fine_loss, consistency_loss


def add_case_supervision_loss(
    loss: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    case_targets: torch.Tensor,
    selected_idx: list[int],
    weight: float,
    pos_weight: torch.Tensor | None,
    logit_key: str = "case_logits",
) -> torch.Tensor:
    if weight <= 0.0:
        return loss
    case_logits = outputs.get(logit_key)
    if case_logits is None:
        raise RuntimeError(f"Case supervision requires model outputs[{logit_key!r}]")
    case_loss = F.binary_cross_entropy_with_logits(
        case_logits[:, selected_idx].float(),
        case_targets.float(),
        pos_weight=pos_weight,
    )
    return loss + weight * case_loss


def unpack_patch_outputs(
    outputs: torch.Tensor | dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(outputs, torch.Tensor):
        return outputs, {}
    logits = outputs.get("patch_logits")
    if logits is None:
        raise RuntimeError("Model output dictionary does not contain patch_logits")
    return logits, outputs


def _normalize_localization_map(values: torch.Tensor) -> torch.Tensor:
    """Per-case/class min-max normalization used for CAM-style maps."""
    flat = values.flatten(2)
    minimum = flat.amin(dim=-1, keepdim=True)
    maximum = flat.amax(dim=-1, keepdim=True)
    normalized = (flat - minimum) / (maximum - minimum).clamp_min(1e-8)
    return normalized.reshape_as(values)


def mct_localization_scores(
    logits: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    selected_idx: list[int],
    mode: str,
) -> torch.Tensor:
    """Return patch scores for linear or MCT-style localization inference.

    ``mct-fused`` follows MCTformer+'s class-attention × PatchCAM fusion.
    ``mct-lam-fused`` uses the MoRe-style cosine class-patch relation map
    (LAM) in place of PatchCAM, so training and inference use the same
    class-token relation.
    ``mct-affinity-fused`` additionally propagates class attention through a
    dense-token cosine affinity matrix, the 3-D analogue of PatchAffinity.
    The V-JEPA encoder does not expose its internal block attention, so this
    token affinity is explicitly an approximation rather than a claim of
    recovering unavailable transformer attention weights.
    """
    if mode == "linear":
        return logits[:, selected_idx].sigmoid()
    if mode not in PATCH_LOCALIZATION_MODES:
        raise ValueError(f"Unknown patch localization mode: {mode!r}")
    attention = outputs.get("evidence_attention")
    class_token_logits = outputs.get("class_token_patch_logits")
    if attention is None or class_token_logits is None:
        raise RuntimeError("MCT localization requires evidence_attention and class_token_patch_logits")
    attention = attention[:, selected_idx].float()
    patchcam = _normalize_localization_map(
        F.relu(class_token_logits[:, selected_idx].float())
    )
    if mode == "mct-patchcam":
        return patchcam
    if mode == "mct-attention":
        return _normalize_localization_map(attention)
    if mode in {"mct-lam", "mct-lam-fused"}:
        relation_scores = outputs.get("class_patch_relation_scores")
        if relation_scores is None:
            raise RuntimeError("MoRe LAM localization requires class_patch_relation_scores")
        relation_map = _normalize_localization_map(
            relation_scores[:, selected_idx].float()
        )
        if mode == "mct-lam":
            return relation_map
        fused = torch.sqrt(
            _normalize_localization_map(attention).clamp_min(0.0)
            * relation_map.clamp_min(0.0)
        )
        return _normalize_localization_map(fused)
    attention = attention.flatten(2)
    if mode == "mct-affinity-fused":
        patch_tokens = outputs.get("patch_token_features")
        if patch_tokens is None:
            raise RuntimeError("MCT PatchAffinity requires patch_token_features")
        normalized_tokens = F.normalize(patch_tokens.float(), dim=-1)
        affinity = torch.softmax(
            torch.matmul(normalized_tokens, normalized_tokens.transpose(-2, -1))
            * (normalized_tokens.shape[-1] ** -0.5),
            dim=-1,
        )
        attention = torch.matmul(affinity.unsqueeze(1), attention.unsqueeze(-1)).squeeze(-1)
    attention = attention.reshape_as(patchcam)
    fused = torch.sqrt(
        _normalize_localization_map(attention).clamp_min(0.0)
        * patchcam.clamp_min(0.0)
    )
    return _normalize_localization_map(fused)


def localization_informed_relation_loss(
    outputs: dict[str, torch.Tensor],
    patch_targets: torch.Tensor,
    selected_idx: list[int],
    temperature: float,
) -> torch.Tensor:
    """MoRe confident-relation loss adapted to the supervised 3-D grid.

    The official MoRe LIR mines confident regions from CAM pseudo masks.  Our
    protocol already supplies deterministic 6x12x12 masks, so those masks are
    the reliable relation targets directly.  Patch features are detached as in
    MoRe, keeping this auxiliary objective focused on disease class tokens.
    """
    class_token_features = outputs.get("class_token_features")
    patch_token_features = outputs.get("patch_token_features")
    if class_token_features is None or patch_token_features is None:
        raise RuntimeError(
            "Localization-informed regularization requires class-token and patch-token features"
        )
    class_tokens = F.normalize(
        class_token_features[:, selected_idx].float(),
        dim=-1,
    )
    patch_tokens = F.normalize(
        patch_token_features.float(),
        dim=-1,
    ).detach()
    relation_logits = torch.einsum(
        "bcd,bnd->bcn",
        class_tokens,
        patch_tokens,
    ) / temperature
    target_weights = patch_targets.float().flatten(2)
    positive_counts = target_weights.sum(dim=-1)
    valid_pairs = positive_counts > 0
    target_distribution = target_weights / positive_counts.unsqueeze(-1).clamp_min(1.0)
    per_pair_loss = -(
        target_distribution * F.log_softmax(relation_logits, dim=-1)
    ).sum(dim=-1)
    return (
        per_pair_loss[valid_pairs].mean()
        if valid_pairs.any()
        else relation_logits.sum() * 0.0
    )


def forward_patch_model(
    model: nn.Module,
    images: torch.Tensor,
    batch: dict[str, object],
    device: torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if not bool(getattr(model, "crop_aware_anatomical_evidence", False)):
        return model(images)
    crop_starts = batch.get("crop_starts")
    if not isinstance(crop_starts, torch.Tensor):
        raise RuntimeError("Crop-aware anatomical evidence requires tensor crop_starts")
    return model(images, crop_starts=crop_starts.to(device, non_blocking=True))


def add_innovation_losses(
    loss: torch.Tensor,
    logits: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    patch_targets: torch.Tensor,
    case_targets: torch.Tensor,
    selected_idx: list[int],
    token_consistency_weight: float,
    evidence_alignment_weight: float,
    evidence_alignment_loss_type: str,
    cct_weight: float,
    lir_weight: float,
    lir_temperature: float,
) -> torch.Tensor:
    if (
        token_consistency_weight <= 0.0
        and evidence_alignment_weight <= 0.0
        and cct_weight <= 0.0
        and lir_weight <= 0.0
    ):
        return loss
    if token_consistency_weight > 0.0 or evidence_alignment_weight > 0.0:
        evidence_attention = outputs.get("evidence_attention")
        if evidence_attention is None:
            raise RuntimeError("Attention innovation losses require an evidence_attention output")
        consistency_logits = outputs.get("class_token_patch_logits", logits)
        selected_logits = consistency_logits[:, selected_idx]
        selected_attention = evidence_attention[:, selected_idx]
        if token_consistency_weight > 0.0:
            loss = loss + token_consistency_weight * token_consistency_loss(
                selected_logits,
                selected_attention,
                case_targets,
            )
        if evidence_alignment_weight > 0.0:
            loss = loss + evidence_alignment_weight * evidence_alignment_loss(
                selected_attention,
                patch_targets,
                evidence_alignment_loss_type,
            )
    if cct_weight > 0.0:
        final_class_token_features = outputs.get("class_token_features")
        if final_class_token_features is None:
            raise RuntimeError("CCT loss requires a class_token_features output")
        class_token_features = outputs.get("class_token_features_layers")
        if class_token_features is None:
            class_token_features = final_class_token_features.unsqueeze(1)
        selected_features = F.normalize(
            class_token_features[:, :, selected_idx].float(),
            dim=-1,
        )
        similarity = torch.matmul(
            selected_features,
            selected_features.transpose(-2, -1),
        )
        identity_targets = torch.arange(
            len(selected_idx),
            device=similarity.device,
        ).view(1, 1, -1).expand(
            similarity.shape[0],
            similarity.shape[1],
            -1,
        )
        per_anchor = F.cross_entropy(
            similarity.flatten(0, 1).transpose(1, 2),
            identity_targets.flatten(0, 1),
            reduction="none",
        ).reshape(similarity.shape[0], similarity.shape[1], -1)
        positive_anchors = case_targets.float().unsqueeze(1)
        positive_counts = case_targets.float().sum(dim=-1)
        # MCTformer+ first averages positive class anchors within each case.
        # PatchChestCT additionally contains negative-only cases, which have no
        # valid CCT anchor and are excluded from the final case average.
        per_case_cct = (
            (per_anchor * positive_anchors).sum(dim=-1)
            / (positive_counts.unsqueeze(1) + 1e-8)
        )
        valid_cases = positive_counts > 0
        cct_loss = (
            per_case_cct[valid_cases].mean()
            if valid_cases.any()
            else per_case_cct.sum() * 0.0
        )
        loss = loss + cct_weight * cct_loss
    if lir_weight > 0.0:
        loss = loss + lir_weight * localization_informed_relation_loss(
            outputs,
            patch_targets,
            selected_idx,
            lir_temperature,
        )
    return loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    selected_idx: list[int],
    patch_grid_protocol: str,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    dice_weight: float,
    patch_loss_name: str,
    focal_gamma_positive: float,
    focal_gamma_negative: float,
    fine_annotation_supervision_weight: float,
    fine_coarse_consistency_weight: float,
    smooth_or_temperature: float,
    case_loss_weight: float,
    local_case_loss_weight: float,
    class_token_case_loss_weight: float,
    case_pos_weight: torch.Tensor | None,
    token_consistency_weight: float,
    evidence_alignment_weight: float,
    evidence_alignment_loss_type: str,
    cct_weight: float,
    lir_weight: float,
    lir_temperature: float,
    gradient_accumulation_steps: int,
    max_batches: int | None,
    debug_breakpoints: bool,
) -> EpochLosses:
    model.train()
    total_loss = 0.0
    total_coarse_loss = 0.0
    total_fine_loss = 0.0
    total_consistency_loss = 0.0
    total_batches = 0
    accumulation_steps = max(int(gradient_accumulation_steps), 1)
    batches_to_run = min(len(loader), max_batches) if max_batches is not None else len(loader)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        case_targets = batch["case_target"].to(device, non_blocking=True)
        patch_target24 = batch["patch_target_24"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            raw_outputs = forward_patch_model(model, images, batch, device)
        logits, outputs = unpack_patch_outputs(raw_outputs)
        loss, patch_probs, patch_targets = patch_loss(
            logits.float(),
            patch_target24,
            selected_idx,
            criterion,
            dice_weight,
            patch_loss_name,
            focal_gamma_positive,
            focal_gamma_negative,
            patch_grid_protocol,
        )
        coarse_supervision_loss = loss
        fine_loss = loss.new_zeros(())
        consistency_loss = loss.new_zeros(())
        if fine_annotation_supervision_weight > 0.0:
            fine_loss, consistency_loss = fine_annotation_losses(
                outputs,
                logits,
                patch_target24,
                selected_idx,
                dice_weight,
                patch_loss_name,
                focal_gamma_positive,
                focal_gamma_negative,
                smooth_or_temperature,
            )
            loss = (
                loss
                + fine_annotation_supervision_weight * fine_loss
                + fine_coarse_consistency_weight * consistency_loss
            )
        patch_supervision_loss = loss.detach()
        loss = add_case_supervision_loss(
            loss,
            outputs,
            case_targets,
            selected_idx,
            case_loss_weight,
            case_pos_weight,
        )
        loss = add_case_supervision_loss(
            loss,
            outputs,
            case_targets,
            selected_idx,
            local_case_loss_weight,
            case_pos_weight,
            logit_key="local_case_logits",
        )
        loss = add_case_supervision_loss(
            loss,
            outputs,
            case_targets,
            selected_idx,
            class_token_case_loss_weight,
            case_pos_weight,
            logit_key="class_token_case_logits",
        )
        loss = add_innovation_losses(
            loss,
            logits,
            outputs,
            patch_targets,
            case_targets,
            selected_idx,
            token_consistency_weight,
            evidence_alignment_weight,
            evidence_alignment_loss_type,
            cct_weight,
            lir_weight,
            lir_temperature,
        )
        if debug_breakpoints and batch_index == 0:
            debug_loss_components = {
                "patch_supervision": float(patch_supervision_loss.item()),
                "case_and_innovation_addition": float((loss.detach() - patch_supervision_loss).item()),
                "total": float(loss.detach().item()),
            }
            print("[debug 4/4] Patch probabilities, targets, and loss components", flush=True)
            breakpoint()
        remainder = batches_to_run % accumulation_steps
        final_group_size = remainder if remainder else accumulation_steps
        group_size = final_group_size if batch_index >= batches_to_run - final_group_size else accumulation_steps
        scaled_loss = loss / group_size
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == batches_to_run
        if should_step:
            optimizer_was_run = True
            if use_amp:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_was_run = scaler.get_scale() >= scale_before
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if optimizer_was_run:
                scheduler.step()
        total_loss += float(loss.item())
        total_coarse_loss += float(coarse_supervision_loss.detach().item())
        total_fine_loss += float(fine_loss.detach().item())
        total_consistency_loss += float(consistency_loss.detach().item())
        total_batches += 1
    if total_batches == 0:
        raise RuntimeError("No training batches were produced")
    return EpochLosses(
        total=total_loss / total_batches,
        coarse=total_coarse_loss / total_batches,
        fine=total_fine_loss / total_batches,
        consistency=total_consistency_loss / total_batches,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    selected_idx: list[int],
    patch_grid_protocol: str,
    use_amp: bool,
    dice_weight: float,
    patch_loss_name: str,
    focal_gamma_positive: float,
    focal_gamma_negative: float,
    fine_annotation_supervision_weight: float,
    fine_coarse_consistency_weight: float,
    smooth_or_temperature: float,
    case_loss_weight: float,
    local_case_loss_weight: float,
    class_token_case_loss_weight: float,
    case_pos_weight: torch.Tensor | None,
    token_consistency_weight: float,
    evidence_alignment_weight: float,
    evidence_alignment_loss_type: str,
    cct_weight: float,
    lir_weight: float,
    lir_temperature: float,
    desc: str,
    max_batches: int | None,
    collect_patch: bool,
    patch_localization: str = "linear",
) -> PredictionBundle:
    model.eval()
    total_coarse_loss = 0.0
    total_objective_loss = 0.0
    total_fine_loss = 0.0
    total_consistency_loss = 0.0
    total_batches = 0
    all_ids: list[str] = []
    all_case_probs: list[np.ndarray] = []
    all_case_targets: list[np.ndarray] = []
    patch_scores: list[list[float]] = [[] for _ in CLASSES]
    patch_targets: list[list[int]] = [[] for _ in CLASSES]
    patch_positive_cases = [0 for _ in CLASSES]

    for batch_index, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        case_targets = batch["case_target"].to(device, non_blocking=True)
        patch_target24 = batch["patch_target_24"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            raw_outputs = forward_patch_model(model, images, batch, device)
        logits, outputs = unpack_patch_outputs(raw_outputs)
        loss, patch_probs, patch_target6 = patch_loss(
            logits.float(),
            patch_target24,
            selected_idx,
            criterion,
            dice_weight,
            patch_loss_name,
            focal_gamma_positive,
            focal_gamma_negative,
            patch_grid_protocol,
        )
        coarse_loss = loss
        fine_loss = loss.new_zeros(())
        consistency_loss = loss.new_zeros(())
        if fine_annotation_supervision_weight > 0.0:
            fine_loss, consistency_loss = fine_annotation_losses(
                outputs,
                logits,
                patch_target24,
                selected_idx,
                dice_weight,
                patch_loss_name,
                focal_gamma_positive,
                focal_gamma_negative,
                smooth_or_temperature,
            )
            loss = (
                loss
                + fine_annotation_supervision_weight * fine_loss
                + fine_coarse_consistency_weight * consistency_loss
            )
        loss = add_case_supervision_loss(
            loss,
            outputs,
            case_targets,
            selected_idx,
            case_loss_weight,
            case_pos_weight,
        )
        loss = add_case_supervision_loss(
            loss,
            outputs,
            case_targets,
            selected_idx,
            local_case_loss_weight,
            case_pos_weight,
            logit_key="local_case_logits",
        )
        loss = add_case_supervision_loss(
            loss,
            outputs,
            case_targets,
            selected_idx,
            class_token_case_loss_weight,
            case_pos_weight,
            logit_key="class_token_case_logits",
        )
        loss = add_innovation_losses(
            loss,
            logits,
            outputs,
            patch_target6,
            case_targets,
            selected_idx,
            token_consistency_weight,
            evidence_alignment_weight,
            evidence_alignment_loss_type,
            cct_weight,
            lir_weight,
            lir_temperature,
        )
        total_coarse_loss += float(coarse_loss.item())
        total_objective_loss += float(loss.item())
        total_fine_loss += float(fine_loss.item())
        total_consistency_loss += float(consistency_loss.item())
        total_batches += 1
        all_ids.extend(get_volume_ids(batch["volume_id"]))
        case_logits = outputs.get("case_logits")
        case_probs = (
            case_logits[:, selected_idx].sigmoid()
            if case_logits is not None
            else patch_probs.amax(dim=(2, 3, 4))
        )
        all_case_probs.append(case_probs.detach().cpu().float().numpy())
        all_case_targets.append(case_targets.detach().cpu().numpy())

        if collect_patch:
            localization_scores = mct_localization_scores(
                logits.float(), outputs, selected_idx, patch_localization
            )
            probs_np = localization_scores.detach().cpu().numpy()
            targets_np = patch_target6.detach().cpu().numpy().astype(np.uint8, copy=False)
            for b in range(probs_np.shape[0]):
                for c in range(len(CLASSES)):
                    if targets_np[b, c].sum() == 0:
                        continue
                    patch_positive_cases[c] += 1
                    patch_scores[c].extend(probs_np[b, c].reshape(-1).astype(float).tolist())
                    patch_targets[c].extend(targets_np[b, c].reshape(-1).astype(int).tolist())

    if total_batches == 0:
        raise RuntimeError(f"No {desc} batches were produced")
    return PredictionBundle(
        loss=total_coarse_loss / total_batches,
        total_loss=total_objective_loss / total_batches,
        fine_loss=total_fine_loss / total_batches,
        consistency_loss=total_consistency_loss / total_batches,
        volume_ids=all_ids,
        case_probabilities=np.concatenate(all_case_probs, axis=0),
        case_targets=np.concatenate(all_case_targets, axis=0),
        patch_scores=patch_scores,
        patch_targets=patch_targets,
        patch_positive_cases=patch_positive_cases,
    )


def select_thresholds(
    classes: list[str],
    val_bundle: PredictionBundle,
    objective: str,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    thresholds: dict[str, float] = {}
    rows: list[dict[str, float | int | str]] = []
    for class_index, class_name in enumerate(classes):
        labels = [int(value) for value in val_bundle.case_targets[:, class_index].tolist()]
        scores = [float(value) for value in val_bundle.case_probabilities[:, class_index].tolist()]
        threshold, metrics = choose_threshold(labels, scores, objective)
        thresholds[class_name] = threshold
        rows.append(
            {
                "class": class_name,
                "threshold": threshold,
                "val_positive_cases": sum(labels),
                "val_negative_cases": len(labels) - sum(labels),
                "val_f1": metrics["f1"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "selection_objective": objective,
            }
        )
    return thresholds, rows


def per_class_metrics(
    classes: list[str],
    bundle: PredictionBundle,
    thresholds: dict[str, float],
    compute_patch_dice: bool = True,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for class_index, class_name in enumerate(classes):
        labels = [int(value) for value in bundle.case_targets[:, class_index].tolist()]
        scores = [float(value) for value in bundle.case_probabilities[:, class_index].tolist()]
        threshold = thresholds[class_name]
        threshold_metrics = binary_metrics(labels, scores, threshold)
        patch_labels = bundle.patch_targets[class_index]
        patch_scores = bundle.patch_scores[class_index]
        patch_ap = average_precision(patch_labels, patch_scores) if patch_labels else math.nan
        if compute_patch_dice:
            patch_dsc, patch_threshold, patch_counts = best_dice(patch_labels, patch_scores)
        else:
            patch_dsc = math.nan
            patch_threshold = math.nan
            patch_counts = {"tp": math.nan, "fp": math.nan, "fn": math.nan, "tn": math.nan}
        rows.append(
            {
                "class": class_name,
                "num_cases": len(labels),
                "positive_cases": sum(labels),
                "negative_cases": len(labels) - sum(labels),
                "case_threshold": threshold,
                "auroc": roc_auc(labels, scores),
                "auprc": average_precision(labels, scores),
                "macro_f1": threshold_metrics["f1"],
                "bacc": threshold_metrics["balanced_accuracy"],
                "patch_auprc": patch_ap,
                "patch_dsc": patch_dsc,
                "patch_threshold": patch_threshold,
                "patch_positive_cases": bundle.patch_positive_cases[class_index],
                "patch_positive_cells": sum(patch_labels) if patch_labels else 0,
                "patch_tp": patch_counts["tp"],
                "patch_fp": patch_counts["fp"],
                "patch_fn": patch_counts["fn"],
                "patch_tn": patch_counts["tn"],
            }
        )
    return rows


def summarize_metrics(spec: BackboneSpec, rows: list[dict[str, float | int | str]]) -> dict[str, str]:
    metric_values = {
        "AUROC": [float(row["auroc"]) for row in rows],
        "AUPRC": [float(row["auprc"]) for row in rows],
        "Macro-F1": [float(row["macro_f1"]) for row in rows],
        "BACC": [float(row["bacc"]) for row in rows],
        "Patch-AUPRC": [float(row["patch_auprc"]) for row in rows],
        "Patch-DSC": [float(row["patch_dsc"]) for row in rows],
    }
    return {
        "Backbone": spec.display_name,
        "Pretraining": spec.pretraining,
        "Input type": "spacing 1.5/1.5/3.0 CT NPZ, official patch crop",
        "Supervision": "fully supervised patch-level grounding",
        "Mean/std scope": "over nine abnormalities",
        **{name: mean_std_percent(values) for name, values in metric_values.items()},
    }


def selection_value(
    metric_name: str,
    val_loss: float,
    rows_at_0p5: list[dict[str, float | int | str]],
) -> tuple[float, bool]:
    if metric_name == "val_loss":
        return val_loss, False
    if metric_name == "val_joint_auprc":
        case_auprc = mean_or_nan([float(row["auprc"]) for row in rows_at_0p5])
        patch_auprc = mean_or_nan([float(row["patch_auprc"]) for row in rows_at_0p5])
        if not math.isfinite(case_auprc) or not math.isfinite(patch_auprc):
            return math.nan, True
        return safe_div(2.0 * case_auprc * patch_auprc, case_auprc + patch_auprc), True
    lookup = {
        "val_macro_auroc": "auroc",
        "val_macro_auprc": "auprc",
        "val_macro_f1_0p5": "macro_f1",
        "val_macro_bacc_0p5": "bacc",
    }
    values = [float(row[lookup[metric_name]]) for row in rows_at_0p5]
    return mean_or_nan(values), True


def is_better(current: float, best: float | None, higher_is_better: bool) -> bool:
    if not math.isfinite(current):
        return False
    if best is None:
        return True
    return current > best if higher_is_better else current < best


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_adaptive_pool_weights(
    path: Path,
    model: nn.Module,
    selected_idx: list[int],
    classes: list[str],
) -> bool:
    pool_logits = getattr(model, "adaptive_pool_logits", None)
    pool_topks = getattr(model, "adaptive_pool_topks", None)
    if not isinstance(pool_logits, torch.Tensor) or pool_topks is None:
        return False
    weights = F.softmax(pool_logits.detach().float(), dim=-1).cpu().numpy()
    expert_names = ["mean" if int(topk) == 0 else ("max" if int(topk) == 1 else f"top-{int(topk)}") for topk in pool_topks]
    rows: list[dict[str, str]] = []
    for class_name, output_index in zip(classes, selected_idx):
        row = {"Class": class_name}
        row.update({name: fmt_float(float(weight), 8) for name, weight in zip(expert_names, weights[output_index])})
        rows.append(row)
    write_csv(path, rows)
    return True


def save_prediction_csv(path: Path, classes: list[str], bundle: PredictionBundle, thresholds: dict[str, float]) -> None:
    rows: list[dict[str, str]] = []
    for case_index, volume_id in enumerate(bundle.volume_ids):
        for class_index, class_name in enumerate(classes):
            probability = float(bundle.case_probabilities[case_index, class_index])
            threshold = thresholds[class_name]
            rows.append(
                {
                    "volume_id": volume_id,
                    "class": class_name,
                    "true_label": str(int(bundle.case_targets[case_index, class_index])),
                    "probability": fmt_float(probability, 8),
                    "threshold": fmt_float(threshold, 8),
                    "predicted_label": str(int(probability >= threshold)),
                }
            )
    write_csv(path, rows)


def save_threshold_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    csv_rows = [
        {
            "Class": str(row["class"]),
            "Case Threshold": fmt_float(float(row["threshold"])),
            "Val Positive Cases": str(row["val_positive_cases"]),
            "Val Negative Cases": str(row["val_negative_cases"]),
            "Val F1 (%)": fmt_float(float(row["val_f1"]) * 100.0, 2),
            "Val BACC (%)": fmt_float(float(row["val_balanced_accuracy"]) * 100.0, 2),
            "Selection Objective": str(row["selection_objective"]),
        }
        for row in rows
    ]
    write_csv(path, csv_rows)


def save_per_class_csv(path: Path, spec: BackboneSpec, rows: list[dict[str, float | int | str]]) -> None:
    csv_rows = [
        {
            "Backbone": spec.display_name,
            "Class": str(row["class"]),
            "N": str(row["num_cases"]),
            "Positive Cases": str(row["positive_cases"]),
            "Negative Cases": str(row["negative_cases"]),
            "Case Threshold": fmt_float(float(row["case_threshold"])),
            "AUROC (%)": fmt_float(float(row["auroc"]) * 100.0, 2),
            "AUPRC (%)": fmt_float(float(row["auprc"]) * 100.0, 2),
            "Macro-F1 (%)": fmt_float(float(row["macro_f1"]) * 100.0, 2),
            "BACC (%)": fmt_float(float(row["bacc"]) * 100.0, 2),
            "Patch-AUPRC (%)": fmt_float(float(row["patch_auprc"]) * 100.0, 2),
            "Patch-DSC (%)": fmt_float(float(row["patch_dsc"]) * 100.0, 2),
            "Patch Threshold": fmt_float(float(row["patch_threshold"])),
            "Patch Positive Cases": str(row["patch_positive_cases"]),
            "Patch Positive Cells": str(row["patch_positive_cells"]),
            "Patch TP": fmt_float(float(row["patch_tp"]), 0),
            "Patch FP": fmt_float(float(row["patch_fp"]), 0),
            "Patch FN": fmt_float(float(row["patch_fn"]), 0),
            "Patch TN": fmt_float(float(row["patch_tn"]), 0),
        }
        for row in rows
    ]
    write_csv(path, csv_rows)


def compute_case_pos_weight(
    dataset: PatchChestCTPatchDataset,
    mode: str,
    max_weight: float,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    positives = torch.tensor(
        [sum(float(row[f"{class_name}_label"]) > 0.0 for row in dataset.rows) for class_name in CLASSES],
        dtype=torch.float32,
    )
    negatives = float(len(dataset)) - positives
    ratio = negatives / positives.clamp_min(1.0)
    weights = ratio.sqrt() if mode == "sqrt" else ratio
    return weights.clamp(min=1.0, max=max_weight).to(device)


def make_optimizer(model: nn.Module, args: argparse.Namespace, spec: BackboneSpec) -> torch.optim.Optimizer:
    optimizer_name = args.optimizer or spec.optimizer
    if optimizer_name == "adamw":
        if spec.key == "vjepa2_1_b" and (args.head_lr_multiplier != 1.0 or args.adaptive_pool_lr is not None):
            head_prefixes = (
                "classifier.",
                "patch_head.",
                "class_token_case_head.",
                "class_token_residual_logit",
                "mct_attention_residual_gate",
                "anatomical_evidence.",
                "global_classifier.",
                "fusion_logit",
            )
            backbone_parameters: list[nn.Parameter] = []
            head_parameters: list[nn.Parameter] = []
            adaptive_pool_parameters: list[nn.Parameter] = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.startswith("adaptive_pool_logits"):
                    target = adaptive_pool_parameters
                else:
                    target = head_parameters if name.startswith(head_prefixes) else backbone_parameters
                target.append(parameter)
            parameter_groups = [
                {"params": backbone_parameters, "lr": args.lr},
                {"params": head_parameters, "lr": args.lr * args.head_lr_multiplier},
            ]
            if adaptive_pool_parameters:
                parameter_groups.append(
                    {
                        "params": adaptive_pool_parameters,
                        "lr": args.adaptive_pool_lr or args.lr * args.head_lr_multiplier,
                        "weight_decay": 0.0,
                    }
                )
            return torch.optim.AdamW(
                parameter_groups,
                weight_decay=args.weight_decay,
            )
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer {optimizer_name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume model, optimizer, scheduler, and log from this checkpoint.")
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize model weights only and start a fresh schedule.")
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Skip training and evaluate this checkpoint with val-selected thresholds.")
    parser.add_argument(
        "--voco-pretrained-checkpoint",
        type=Path,
        help="Verified VoCo_10k.pt used only by --backbone voco10k_swinunetr.",
    )
    parser.add_argument(
        "--backbone",
        choices=(
            "r3d18",
            "swin3d_t",
            "mvit",
            "mvit_v2_s",
            "vjepa2_1_b",
            "voco10k_swinunetr",
        ),
        default="r3d18",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"))
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--patch-loss", choices=("bce", "asymmetric_focal"), default="bce")
    parser.add_argument(
        "--patch-grid-protocol",
        choices=PATCH_GRID_PROTOCOLS,
        default=LEGACY_OFFICIAL_GRID,
        help=(
            "24-to-6 annotation reduction. The default exactly preserves the historical "
            "PatchChestCT modulo-six reshape; anatomical_grid_v2_6x12x12 pools six "
            "consecutive groups of four planes."
        ),
    )
    parser.add_argument(
        "--patch-token-pooling",
        choices=("mean", "smooth-or"),
        default="mean",
        help=(
            "Native-token aggregation into the 6x12x12 prediction grid. With the legacy "
            "protocol, mean preserves the historical adaptive-average path. With the v2 "
            "protocol, mean and smooth-or share mutually exclusive physical-center bins."
        ),
    )
    parser.add_argument(
        "--smooth-or-temperature",
        type=float,
        default=1.0,
        help="Positive LogMeanExp temperature used by --patch-token-pooling smooth-or.",
    )
    parser.add_argument(
        "--fine-annotation-supervision-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for auxiliary 24x12x12 annotation-center BCE+Dice supervision. "
            "The original 6x12x12 coarse validation loss remains the checkpoint metric."
        ),
    )
    parser.add_argument(
        "--fine-coarse-consistency-weight",
        type=float,
        default=0.0,
        help="Weight for fine-to-coarse SmoothOR probability consistency.",
    )
    parser.add_argument(
        "--fine-supervision-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp fine and consistency weights over this many training epochs.",
    )
    parser.add_argument("--focal-gamma-positive", type=float, default=0.0)
    parser.add_argument("--focal-gamma-negative", type=float, default=2.0)
    parser.add_argument(
        "--mil-head",
        choices=(
            "basic",
            "class-token",
            "hybrid-class-token",
            "decoupled-class-token",
            "mct-decoupled",
        ),
        default="basic",
    )
    parser.add_argument("--class-token-heads", type=int, default=8)
    parser.add_argument("--class-token-dropout", type=float, default=0.1)
    parser.add_argument("--class-token-decoder-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--class-token-coord-embedding", action="store_true")
    parser.add_argument("--class-token-coord-mode", choices=("add", "concat"), default="add")
    parser.add_argument("--class-token-anatomical-prior", action="store_true")
    parser.add_argument("--class-token-prior-gamma", type=float, default=1.0)
    parser.add_argument("--class-token-residual-init", type=float, default=0.1)
    parser.add_argument(
        "--class-token-prior-grid",
        nargs=3,
        type=int,
        default=[6, 12, 12],
        metavar=("D", "H", "W"),
    )
    parser.add_argument("--token-consistency-weight", type=float, default=0.0)
    parser.add_argument("--evidence-alignment-weight", type=float, default=0.0)
    parser.add_argument("--evidence-alignment-loss", choices=("kl", "bce"), default="kl")
    parser.add_argument("--cct-weight", type=float, default=0.0)
    parser.add_argument("--lir-weight", type=float, default=0.0)
    parser.add_argument("--lir-temperature", type=float, default=0.5)
    parser.add_argument("--mct-attention-residual-max", type=float, default=0.25)
    parser.add_argument("--anatomical-evidence", action="store_true")
    parser.add_argument("--anatomical-evidence-hidden-dim", type=int, default=64)
    parser.add_argument("--anatomical-evidence-gate-init", type=float, default=0.1)
    parser.add_argument("--global-local-fusion", action="store_true")
    parser.add_argument(
        "--local-case-pooling",
        choices=("max", "topk", "adaptive", "gwrp"),
        default="max",
    )
    parser.add_argument("--local-case-topk", type=int, default=4)
    parser.add_argument("--gwrp-decay", type=float, default=0.996)
    parser.add_argument(
        "--adaptive-pool-topks",
        nargs="+",
        type=int,
        default=[1, 4, 16, 0],
        metavar="K",
        help="Adaptive pooling experts; K=0 denotes global mean.",
    )
    parser.add_argument(
        "--adaptive-pool-init-weights",
        nargs="+",
        type=float,
        default=[0.05, 0.85, 0.08, 0.02],
        metavar="W",
    )
    parser.add_argument("--adaptive-pool-lr", type=float)
    parser.add_argument("--fusion-local-init", type=float, default=0.8)
    parser.add_argument("--case-loss-weight", type=float, default=0.0)
    parser.add_argument("--local-case-loss-weight", type=float, default=0.0)
    parser.add_argument("--class-token-case-loss-weight", type=float, default=0.0)
    parser.add_argument("--case-pos-weight", choices=("none", "sqrt", "ratio"), default="none")
    parser.add_argument("--max-case-pos-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value, e.g. --gpu 0")
    parser.add_argument("--device", help="Torch device. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-output-classes", type=int, default=18)
    parser.add_argument("--pad-shape", nargs=3, type=int, default=[120, 240, 240], metavar=("D", "H", "W"))
    parser.add_argument("--crop-shape", nargs=3, type=int, default=[96, 192, 192], metavar=("D", "H", "W"))
    parser.add_argument("--clip-hu", nargs=2, type=float, default=[-1000.0, 200.0], metavar=("MIN", "MAX"))
    parser.add_argument("--vjepa-input-mode", choices=("grayscale", "multi-window"), default="grayscale")
    parser.add_argument(
        "--checkpoint-metric",
        choices=(
            "val_loss",
            "val_joint_auprc",
            "val_macro_auroc",
            "val_macro_auprc",
            "val_macro_f1_0p5",
            "val_macro_bacc_0p5",
        ),
        default="val_loss",
    )
    parser.add_argument("--threshold-objective", choices=("f1", "balanced_accuracy", "youden"), default="f1")
    parser.add_argument("--max-train-batches", type=int, help="Smoke-test limiter; omit for full training.")
    parser.add_argument("--max-val-batches", type=int, help="Smoke-test limiter; omit for full validation.")
    parser.add_argument("--max-test-batches", type=int, help="Smoke-test limiter; omit for full test evaluation.")
    parser.add_argument(
        "--patch-localization",
        choices=PATCH_LOCALIZATION_MODES,
        default="linear",
        help="Patch score used for localization metrics and maps.",
    )
    parser.add_argument(
        "--checkpoint-patch-localization",
        choices=PATCH_LOCALIZATION_MODES,
        help=(
            "Patch score used only for validation checkpoint selection; "
            "defaults to --patch-localization."
        ),
    )
    parser.add_argument(
        "--debug-breakpoints",
        action="store_true",
        help="Stop once at four key multi-window, adaptive-pooling, fusion, and loss locations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_patch_localization is None:
        args.checkpoint_patch_localization = args.patch_localization
    args.backbone = normalize_backbone(args.backbone)
    spec = BACKBONES[args.backbone]
    uses_class_token = args.mil_head in {
        "class-token",
        "hybrid-class-token",
        "decoupled-class-token",
        "mct-decoupled",
    }
    uses_innovation_options = (
        uses_class_token
        or args.patch_token_pooling == "smooth-or"
        or args.anatomical_evidence
        or args.class_token_coord_embedding
        or args.class_token_anatomical_prior
        or args.token_consistency_weight > 0.0
        or args.evidence_alignment_weight > 0.0
        or args.cct_weight > 0.0
        or args.lir_weight > 0.0
        or args.fine_annotation_supervision_weight > 0.0
        or args.fine_coarse_consistency_weight > 0.0
    )
    if uses_innovation_options and spec.key != "vjepa2_1_b":
        raise ValueError("Innovation options are only supported for vjepa2_1_b")
    if not math.isfinite(args.smooth_or_temperature) or args.smooth_or_temperature <= 0.0:
        raise ValueError("--smooth-or-temperature must be finite and positive")
    if args.patch_token_pooling == "smooth-or":
        if args.patch_grid_protocol != ANATOMICAL_GRID_V2:
            raise ValueError(
                "--patch-token-pooling smooth-or requires "
                "--patch-grid-protocol anatomical_grid_v2_6x12x12"
            )
        if args.mil_head != "basic":
            raise ValueError("--patch-token-pooling smooth-or currently requires --mil-head basic")
        if args.global_local_fusion:
            raise ValueError(
                "Smooth-OR keeps case prediction as max over the same patch map; "
                "do not combine it with --global-local-fusion"
            )
    if (
        not math.isfinite(args.fine_annotation_supervision_weight)
        or not math.isfinite(args.fine_coarse_consistency_weight)
        or args.fine_annotation_supervision_weight < 0.0
        or args.fine_coarse_consistency_weight < 0.0
    ):
        raise ValueError("Fine supervision and consistency weights must be finite and non-negative")
    if args.fine_supervision_ramp_epochs < 0:
        raise ValueError("--fine-supervision-ramp-epochs must be non-negative")
    if args.fine_coarse_consistency_weight > 0.0 and args.fine_annotation_supervision_weight <= 0.0:
        raise ValueError("Fine-to-coarse consistency requires positive fine annotation supervision")
    if args.fine_annotation_supervision_weight > 0.0 and (
        spec.key != "vjepa2_1_b"
        or args.patch_grid_protocol != ANATOMICAL_GRID_V2
        or args.patch_token_pooling not in {"mean", "smooth-or"}
        or args.mil_head != "basic"
        or args.global_local_fusion
        or args.anatomical_evidence
    ):
        raise ValueError(
            "Fine annotation supervision requires basic V-JEPA, anatomical-grid-v2, "
            "mean or SmoothOR pooling, and no global/anatomical auxiliary head"
        )
    if not uses_class_token and (
        args.class_token_coord_embedding
        or args.class_token_anatomical_prior
        or args.token_consistency_weight > 0.0
        or args.cct_weight > 0.0
        or args.lir_weight > 0.0
    ):
        raise ValueError("Class-token coordinate/prior/consistency/LIR options require a class-token MIL head")
    if args.evidence_alignment_weight > 0.0 and not (uses_class_token or args.anatomical_evidence):
        raise ValueError("Evidence alignment requires a class-token head or --anatomical-evidence")
    if (
        args.token_consistency_weight < 0.0
        or args.evidence_alignment_weight < 0.0
        or args.cct_weight < 0.0
        or args.lir_weight < 0.0
    ):
        raise ValueError("Innovation loss weights must be non-negative")
    if args.lir_temperature <= 0.0:
        raise ValueError("--lir-temperature must be positive")
    if args.class_token_decoder_depth > 1 and not uses_class_token:
        raise ValueError("--class-token-decoder-depth > 1 requires a class-token MIL head")
    if (
        args.patch_localization != "linear"
        or args.checkpoint_patch_localization != "linear"
    ) and not uses_class_token:
        raise ValueError("MCT localization modes require a class-token MIL head")
    if not 0.0 < args.class_token_residual_init < 1.0:
        raise ValueError("--class-token-residual-init must be strictly between 0 and 1")
    if not 0.0 < args.anatomical_evidence_gate_init < 1.0:
        raise ValueError("--anatomical-evidence-gate-init must be strictly between 0 and 1")
    if args.anatomical_evidence_hidden_dim <= 0:
        raise ValueError("--anatomical-evidence-hidden-dim must be positive")
    if args.mct_attention_residual_max <= 0.0:
        raise ValueError("--mct-attention-residual-max must be positive")
    if not 0.0 < args.gwrp_decay <= 1.0:
        raise ValueError("--gwrp-decay must be in (0, 1]")
    checkpoint_modes = sum(
        value is not None for value in (args.resume_checkpoint, args.init_checkpoint, args.evaluate_checkpoint)
    )
    if checkpoint_modes > 1:
        raise ValueError("--resume-checkpoint, --init-checkpoint, and --evaluate-checkpoint are mutually exclusive")
    if spec.key == "voco10k_swinunetr" and args.voco_pretrained_checkpoint is None:
        raise ValueError("--backbone voco10k_swinunetr requires --voco-pretrained-checkpoint")
    if spec.key != "voco10k_swinunetr" and args.voco_pretrained_checkpoint is not None:
        raise ValueError("--voco-pretrained-checkpoint is only valid for voco10k_swinunetr")
    if args.vjepa_input_mode != "grayscale" and spec.key != "vjepa2_1_b":
        raise ValueError("--vjepa-input-mode multi-window is only supported for vjepa2_1_b")
    if args.global_local_fusion and spec.key != "vjepa2_1_b":
        raise ValueError("--global-local-fusion is only supported for vjepa2_1_b")
    if args.mil_head in {"decoupled-class-token", "mct-decoupled"} and not args.global_local_fusion:
        raise ValueError("Decoupled class-token heads require --global-local-fusion")
    if (
        args.class_token_case_loss_weight > 0.0
        and args.mil_head not in {"decoupled-class-token", "mct-decoupled"}
    ):
        raise ValueError("--class-token-case-loss-weight requires a decoupled class-token head")
    if args.mil_head == "mct-decoupled" and args.local_case_pooling != "gwrp":
        raise ValueError("--mil-head mct-decoupled requires --local-case-pooling gwrp")
    if args.mil_head == "mct-decoupled" and args.anatomical_evidence:
        raise ValueError("--mil-head mct-decoupled cannot be combined with --anatomical-evidence")
    if (
        args.case_loss_weight > 0.0
        or args.local_case_loss_weight > 0.0
        or args.class_token_case_loss_weight > 0.0
    ) and not args.global_local_fusion:
        raise ValueError("Case supervision losses require --global-local-fusion")
    if (
        args.case_loss_weight < 0.0
        or args.local_case_loss_weight < 0.0
        or args.class_token_case_loss_weight < 0.0
        or args.focal_gamma_positive < 0.0
        or args.focal_gamma_negative < 0.0
    ):
        raise ValueError("Loss weights and focal gamma values must be non-negative")
    if args.local_case_topk <= 0 or args.head_lr_multiplier <= 0.0 or args.max_case_pos_weight < 1.0:
        raise ValueError("Top-k and learning-rate/positive-weight multipliers must be positive")
    if not args.adaptive_pool_topks or any(value < 0 for value in args.adaptive_pool_topks):
        raise ValueError("--adaptive-pool-topks must contain non-negative values")
    if len(args.adaptive_pool_topks) != len(args.adaptive_pool_init_weights):
        raise ValueError("Adaptive pooling expert and initial-weight counts must match")
    if any(value <= 0.0 for value in args.adaptive_pool_init_weights):
        raise ValueError("Adaptive pooling initial weights must be positive")
    if args.adaptive_pool_lr is not None and args.adaptive_pool_lr <= 0.0:
        raise ValueError("--adaptive-pool-lr must be positive")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    seed_everything(args.seed, args.deterministic)
    runtime_determinism = deterministic_runtime_state()
    provenance = source_provenance(spec)
    print("Deterministic runtime:", json.dumps(runtime_determinism, indent=2), flush=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    fold_dir = args.splits_dir / f"fold_{args.fold}"
    train_csv = args.train_csv or fold_dir / "train.csv"
    val_csv = args.val_csv or fold_dir / "val.csv"
    test_csv = args.test_csv or args.splits_dir / "test.csv"
    for path in (train_csv, val_csv, test_csv):
        if not path.exists():
            raise FileNotFoundError(path)

    pad_shape = parse_shape(args.pad_shape)
    crop_shape = parse_shape(args.crop_shape)
    clip_hu = (float(args.clip_hu[0]), float(args.clip_hu[1]))
    if clip_hu[0] >= clip_hu[1]:
        raise ValueError("--clip-hu MIN must be smaller than MAX")

    batch_size = args.batch_size or spec.batch_size
    run_name = args.run_name or f"{spec.key}_patch_official_seed{args.seed}"
    output_dir = args.output_root / run_name / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = PatchChestCTPatchDataset(
        train_csv, CLASSES, pad_shape, crop_shape, clip_hu, random_crop=True, input_mode=args.vjepa_input_mode
    )
    val_set = PatchChestCTPatchDataset(
        val_csv, CLASSES, pad_shape, crop_shape, clip_hu, random_crop=False, input_mode=args.vjepa_input_mode
    )
    test_set = PatchChestCTPatchDataset(
        test_csv, CLASSES, pad_shape, crop_shape, clip_hu, random_crop=False, input_mode=args.vjepa_input_mode
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_batches = len(train_loader) if args.max_train_batches is None else min(len(train_loader), args.max_train_batches)
    if train_batches <= 0:
        raise RuntimeError("No training batches available")

    model = build_model(spec, crop_shape, args.num_output_classes, args=args).to(device)
    pretraining_report = getattr(model, "pretraining_report", None)
    if pretraining_report is not None:
        print(f"Pretraining initialization: {json.dumps(pretraining_report, indent=2)}", flush=True)
    initialization_report: dict[str, object] | None = None
    if args.init_checkpoint is not None:
        init_checkpoint = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        init_state = init_checkpoint.get("model", init_checkpoint)
        incompatible = model.load_state_dict(init_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in --init-checkpoint: {unexpected}")
        allowed_missing: set[str] = set()
        model_keys = set(model.state_dict())
        if args.mil_head in {
            "class-token",
            "hybrid-class-token",
            "decoupled-class-token",
            "mct-decoupled",
        }:
            allowed_missing.update(key for key in model_keys if key.startswith("patch_head."))
        if args.mil_head == "hybrid-class-token":
            allowed_missing.add("class_token_residual_logit")
        if args.mil_head == "mct-decoupled":
            allowed_missing.update(
                key for key in model_keys if key.startswith("class_token_case_head.")
            )
            allowed_missing.add("mct_attention_residual_gate")
        if args.anatomical_evidence:
            allowed_missing.update(
                key for key in model_keys if key.startswith("anatomical_evidence.")
            )
        if args.global_local_fusion:
            allowed_missing.add("fusion_logit")
            allowed_missing.update(
                key for key in model_keys if key.startswith("global_classifier.")
            )
        if args.local_case_pooling == "adaptive":
            allowed_missing.add("adaptive_pool_logits")
        missing = list(incompatible.missing_keys)
        invalid_missing = sorted(set(missing) - allowed_missing)
        if invalid_missing:
            raise RuntimeError(
                "Shared or unapproved keys are missing from --init-checkpoint: "
                f"{invalid_missing}"
            )
        required_prefixes = ["backbone."]
        if getattr(model, "classifier", None) is not None:
            required_prefixes.append("classifier.")
        absent_required = [
            prefix
            for prefix in required_prefixes
            if not any(key.startswith(prefix) for key in init_state)
        ]
        if absent_required:
            raise RuntimeError(
                "Initialization checkpoint lacks required parameter groups: "
                f"{absent_required}"
            )
        source_epoch = init_checkpoint.get("epoch") if isinstance(init_checkpoint, dict) else None
        initialization_report = {
            "path": str(args.init_checkpoint),
            "sha256": sha256_file(args.init_checkpoint),
            "source_epoch": source_epoch,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        del init_state, init_checkpoint
        print(f"Initialized model weights: {json.dumps(initialization_report)}", flush=True)
    criterion = nn.BCELoss()
    case_pos_weight = compute_case_pos_weight(
        train_set,
        args.case_pos_weight,
        args.max_case_pos_weight,
        device,
    )
    if case_pos_weight is not None:
        print(
            "Case pos_weight:",
            {class_name: float(weight) for class_name, weight in zip(CLASSES, case_pos_weight)},
            flush=True,
        )
    optimizer = make_optimizer(model, args, spec)
    optimizer_steps_per_epoch = math.ceil(train_batches / max(args.gradient_accumulation_steps, 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * optimizer_steps_per_epoch, 1),
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = {
        "script": str(Path(__file__).resolve()),
        "fold": args.fold,
        "split_protocol": {
            "train": str(train_csv),
            "val": str(val_csv),
            "test": str(test_csv),
            "rule": "train fits model; val selects best checkpoint and case thresholds; test is final evaluation",
        },
        "backbone": spec.key,
        "backbone_display_name": spec.display_name,
        "backbone_initialization": spec.init_description,
        "pretraining": spec.pretraining,
        "input_processing": {
            "source": "spacing=(1.5,1.5,3.0) NPZ paths from manifest",
            "hu_clip": list(clip_hu),
            "vjepa_input_mode": args.vjepa_input_mode,
            "multi_windows": [list(window) for window in VJEPA_MULTI_WINDOWS]
            if args.vjepa_input_mode == "multi-window"
            else None,
            "normalization": "(HU - min) / (max - min), official grounding code style",
            "pad_or_crop_shape": list(pad_shape),
            "orientation": "image rot90(k=-1, H/W axes) then left-right flip; annotation already in display frame",
            "train_crop": "random crop to crop_shape, same crop applied to image and annotation mask",
            "val_test_crop": "center crop to crop_shape",
            "crop_shape": list(crop_shape),
            "backbone_adapter": spec.init_description,
        },
        "annotation_processing": {
            "high_res_mask": "manual 24x12x12 annotations repeated to 96x192x192",
            "official_sampling": "after crop, use mask[:, 2::4, 8::16, 8::16]",
            "patch_grid_protocol": args.patch_grid_protocol,
            "official_24_to_6_pooling": (
                "reshape(B,4,6,12,12,C).sum(dim=1)>0 exactly as train_grounding.py"
                if args.patch_grid_protocol == LEGACY_OFFICIAL_GRID
                else None
            ),
            "24_to_6_pooling": (
                "reshape(B,4,6,12,12,C).sum(dim=1)>0 exactly as train_grounding.py; "
                "groups annotation planes by index modulo six"
                if args.patch_grid_protocol == LEGACY_OFFICIAL_GRID
                else "six consecutive physical-depth groups of four planes; any-positive reduction"
            ),
            "output_grid": [6, 12, 12],
            "fine_annotation_grid": (
                [24, 12, 12]
                if args.fine_annotation_supervision_weight > 0.0
                else None
            ),
            "fine_annotation_alignment": (
                "deterministic align_corners=False linear sampling of native logits at "
                "the cropped annotation cell centers"
                if args.fine_annotation_supervision_weight > 0.0
                else None
            ),
            "grid_identifier": (
                "official_patch_grid_6x12x12_spacing1p5_1p5_3p0"
                if args.patch_grid_protocol == LEGACY_OFFICIAL_GRID
                else ANATOMICAL_GRID_V2
            ),
        },
        "supervision": "fully supervised patch-level grounding",
        "loss": (
            f"patch {args.patch_loss} on sigmoid probabilities plus {args.dice_weight} * Dice loss"
            + (
                f" + {args.case_loss_weight} * case BCEWithLogits ({args.case_pos_weight} pos_weight)"
                if args.case_loss_weight > 0.0
                else ""
            )
            + (
                f" + {args.local_case_loss_weight} * local-case BCEWithLogits"
                if args.local_case_loss_weight > 0.0
                else ""
            )
            + (
                f" + {args.class_token_case_loss_weight} * class-token-case BCEWithLogits"
                if args.class_token_case_loss_weight > 0.0
                else ""
            )
            + (
                f" + {args.token_consistency_weight} * token consistency"
                f" + {args.evidence_alignment_weight} * {args.evidence_alignment_loss} evidence alignment"
                f" + {args.cct_weight} * positive-anchor CCT"
                f" + {args.lir_weight} * supervised MoRe confident-relation LIR"
                if uses_class_token or args.anatomical_evidence
                else ""
            )
            + (
                f" + {args.fine_annotation_supervision_weight} * fine 24x12x12 "
                f"({args.patch_loss} + {args.dice_weight} * Dice)"
                f" + {args.fine_coarse_consistency_weight} * fine-to-coarse probability MSE"
                f" (linear ramp {args.fine_supervision_ramp_epochs} epochs)"
                if args.fine_annotation_supervision_weight > 0.0
                else ""
            )
        ),
        "innovation": {
            "mil_head": args.mil_head,
            "patch_token_pooling": args.patch_token_pooling,
            "smooth_or_temperature": (
                args.smooth_or_temperature
                if args.patch_token_pooling == "smooth-or"
                else None
            ),
            "fine_annotation_supervision_weight": args.fine_annotation_supervision_weight,
            "fine_coarse_consistency_weight": args.fine_coarse_consistency_weight,
            "fine_supervision_ramp_epochs": args.fine_supervision_ramp_epochs,
            "fine_annotation_grid": (
                [24, 12, 12]
                if args.fine_annotation_supervision_weight > 0.0
                else None
            ),
            "fine_annotation_projection": (
                "parameter-free deterministic physical-center trilinear logit resampling "
                "from 32x24x24 to 24x12x12"
                if args.fine_annotation_supervision_weight > 0.0
                else None
            ),
            "fine_to_coarse_consistency": (
                "24x12x12 fine logits -> continuous-depth 6x12x12 LogMeanExp tau=1; "
                "probability MSE to stop-gradient direct coarse SmoothOR teacher"
                if args.fine_coarse_consistency_weight > 0.0
                else None
            ),
            "native_to_grid_pooling": (
                "legacy PyTorch adaptive-average feature pooling"
                if args.patch_grid_protocol == LEGACY_OFFICIAL_GRID
                else "mutually exclusive physical-center native-logit arithmetic mean"
                if args.patch_token_pooling == "mean"
                else "mutually exclusive physical-center native-logit LogMeanExp (Smooth-OR)"
            ),
            "class_token_heads": args.class_token_heads,
            "class_token_dropout": args.class_token_dropout,
            "class_token_decoder_depth": args.class_token_decoder_depth,
            "multi_level_attention_fusion": (
                "mean of all class-to-patch decoder attentions"
                if args.class_token_decoder_depth > 1
                else "single decoder attention"
            ),
            "coordinate_embedding": args.class_token_coord_embedding,
            "coordinate_mode": args.class_token_coord_mode,
            "anatomical_prior": args.class_token_anatomical_prior,
            "anatomical_prior_gamma": args.class_token_prior_gamma,
            "anatomical_prior_grid": list(args.class_token_prior_grid),
            "class_token_residual_init": args.class_token_residual_init,
            "token_consistency_weight": args.token_consistency_weight,
            "evidence_alignment_weight": args.evidence_alignment_weight,
            "evidence_alignment_loss": args.evidence_alignment_loss,
            "cct_weight": args.cct_weight,
            "cct_scope": (
                "selected nine class tokens at every decoder level; positive case-label anchors; "
                "mean over decoder levels and valid cases; negative-only cases excluded"
            ),
            "lir_weight": args.lir_weight,
            "lir_temperature": args.lir_temperature,
            "lir_scope": (
                "MoRe confident class-patch relation contrast adapted to supervised "
                "official 6x12x12 masks; detached patch-token targets; empty masks excluded"
            ),
            "mct_attention_residual_max": args.mct_attention_residual_max,
            "mct_attention_residual_parameterization": (
                f"{args.mct_attention_residual_max} * tanh(theta), theta initialized to zero"
                if args.mil_head == "mct-decoupled"
                else None
            ),
            "crop_aware_anatomical_evidence": args.anatomical_evidence,
            "anatomical_evidence_hidden_dim": args.anatomical_evidence_hidden_dim,
            "anatomical_evidence_gate_init": args.anatomical_evidence_gate_init,
            "global_local_fusion": args.global_local_fusion,
            "local_case_pooling": args.local_case_pooling,
            "local_case_topk": args.local_case_topk,
            "gwrp_decay": args.gwrp_decay,
            "adaptive_pool_topks": args.adaptive_pool_topks,
            "adaptive_pool_init_weights": args.adaptive_pool_init_weights,
            "adaptive_pool_lr": args.adaptive_pool_lr,
            "fusion_local_init": args.fusion_local_init,
            "case_loss_weight": args.case_loss_weight,
            "local_case_loss_weight": args.local_case_loss_weight,
            "class_token_case_loss_weight": args.class_token_case_loss_weight,
            "patch_output_head": (
                "linear plus zero-initialized class-attention logit residual"
                if args.mil_head == "mct-decoupled"
                else "linear"
                if args.mil_head == "decoupled-class-token"
                else args.mil_head
            ),
            "class_token_case_pooling": (
                "shared Linear(D,1)"
                if args.mil_head == "mct-decoupled"
                else "max"
                if args.mil_head == "decoupled-class-token"
                else None
            ),
            "case_pos_weight": args.case_pos_weight,
            "patch_loss": args.patch_loss,
            "focal_gamma_positive": args.focal_gamma_positive,
            "focal_gamma_negative": args.focal_gamma_negative,
        },
        "optimizer": {
            "name": args.optimizer or spec.optimizer,
            "backbone_lr": args.lr,
            "head_lr": args.lr * args.head_lr_multiplier,
            "head_lr_multiplier": args.head_lr_multiplier,
            "adaptive_pool_lr": args.adaptive_pool_lr or args.lr * args.head_lr_multiplier,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
        },
        "pretraining_checkpoint": pretraining_report,
        "initialization_checkpoint": initialization_report,
        "scheduler": "CosineAnnealingLR stepped after each optimizer update",
        "checkpoint_selection_metric": args.checkpoint_metric,
        "checkpoint_selection_loss_scope": (
            "original coarse 6x12x12 patch BCE+Dice only; fine auxiliary losses excluded"
            if args.fine_annotation_supervision_weight > 0.0
            else "complete configured validation loss"
        ),
        "case_threshold_selection_rule": f"per-class threshold selected on val by maximizing {args.threshold_objective}",
        "patch_metric_rule": "Patch-AUPRC and Patch-DSC use official positive-annotation-case patch set; Patch-DSC is per-class best threshold over 0..1 step 0.01",
        "summary_metrics": ["AUROC", "AUPRC", "Macro-F1", "BACC", "Patch-AUPRC", "Patch-DSC"],
        "mean_std_scope": "over nine abnormalities, not over random seeds",
        "num_output_classes": args.num_output_classes,
        "classes": CLASSES,
        "selected_idx": SELECTED_IDX,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "deterministic_runtime": runtime_determinism,
        "deterministic_fallbacks": {
            **(
                {
                    "mvit_residual_max_pool": (
                        "CUDA max-pool forward with saved argmax; deterministic CPU scatter-add backward"
                    )
                }
                if args.deterministic and spec.key == "mvit_v2_s"
                else {}
            ),
            **(
                {
                    "vjepa_adaptive_avg_pool": (
                        "CUDA adaptive-average forward with deterministic CPU backward"
                    )
                }
                if (
                    args.deterministic
                    and spec.key == "vjepa2_1_b"
                    and args.patch_grid_protocol == LEGACY_OFFICIAL_GRID
                )
                else {}
            ),
        },
        "source_provenance": provenance,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": batch_size * args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "amp": use_amp,
        "patch_localization": args.patch_localization,
        "checkpoint_patch_localization": args.checkpoint_patch_localization,
        "num_cases": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "smoke_limiters": {
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "max_test_batches": args.max_test_batches,
        },
    }
    (output_dir / "config.json").write_text(json.dumps(jsonable(config), indent=2), encoding="utf-8")

    start_epoch = 0
    log_rows: list[dict[str, str]] = []
    best_metric: float | None = None
    best_epoch = 0
    best_higher = args.checkpoint_metric != "val_loss"
    if args.evaluate_checkpoint is not None:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        best_metric = checkpoint.get("best_metric")
        best_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = args.epochs
        print(f"Evaluating checkpoint {args.evaluate_checkpoint} from epoch {best_epoch}.", flush=True)
    elif args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_metric = checkpoint.get("best_metric")
        start_epoch = int(checkpoint.get("epoch", 0))
        log_rows = [row for row in read_csv(output_dir / "train_log.csv") if int(row["epoch"]) <= start_epoch]
        for row in log_rows:
            if int(row.get("is_best", "0")):
                best_epoch = int(row["epoch"])
        if best_epoch == 0 and start_epoch > 0:
            best_epoch = start_epoch
        print(
            f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}; "
            f"loaded {len(log_rows)} log rows; next epoch is {start_epoch + 1}.",
            flush=True,
        )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        auxiliary_scale = (
            min(epoch / float(args.fine_supervision_ramp_epochs), 1.0)
            if args.fine_supervision_ramp_epochs > 0
            else 1.0
        )
        effective_fine_weight = args.fine_annotation_supervision_weight * auxiliary_scale
        effective_consistency_weight = args.fine_coarse_consistency_weight * auxiliary_scale
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            SELECTED_IDX,
            args.patch_grid_protocol,
            scaler,
            use_amp,
            args.dice_weight,
            args.patch_loss,
            args.focal_gamma_positive,
            args.focal_gamma_negative,
            effective_fine_weight,
            effective_consistency_weight,
            args.smooth_or_temperature,
            args.case_loss_weight,
            args.local_case_loss_weight,
            args.class_token_case_loss_weight,
            case_pos_weight,
            args.token_consistency_weight,
            args.evidence_alignment_weight,
            args.evidence_alignment_loss,
            args.cct_weight,
            args.lir_weight,
            args.lir_temperature,
            args.gradient_accumulation_steps,
            args.max_train_batches,
            args.debug_breakpoints,
        )
        val_bundle = predict(
            model,
            val_loader,
            criterion,
            device,
            SELECTED_IDX,
            args.patch_grid_protocol,
            use_amp,
            args.dice_weight,
            args.patch_loss,
            args.focal_gamma_positive,
            args.focal_gamma_negative,
            effective_fine_weight,
            effective_consistency_weight,
            args.smooth_or_temperature,
            args.case_loss_weight,
            args.local_case_loss_weight,
            args.class_token_case_loss_weight,
            case_pos_weight,
            args.token_consistency_weight,
            args.evidence_alignment_weight,
            args.evidence_alignment_loss,
            args.cct_weight,
            args.lir_weight,
            args.lir_temperature,
            desc="val",
            max_batches=args.max_val_batches,
            collect_patch=args.checkpoint_metric == "val_joint_auprc",
            patch_localization=args.checkpoint_patch_localization,
        )
        thresholds_0p5 = {class_name: 0.5 for class_name in CLASSES}
        val_rows_0p5 = per_class_metrics(
            CLASSES,
            val_bundle,
            thresholds_0p5,
            compute_patch_dice=False,
        )
        current_metric, higher_is_better = selection_value(args.checkpoint_metric, val_bundle.loss, val_rows_0p5)
        if higher_is_better != best_higher:
            raise RuntimeError("Internal checkpoint metric direction mismatch")
        is_best = is_better(current_metric, best_metric, higher_is_better)
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best_metric,
                    "config": config,
                },
                output_dir / "best.pt",
            )
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_metric,
                "config": config,
            },
            output_dir / "last.pt",
        )

        log_row = {
            "epoch": str(epoch),
            "train_loss": fmt_float(train_losses.total),
            "train_coarse_loss": fmt_float(train_losses.coarse),
            "train_fine_loss": fmt_float(train_losses.fine),
            "train_consistency_loss": fmt_float(train_losses.consistency),
            "val_loss": fmt_float(val_bundle.loss),
            "val_total_loss": fmt_float(val_bundle.total_loss),
            "val_fine_loss": fmt_float(val_bundle.fine_loss),
            "val_consistency_loss": fmt_float(val_bundle.consistency_loss),
            "fine_auxiliary_scale": fmt_float(auxiliary_scale),
            "effective_fine_weight": fmt_float(effective_fine_weight),
            "effective_consistency_weight": fmt_float(effective_consistency_weight),
            "val_macro_auroc": fmt_float(mean_or_nan([float(row["auroc"]) for row in val_rows_0p5])),
            "val_macro_auprc": fmt_float(mean_or_nan([float(row["auprc"]) for row in val_rows_0p5])),
            "val_macro_patch_auprc": fmt_float(
                mean_or_nan([float(row["patch_auprc"]) for row in val_rows_0p5])
            ),
            "val_macro_f1_0p5": fmt_float(mean_or_nan([float(row["macro_f1"]) for row in val_rows_0p5])),
            "val_macro_bacc_0p5": fmt_float(mean_or_nan([float(row["bacc"]) for row in val_rows_0p5])),
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_metric_value": fmt_float(current_metric),
            "is_best": str(int(is_best)),
            "best_epoch": str(best_epoch),
            "lr": fmt_float(optimizer.param_groups[0]["lr"], 10),
        }
        log_rows.append(log_row)
        write_csv(output_dir / "train_log.csv", log_rows)
        print(json.dumps(log_row, indent=2), flush=True)

    best_path = args.evaluate_checkpoint or (output_dir / "best.pt")
    if not best_path.exists():
        raise RuntimeError("Training finished without writing best.pt")
    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    val_bundle = predict(
        model,
        val_loader,
        criterion,
        device,
        SELECTED_IDX,
        args.patch_grid_protocol,
        use_amp,
        args.dice_weight,
        args.patch_loss,
        args.focal_gamma_positive,
        args.focal_gamma_negative,
        args.fine_annotation_supervision_weight,
        args.fine_coarse_consistency_weight,
        args.smooth_or_temperature,
        args.case_loss_weight,
        args.local_case_loss_weight,
        args.class_token_case_loss_weight,
        case_pos_weight,
        args.token_consistency_weight,
        args.evidence_alignment_weight,
        args.evidence_alignment_loss,
        args.cct_weight,
        args.lir_weight,
        args.lir_temperature,
        desc="val-best",
        max_batches=args.max_val_batches,
        collect_patch=False,
        patch_localization=args.patch_localization,
    )
    thresholds, threshold_rows = select_thresholds(CLASSES, val_bundle, args.threshold_objective)
    test_bundle = predict(
        model,
        test_loader,
        criterion,
        device,
        SELECTED_IDX,
        args.patch_grid_protocol,
        use_amp,
        args.dice_weight,
        args.patch_loss,
        args.focal_gamma_positive,
        args.focal_gamma_negative,
        args.fine_annotation_supervision_weight,
        args.fine_coarse_consistency_weight,
        args.smooth_or_temperature,
        args.case_loss_weight,
        args.local_case_loss_weight,
        args.class_token_case_loss_weight,
        case_pos_weight,
        args.token_consistency_weight,
        args.evidence_alignment_weight,
        args.evidence_alignment_loss,
        args.cct_weight,
        args.lir_weight,
        args.lir_temperature,
        desc="test",
        max_batches=args.max_test_batches,
        collect_patch=True,
        patch_localization=args.patch_localization,
    )
    test_rows = per_class_metrics(CLASSES, test_bundle, thresholds)
    summary_row = summarize_metrics(spec, test_rows)

    save_prediction_csv(output_dir / "val_predictions.csv", CLASSES, val_bundle, thresholds)
    save_prediction_csv(output_dir / "test_predictions.csv", CLASSES, test_bundle, thresholds)
    save_threshold_csv(output_dir / "thresholds.csv", threshold_rows)
    save_per_class_csv(output_dir / "per_class_metrics.csv", spec, test_rows)
    write_csv(output_dir / "summary_metrics.csv", [summary_row])
    adaptive_pool_path = output_dir / "adaptive_pool_weights.csv"
    saved_adaptive_pool_weights = save_adaptive_pool_weights(
        adaptive_pool_path,
        model,
        SELECTED_IDX,
        CLASSES,
    )

    final_config = dict(config)
    final_config["best_epoch"] = best_epoch
    final_config["best_metric_value"] = best_metric
    final_config["outputs"] = {
        "best_checkpoint": str(best_path),
        "train_log": str(output_dir / "train_log.csv"),
        "val_predictions": str(output_dir / "val_predictions.csv"),
        "test_predictions": str(output_dir / "test_predictions.csv"),
        "thresholds": str(output_dir / "thresholds.csv"),
        "per_class_metrics": str(output_dir / "per_class_metrics.csv"),
        "summary_metrics": str(output_dir / "summary_metrics.csv"),
    }
    if saved_adaptive_pool_weights:
        final_config["outputs"]["adaptive_pool_weights"] = str(adaptive_pool_path)
    (output_dir / "config.json").write_text(json.dumps(jsonable(final_config), indent=2), encoding="utf-8")

    print(f"Best epoch: {best_epoch}")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()

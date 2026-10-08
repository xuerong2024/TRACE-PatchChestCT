#!/usr/bin/env python3
"""Train PatchChestCT fold-0 case-level baselines with official NoisyOR BCE.

This script keeps PatchChestCT's official weak/case-level method choices while
using our explicit train/val/test split:

- train: model fitting only
- val: best checkpoint selection and per-class threshold selection
- test: final reporting only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
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

# This must be configured before the first CUDA BLAS handle is created. It is
# harmless for non-deterministic runs and required by strict CUDA determinism.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PATCHCHESTCT_ROOT = REPO_ROOT / "nnunet_data" / "Bronchidata" / "PatchChestCT"
DEFAULT_SPLITS_DIR = PATCHCHESTCT_ROOT / "manifests" / "cv5_validtest_seed2026_spacing1p5_1p5_3p0"
DEFAULT_OUTPUT_ROOT = PATCHCHESTCT_ROOT / "case_official_spacing1p5_1p5_3p0_fold0_runs"

ALL_PATHOLOGIES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

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
OFFICIAL_LOGITS_GRID = (6, 12, 12)
OFFICIAL_BAG_SIZE = 6 * 12 * 12


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    display_name: str
    model_file: Path
    class_name: str | None
    batch_size: int
    latent_dim: int
    pretraining: str
    init_description: str
    builder_model_name: str | None = None
    source_files: tuple[Path, ...] = ()


BACKBONES = {
    "r3d18": BackboneSpec(
        key="r3d18",
        display_name="R3D-18",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_r3d18" / "model.py",
        class_name="R3D18PatchClassifier",
        batch_size=10,
        latent_dim=512,
        pretraining="Torchvision Kinetics-400",
        init_description=(
            "torchvision.models.video.r3d_18 with R3D_18_Weights.DEFAULT via pretrained=True; "
            "first conv replaced for 1-channel CT; classifier head newly initialized"
        ),
    ),
    "swin3d_t": BackboneSpec(
        key="swin3d_t",
        display_name="Swin3D-T",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_swin3d_t" / "model.py",
        class_name="Swin3DTPatchClassifier",
        batch_size=6,
        latent_dim=768,
        pretraining="Torchvision Kinetics-400",
        init_description=(
            "torchvision.models.video.swin3d_t with Swin3D_T_Weights.DEFAULT; "
            "tubelet projection replaced for 1-channel CT; classifier head newly initialized"
        ),
    ),
    "mvit_v2_s": BackboneSpec(
        key="mvit_v2_s",
        display_name="MViT-v2-S",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_mvit_v2_s" / "model.py",
        class_name="MViTV2SPatchClassifier",
        batch_size=6,
        latent_dim=512,
        pretraining="None",
        init_description=(
            "official custom torchvision.models.video.mvit.MViT block setting; "
            "no explicit torchvision pretrained weights are loaded; 1-channel conv and classifier head newly initialized"
        ),
    ),
    "vjepa2_1_b": BackboneSpec(
        key="vjepa2_1_b",
        display_name="V-JEPA2.1-B",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_vjepa2_1" / "model.py",
        class_name="VJEPA21OfficialPatchClassifier",
        batch_size=2,
        latent_dim=768,
        pretraining="Official V-JEPA 2.1 ViT-B/16 384px",
        init_description=(
            "facebookresearch/vjepa2 V-JEPA 2.1 ViT-B/16 384px pretrained encoder; "
            "dense token classifier newly initialized and adapted to the official 6x12x12 patch grid"
        ),
    ),
    "voco10k_swinunetr": BackboneSpec(
        key="voco10k_swinunetr",
        display_name="VoCo-10K SwinUNETR/16",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_voco10k" / "model.py",
        class_name="VoCo10KSwinUNETRPatchClassifier",
        batch_size=1,
        latent_dim=384,
        pretraining="VoCo-10K SwinUNETR-v2",
        init_description=(
            "MONAI SwinUNETR-v2 /16 encoder initialized from the verified VoCo_10k.pt; "
            "segmentation decoder removed and a new dense 18-label linear head added"
        ),
    ),
    "ctclip": BackboneSpec(
        key="ctclip",
        display_name="CT-CLIP",
        model_file=(
            REPO_ROOT
            / "classification_code"
            / "patchchestct_official_baselines"
            / "ctclip.py"
        ),
        class_name=None,
        batch_size=1,
        latent_dim=512,
        pretraining="Official CT-CLIP v2 CTViT image tower",
        init_description=(
            "released CT-CLIP_v2.pt active image path with all 147 patch-embed, "
            "spatial/temporal Transformer, and cosine-VQ tensors loaded; text/global "
            "projection removed and a new per-token dense 18-label head added"
        ),
        builder_model_name="ctclip",
        source_files=(
            REPO_ROOT
            / "classification_code"
            / "official_repos"
            / "CT-CLIP"
            / "transformer_maskgit"
            / "transformer_maskgit"
            / "ctvit.py",
            REPO_ROOT
            / "classification_code"
            / "official_repos"
            / "CT-CLIP"
            / "transformer_maskgit"
            / "transformer_maskgit"
            / "attention.py",
            REPO_ROOT
            / "classification_code"
            / "official_repos"
            / "CT-CLIP"
            / "transformer_maskgit"
            / "setup.py",
        ),
    ),
    "medicalnet_resnet18_23": BackboneSpec(
        key="medicalnet_resnet18_23",
        display_name="MedicalNet-ResNet18-23",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_official_baselines" / "medical.py",
        class_name=None,
        batch_size=1,
        latent_dim=512,
        pretraining="MedicalNet / Med3D 23-dataset checkpoint",
        init_description=(
            "MedicalNet 3D-ResNet18 pretrained encoder; native global head removed; "
            "12x24x24 trunk grid reduced by exact non-overlapping 2x2x2 mean and "
            "classified by a new dense 1x1x1 Conv3d head"
        ),
        builder_model_name="medicalnet_resnet18_23",
        source_files=(
            REPO_ROOT / "classification_code" / "patchchestct_medical_models" / "models.py",
            REPO_ROOT / "classification_code" / "deterministic_ops.py",
        ),
    ),
    "medicalnet_resnet50_23": BackboneSpec(
        key="medicalnet_resnet50_23",
        display_name="MedicalNet-ResNet50-23",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_official_baselines" / "medical.py",
        class_name=None,
        batch_size=1,
        latent_dim=2048,
        pretraining="MedicalNet / Med3D 23-dataset checkpoint",
        init_description=(
            "MedicalNet 3D-ResNet50 pretrained encoder; native global head removed; "
            "12x24x24 trunk grid reduced by exact non-overlapping 2x2x2 mean and "
            "classified by a new dense 1x1x1 Conv3d head"
        ),
        builder_model_name="medicalnet_resnet50_23",
        source_files=(
            REPO_ROOT / "classification_code" / "patchchestct_medical_models" / "models.py",
            REPO_ROOT / "classification_code" / "deterministic_ops.py",
        ),
    ),
    "modelgenesis_chest_ct": BackboneSpec(
        key="modelgenesis_chest_ct",
        display_name="Models Genesis Chest CT",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_official_baselines" / "medical.py",
        class_name=None,
        batch_size=1,
        latent_dim=512,
        pretraining="Models Genesis Chest CT self-supervised checkpoint",
        init_description=(
            "Models Genesis Chest CT pretrained encoder; decoder/global head removed; "
            "12x24x24 trunk grid reduced by exact non-overlapping 2x2x2 mean and "
            "classified by a new dense 1x1x1 Conv3d head"
        ),
        builder_model_name="modelgenesis_chest_ct",
        source_files=(
            REPO_ROOT / "classification_code" / "patchchestct_medical_models" / "models.py",
            REPO_ROOT / "classification_code" / "deterministic_ops.py",
        ),
    ),
    "videomae": BackboneSpec(
        key="videomae",
        display_name="VideoMAE",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_official_baselines" / "video_transformers.py",
        class_name=None,
        batch_size=1,
        latent_dim=768,
        pretraining="MCG-NJU VideoMAE Base fine-tuned on Kinetics",
        init_description=(
            "pretrained VideoMAE encoder with temporal tubelet kernel/stride adapted "
            "from 2 to 16 so all 96 CT slices produce a native 6x12x12 token grid; "
            "new per-token dense classifier"
        ),
        builder_model_name="videomae",
        source_files=(
            REPO_ROOT / "classification_code" / "patchchestct_video_models" / "models.py",
            REPO_ROOT / "classification_code" / "deterministic_ops.py",
        ),
    ),
    "timesformer": BackboneSpec(
        key="timesformer",
        display_name="TimeSformer",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_official_baselines" / "video_transformers.py",
        class_name=None,
        batch_size=1,
        latent_dim=768,
        pretraining="Facebook TimeSformer Base fine-tuned on Kinetics-400",
        init_description=(
            "pretrained TimeSformer encoder; all 96 CT slices are reduced into six "
            "non-overlapping 16-slice means before the unchanged 16x16 patch embed; "
            "CLS is excluded and a new dense token classifier yields 6x12x12"
        ),
        builder_model_name="timesformer",
        source_files=(
            REPO_ROOT / "classification_code" / "patchchestct_video_models" / "models.py",
            REPO_ROOT / "classification_code" / "deterministic_ops.py",
        ),
    ),
    "i3d_r50": BackboneSpec(
        key="i3d_r50",
        display_name="I3D-R50",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_official_baselines" / "video_conv.py",
        class_name=None,
        batch_size=1,
        latent_dim=2048,
        pretraining="PyTorchVideo I3D-R50 Kinetics-400",
        init_description=(
            "pretrained PyTorchVideo I3D-R50; RGB stem deterministically adapted to "
            "one-channel temporal kernel/stride 16, later temporal pool removed and "
            "final spatial stride retained at 12x12; global head replaced by dense Conv3d"
        ),
        builder_model_name="i3d_r50",
        source_files=(
            REPO_ROOT / "classification_code" / "deterministic_ops.py",
        ),
    ),
    "slow_r50": BackboneSpec(
        key="slow_r50",
        display_name="Slow-R50",
        model_file=REPO_ROOT / "classification_code" / "patchchestct_official_baselines" / "video_conv.py",
        class_name=None,
        batch_size=1,
        latent_dim=2048,
        pretraining="PyTorchVideo Slow-R50 Kinetics-400",
        init_description=(
            "pretrained PyTorchVideo Slow-R50; RGB stem deterministically adapted to "
            "one-channel temporal kernel/stride 16 and final spatial stride retained "
            "at 12x12; global head replaced by dense Conv3d"
        ),
        builder_model_name="slow_r50",
        source_files=(
            REPO_ROOT / "classification_code" / "deterministic_ops.py",
        ),
    ),
}


def parse_shape(values: list[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("shape must contain D H W")
    return int(values[0]), int(values[1]), int(values[2])


def normalize_backbone(name: str) -> str:
    if name == "mvit":
        return "mvit_v2_s"
    return name


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


def crop_3d(arr: np.ndarray, target: tuple[int, int, int], random_crop: bool) -> np.ndarray:
    starts: list[int] = []
    for axis, tgt in enumerate(target):
        cur = arr.shape[axis]
        if cur < tgt:
            raise ValueError(f"Cannot crop axis {axis} from {cur} to {tgt}")
        if random_crop and cur > tgt:
            starts.append(random.randint(0, cur - tgt))
        else:
            starts.append((cur - tgt) // 2)
    d0, h0, w0 = starts
    d, h, w = target
    return arr[d0 : d0 + d, h0 : h0 + h, w0 : w0 + w]


def load_official_case_input(
    image_path: Path,
    pad_shape: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
    random_crop: bool,
    clip_hu: tuple[float, float],
) -> torch.Tensor:
    with np.load(image_path) as data:
        arr = np.asarray(data["arr_0"], dtype=np.float32) * 1000.0
    if arr.ndim != 3:
        raise ValueError(f"{image_path} arr_0 has shape {arr.shape}; expected 3D")

    hu_min, hu_max = clip_hu
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)

    arr = center_crop_or_pad_3d(arr, pad_shape)
    arr = np.rot90(arr, k=-1, axes=(1, 2))
    arr = np.flip(arr, axis=2).copy()
    arr = crop_3d(arr, crop_shape, random_crop=random_crop)
    return torch.from_numpy(arr[None].astype(np.float32, copy=False))


class PatchChestCTCaseDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        classes: list[str],
        pad_shape: tuple[int, int, int],
        crop_shape: tuple[int, int, int],
        clip_hu: tuple[float, float],
        random_crop: bool,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.base_dir = self.manifest_csv.parent
        self.classes = classes
        self.pad_shape = pad_shape
        self.crop_shape = crop_shape
        self.clip_hu = clip_hu
        self.random_crop = random_crop
        with self.manifest_csv.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise RuntimeError(f"No rows found in {self.manifest_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_path = resolve_path(row["image_path"], self.base_dir)
        labels = [float(row[f"{class_name}_label"]) for class_name in self.classes]
        return {
            "image": load_official_case_input(
                image_path=image_path,
                pad_shape=self.pad_shape,
                crop_shape=self.crop_shape,
                random_crop=self.random_crop,
                clip_hu=self.clip_hu,
            ),
            "case_target": torch.tensor(labels, dtype=torch.float32),
            "volume_id": row["volume_id"],
        }


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # class decorator runs, so dynamically loaded adapter modules must be
    # registered before executing their source.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def build_model(
    spec: BackboneSpec,
    crop_shape: tuple[int, int, int],
    num_output_classes: int,
    deterministic: bool,
) -> nn.Module:
    module = load_module(spec.model_file, f"patchchestct_{spec.key}_model_for_case")
    if spec.builder_model_name is not None:
        builder = getattr(module, "build_model")
        if spec.key in {
            "medicalnet_resnet18_23",
            "medicalnet_resnet50_23",
            "modelgenesis_chest_ct",
        }:
            return builder(
                spec.builder_model_name,
                num_classes=num_output_classes,
                pretrained=True,
            )
        if spec.key in {"videomae", "timesformer"}:
            return builder(
                spec.builder_model_name,
                num_classes=num_output_classes,
                pretrained=True,
                deterministic=deterministic,
            )
        if spec.key in {"i3d_r50", "slow_r50"}:
            return builder(
                spec.builder_model_name,
                num_classes=num_output_classes,
                pretrained=True,
                deterministic_max_pool=deterministic,
            )
        if spec.key == "ctclip":
            return builder(
                spec.builder_model_name,
                num_classes=num_output_classes,
                pretrained=True,
                deterministic=deterministic,
            )
        raise RuntimeError(f"No official builder dispatch is defined for {spec.key}")
    if spec.class_name is None:
        raise RuntimeError(f"Backbone {spec.key} has neither class_name nor builder_model_name")
    model_cls = getattr(module, spec.class_name)
    if spec.key == "mvit_v2_s":
        return model_cls(
            num_classes=num_output_classes,
            input_shape=crop_shape,
            pretrained=False,
            deterministic_max_pool=deterministic,
        )
    if spec.key == "vjepa2_1_b":
        return model_cls(
            num_classes=num_output_classes,
            pretrained=True,
            deterministic_adaptive_pool=deterministic,
        )
    if spec.key == "voco10k_swinunetr":
        return model_cls(
            num_classes=num_output_classes,
            pretrained=True,
            pretrained_checkpoint=(
                PATCHCHESTCT_ROOT / "pretrained_weights" / "voco10k" / "VoCo_10k.pt"
            ),
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


def cuda_device_metadata(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    return {
        "logical_index": index,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "uuid": str(getattr(properties, "uuid", "")) or None,
    }


def source_provenance(spec: BackboneSpec) -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        spec.model_file.resolve(),
        *(path.resolve() for path in spec.source_files),
    ]
    if spec.key == "vjepa2_1_b":
        source_files.append(
            (REPO_ROOT / "classification_code" / "patchchestct_video_models" / "models.py").resolve()
        )
    if spec.key in {"videomae", "timesformer"}:
        module_name = (
            "transformers.models.videomae.modeling_videomae"
            if spec.key == "videomae"
            else "transformers.models.timesformer.modeling_timesformer"
        )
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or module_spec.origin is None:
            raise RuntimeError(f"Could not resolve installed source for {module_name}")
        module_root = Path(module_spec.origin).resolve().parent
        source_files.extend(module_root.glob("*.py"))
    if spec.key in {"i3d_r50", "slow_r50"}:
        module_spec = importlib.util.find_spec("pytorchvideo.models")
        if module_spec is None or not module_spec.submodule_search_locations:
            raise RuntimeError("Could not resolve installed PyTorchVideo model sources")
        for location in module_spec.submodule_search_locations:
            source_files.extend(Path(location).resolve().rglob("*.py"))
    source_files = sorted(set(source_files))
    package_versions: dict[str, str | None] = {}
    for distribution in ("torch", "torchvision", "transformers", "pytorchvideo", "safetensors"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "packages": package_versions,
        "files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretrained_artifact_provenance(spec: BackboneSpec) -> dict[str, str]:
    """Hash the exact cached checkpoints used by the added official adapters."""

    torch_checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    paths: list[Path]
    if spec.key == "medicalnet_resnet18_23":
        paths = [
            PATCHCHESTCT_ROOT
            / "pretrained_medical_models"
            / "MedicalNet"
            / "resnet_18_23dataset.pth"
        ]
    elif spec.key == "medicalnet_resnet50_23":
        paths = [
            PATCHCHESTCT_ROOT
            / "pretrained_medical_models"
            / "MedicalNet"
            / "resnet_50_23dataset.pth"
        ]
    elif spec.key == "modelgenesis_chest_ct":
        paths = [torch_checkpoints / "Genesis_Chest_CT.pt"]
    elif spec.key == "i3d_r50":
        paths = [torch_checkpoints / "I3D_8x8_R50.pyth"]
    elif spec.key == "slow_r50":
        paths = [torch_checkpoints / "SLOW_8x8_R50.pyth"]
    elif spec.key == "videomae":
        cache = hf_cache / "models--MCG-NJU--videomae-base-finetuned-kinetics"
        ref = cache / "refs" / "main"
        if not ref.is_file():
            raise FileNotFoundError(f"VideoMAE cache ref is missing: {ref}")
        snapshot = cache / "snapshots" / ref.read_text(encoding="utf-8").strip()
        paths = [snapshot / "config.json", snapshot / "model.safetensors"]
    elif spec.key == "timesformer":
        cache = hf_cache / "models--facebook--timesformer-base-finetuned-k400"
        config_candidates = sorted(cache.glob("snapshots/*/config.json"))
        weight_candidates = sorted(cache.glob("snapshots/*/model.safetensors"))
        if not config_candidates or not weight_candidates:
            raise FileNotFoundError(
                f"TimeSformer cached config/safetensors pair is missing under {cache}"
            )
        # This is the same deterministic selection used by
        # _local_timesformer_safetensors_dir in the shared video-model module.
        paths = [config_candidates[0], weight_candidates[0]]
    elif spec.key == "ctclip":
        paths = [
            PATCHCHESTCT_ROOT
            / "pretrained_medical_models"
            / "CT-CLIP"
            / "CT-CLIP_v2.pt"
        ]
    elif spec.key == "voco10k_swinunetr":
        paths = [
            PATCHCHESTCT_ROOT
            / "pretrained_weights"
            / "voco10k"
            / "VoCo_10k.pt"
        ]
    else:
        return {}

    provenance: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Pretrained artifact is missing: {resolved}")
        provenance[str(resolved)] = sha256_file(resolved)
    return provenance


def model_protocol_metadata(model: nn.Module) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    for attribute in (
        "_official_mil_metadata",
        "adapter_metadata",
        "metadata_dict",
        "_pretrained_load_report",
        "pretrained_load_report",
        "pretraining_report",
        "_pretrained_trunk_coverage",
        "_deterministic_replacements",
    ):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            if not callable(value):
                metadata[attribute] = value
    export_metadata = getattr(model, "export_metadata", None)
    if callable(export_metadata):
        metadata["export_metadata"] = export_metadata()
    return metadata


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def noisy_or_case_probs(logits: torch.Tensor, selected_idx: list[int], alpha: float) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError(f"Expected dense logits with shape (B, C, D, H, W), got {tuple(logits.shape)}")
    if logits.shape[1] != len(ALL_PATHOLOGIES):
        raise ValueError(
            "Official case comparison requires exactly "
            f"{len(ALL_PATHOLOGIES)} dense channels before target selection, "
            f"got {logits.shape[1]}"
        )
    if tuple(logits.shape[-3:]) != OFFICIAL_LOGITS_GRID:
        raise ValueError(
            f"Official case comparison requires logits grid {OFFICIAL_LOGITS_GRID} "
            f"(bag size {OFFICIAL_BAG_SIZE}), got {tuple(logits.shape[-3:])}"
        )
    patch_probs = logits[:, selected_idx].sigmoid()
    spatial_probs = patch_probs.flatten(start_dim=2)
    if spatial_probs.shape[-1] != OFFICIAL_BAG_SIZE:
        raise RuntimeError(
            f"Official NoisyOR must aggregate {OFFICIAL_BAG_SIZE} instances, got {spatial_probs.shape[-1]}"
        )
    return 1.0 - (1.0 - alpha * spatial_probs).prod(dim=2)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


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
    volume_ids: list[str]
    probabilities: np.ndarray
    targets: np.ndarray


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    selected_idx: list[int],
    noisy_or_alpha: float,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    max_batches: int | None,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    effective_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["case_target"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        probs = noisy_or_case_probs(logits.float(), selected_idx, noisy_or_alpha)
        loss = criterion(probs, targets.float())
        accumulation_group_start = (total_batches // gradient_accumulation_steps) * gradient_accumulation_steps
        accumulation_group_size = min(
            gradient_accumulation_steps,
            effective_batches - accumulation_group_start,
        )
        scaled_loss = loss / accumulation_group_size
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        total_loss += float(loss.item())
        total_batches += 1
        should_step = (
            total_batches % gradient_accumulation_steps == 0
            or total_batches == effective_batches
        )
        if should_step:
            optimizer_was_run = True
            if use_amp:
                scale_before = scaler.get_scale()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer_was_run = scaler.get_scale() >= scale_before
            else:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm,
                    )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if optimizer_was_run:
                scheduler.step()
    if total_batches == 0:
        raise RuntimeError("No training batches were produced")
    return total_loss / total_batches


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    selected_idx: list[int],
    noisy_or_alpha: float,
    use_amp: bool,
    desc: str,
    max_batches: int | None,
) -> PredictionBundle:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_ids: list[str] = []
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for batch_index, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["case_target"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        probs = noisy_or_case_probs(logits.float(), selected_idx, noisy_or_alpha)
        loss = criterion(probs, targets.float())
        total_loss += float(loss.item())
        total_batches += 1
        all_ids.extend(get_volume_ids(batch["volume_id"]))
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    if total_batches == 0:
        raise RuntimeError(f"No {desc} batches were produced")
    return PredictionBundle(
        loss=total_loss / total_batches,
        volume_ids=all_ids,
        probabilities=np.concatenate(all_probs, axis=0),
        targets=np.concatenate(all_targets, axis=0),
    )


def per_class_metrics(
    classes: list[str],
    bundle: PredictionBundle,
    thresholds: dict[str, float],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for class_index, class_name in enumerate(classes):
        labels = [int(value) for value in bundle.targets[:, class_index].tolist()]
        scores = [float(value) for value in bundle.probabilities[:, class_index].tolist()]
        threshold = thresholds[class_name]
        threshold_metrics = binary_metrics(labels, scores, threshold)
        rows.append(
            {
                "class": class_name,
                "num_cases": len(labels),
                "positive_cases": sum(labels),
                "negative_cases": len(labels) - sum(labels),
                "threshold": threshold,
                "auroc": roc_auc(labels, scores),
                "auprc_ap": average_precision(labels, scores),
                "f1": threshold_metrics["f1"],
                "sensitivity": threshold_metrics["sensitivity"],
                "specificity": threshold_metrics["specificity"],
                "balanced_accuracy": threshold_metrics["balanced_accuracy"],
            }
        )
    return rows


def select_thresholds(
    classes: list[str],
    val_bundle: PredictionBundle,
    objective: str,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    thresholds: dict[str, float] = {}
    rows: list[dict[str, float | int | str]] = []
    for class_index, class_name in enumerate(classes):
        labels = [int(value) for value in val_bundle.targets[:, class_index].tolist()]
        scores = [float(value) for value in val_bundle.probabilities[:, class_index].tolist()]
        threshold, metrics = choose_threshold(labels, scores, objective)
        thresholds[class_name] = threshold
        rows.append(
            {
                "class": class_name,
                "threshold": threshold,
                "val_positive_cases": sum(labels),
                "val_negative_cases": len(labels) - sum(labels),
                "val_f1": metrics["f1"],
                "val_sensitivity": metrics["sensitivity"],
                "val_specificity": metrics["specificity"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "selection_objective": objective,
            }
        )
    return thresholds, rows


def summarize_metrics(spec: BackboneSpec, rows: list[dict[str, float | int | str]]) -> dict[str, str]:
    metric_values = {
        "AUROC": [float(row["auroc"]) for row in rows],
        "AUPRC/AP": [float(row["auprc_ap"]) for row in rows],
        "Macro-F1": [float(row["f1"]) for row in rows],
        "Sens": [float(row["sensitivity"]) for row in rows],
        "Spec": [float(row["specificity"]) for row in rows],
        "BACC": [float(row["balanced_accuracy"]) for row in rows],
    }
    return {
        "Backbone": spec.display_name,
        "Pretraining": spec.pretraining,
        "Input type": "spacing 1.5/1.5/3.0 CT NPZ, official CT crop",
        "Supervision": "case-level NoisyOR BCE",
        "Mean/std scope": "over nine abnormalities",
        **{name: mean_std_percent(values) for name, values in metric_values.items()},
    }


def selection_value(metric_name: str, val_loss: float, rows_at_0p5: list[dict[str, float | int | str]]) -> tuple[float, bool]:
    if metric_name == "val_loss":
        return val_loss, False
    lookup = {
        "val_macro_auroc": "auroc",
        "val_macro_auprc_ap": "auprc_ap",
        "val_macro_f1_0p5": "f1",
        "val_macro_bacc_0p5": "balanced_accuracy",
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_prediction_csv(path: Path, classes: list[str], bundle: PredictionBundle, thresholds: dict[str, float]) -> None:
    rows: list[dict[str, str]] = []
    for case_index, volume_id in enumerate(bundle.volume_ids):
        for class_index, class_name in enumerate(classes):
            probability = float(bundle.probabilities[case_index, class_index])
            threshold = thresholds[class_name]
            rows.append(
                {
                    "volume_id": volume_id,
                    "class": class_name,
                    "true_label": str(int(bundle.targets[case_index, class_index])),
                    # Seventeen decimal places preserve the exact Python float
                    # used for the decision and avoid threshold-edge ambiguity
                    # when completion checks reconstruct predicted_label.
                    "probability": fmt_float(probability, 17),
                    "threshold": fmt_float(threshold, 17),
                    "predicted_label": str(int(probability >= threshold)),
                }
            )
    write_csv(path, rows)


def save_per_class_csv(path: Path, spec: BackboneSpec, rows: list[dict[str, float | int | str]]) -> None:
    csv_rows = [
        {
            "Backbone": spec.display_name,
            "Class": str(row["class"]),
            "N": str(row["num_cases"]),
            "Positive Cases": str(row["positive_cases"]),
            "Negative Cases": str(row["negative_cases"]),
            "Threshold": fmt_float(float(row["threshold"])),
            "AUROC (%)": fmt_float(float(row["auroc"]) * 100.0, 2),
            "AUPRC/AP (%)": fmt_float(float(row["auprc_ap"]) * 100.0, 2),
            "F1 (%)": fmt_float(float(row["f1"]) * 100.0, 2),
            "Sensitivity (%)": fmt_float(float(row["sensitivity"]) * 100.0, 2),
            "Specificity (%)": fmt_float(float(row["specificity"]) * 100.0, 2),
            "Balanced Accuracy (%)": fmt_float(float(row["balanced_accuracy"]) * 100.0, 2),
        }
        for row in rows
    ]
    write_csv(path, csv_rows)


def save_threshold_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    csv_rows = [
        {
            "Class": str(row["class"]),
            "Threshold": fmt_float(float(row["threshold"])),
            "Val Positive Cases": str(row["val_positive_cases"]),
            "Val Negative Cases": str(row["val_negative_cases"]),
            "Val F1 (%)": fmt_float(float(row["val_f1"]) * 100.0, 2),
            "Val Sensitivity (%)": fmt_float(float(row["val_sensitivity"]) * 100.0, 2),
            "Val Specificity (%)": fmt_float(float(row["val_specificity"]) * 100.0, 2),
            "Val Balanced Accuracy (%)": fmt_float(float(row["val_balanced_accuracy"]) * 100.0, 2),
            "Selection Objective": str(row["selection_objective"]),
        }
        for row in rows
    ]
    write_csv(path, csv_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--exact-output-dir",
        type=Path,
        help="Write directly to this directory (used by isolated determinism checks).",
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--backbone",
        choices=tuple(BACKBONES) + ("mvit",),
        default="r3d18",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        help=(
            "Optional global gradient-norm clipping threshold, applied after AMP "
            "unscale and before each optimizer step."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value, e.g. --gpu 0")
    parser.add_argument("--device", help="Torch device. Defaults to cuda when available, otherwise cpu.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-output-classes", type=int, default=18)
    parser.add_argument("--pad-shape", nargs=3, type=int, default=[120, 240, 240], metavar=("D", "H", "W"))
    parser.add_argument("--crop-shape", nargs=3, type=int, default=[96, 192, 192], metavar=("D", "H", "W"))
    parser.add_argument("--clip-hu", nargs=2, type=float, default=[-1000.0, 200.0], metavar=("MIN", "MAX"))
    parser.add_argument("--noisy-or-alpha", type=float, default=0.005)
    parser.add_argument(
        "--checkpoint-metric",
        choices=("val_loss", "val_macro_auroc", "val_macro_auprc_ap", "val_macro_f1_0p5", "val_macro_bacc_0p5"),
        default="val_loss",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume training from a last.pt checkpoint, preserving optimizer/scheduler state.",
    )
    parser.add_argument("--threshold-objective", choices=("f1", "balanced_accuracy", "youden"), default="f1")
    parser.add_argument("--max-train-batches", type=int, help="Smoke-test limiter; omit for full training.")
    parser.add_argument("--max-val-batches", type=int, help="Smoke-test limiter; omit for full validation.")
    parser.add_argument("--max-test-batches", type=int, help="Smoke-test limiter; omit for full test evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.backbone = normalize_backbone(args.backbone)
    spec = BACKBONES[args.backbone]
    if args.num_output_classes != len(ALL_PATHOLOGIES):
        raise ValueError(
            "The official comparable protocol requires exactly "
            f"{len(ALL_PATHOLOGIES)} dense output channels before selecting the nine targets; "
            f"got --num-output-classes={args.num_output_classes}"
        )

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    if args.deterministic and os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError(
            "Strict deterministic mode requires PYTHONHASHSEED to be set before "
            f"interpreter startup; launch with PYTHONHASHSEED={args.seed}"
        )
    seed_everything(args.seed, args.deterministic)
    runtime_determinism = deterministic_runtime_state()
    cuda_metadata = cuda_device_metadata(device)
    provenance = source_provenance(spec)
    pretrained_artifacts = pretrained_artifact_provenance(spec)
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

    batch_size = args.batch_size or spec.batch_size
    run_name = args.run_name or f"{spec.key}_case_official_seed{args.seed}"
    output_dir = (
        args.exact_output_dir
        if args.exact_output_dir is not None
        else args.output_root / run_name / f"fold_{args.fold}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pad_shape = parse_shape(args.pad_shape)
    crop_shape = parse_shape(args.crop_shape)
    clip_hu = (float(args.clip_hu[0]), float(args.clip_hu[1]))
    if clip_hu[0] >= clip_hu[1]:
        raise ValueError("--clip-hu MIN must be smaller than MAX")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive when specified")

    train_set = PatchChestCTCaseDataset(
        train_csv,
        classes=CLASSES,
        pad_shape=pad_shape,
        crop_shape=crop_shape,
        clip_hu=clip_hu,
        random_crop=True,
    )
    val_set = PatchChestCTCaseDataset(
        val_csv,
        classes=CLASSES,
        pad_shape=pad_shape,
        crop_shape=crop_shape,
        clip_hu=clip_hu,
        random_crop=False,
    )
    test_set = PatchChestCTCaseDataset(
        test_csv,
        classes=CLASSES,
        pad_shape=pad_shape,
        crop_shape=crop_shape,
        clip_hu=clip_hu,
        random_crop=False,
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

    model = build_model(spec, crop_shape, args.num_output_classes, args.deterministic)
    adapter_metadata = model_protocol_metadata(model)
    frozen_parameter_names = [
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    ]
    if frozen_parameter_names:
        preview = ", ".join(frozen_parameter_names[:8])
        raise RuntimeError(
            "Paper encoder screening requires end-to-end optimization, but found "
            f"{len(frozen_parameter_names)} frozen parameters: {preview}"
        )
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_steps_per_epoch = math.ceil(train_batches / args.gradient_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * optimizer_steps_per_epoch, 1),
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    start_epoch = 1
    resume_checkpoint: dict[str, Any] | None = None
    resume_metadata: dict[str, Any] | None = None
    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        resumed_epoch = int(resume_checkpoint["epoch"])
        start_epoch = resumed_epoch + 1
        resume_metadata = {
            "checkpoint": str(resume_path),
            "checkpoint_epoch": resumed_epoch,
            "continues_from_epoch": start_epoch,
            "restored_optimizer": True,
            "restored_scheduler": True,
            "restored_scaler": "scaler" in resume_checkpoint,
        }

    config = {
        "script": str(Path(__file__).resolve()),
        "split_protocol": {
            "train": str(train_csv),
            "val": str(val_csv),
            "test": str(test_csv),
            "rule": "train is used only for fitting; val selects best checkpoint and thresholds; test is final evaluation only",
        },
        "manifest_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in (train_csv, val_csv, test_csv)
        },
        "backbone": spec.key,
        "backbone_display_name": spec.display_name,
        "backbone_initialization": spec.init_description,
        "pretraining": spec.pretraining,
        "pretrained_artifacts_sha256": pretrained_artifacts,
        "model_protocol_metadata": adapter_metadata,
        "input_processing": {
            "source": "spacing=(1.5,1.5,3.0) NPZ paths from manifest",
            "hu_clip": list(clip_hu),
            "normalization": "(HU - min) / (max - min), official weak/case code style",
            "pad_or_crop_shape": list(pad_shape),
            "orientation": "rot90(k=-1, H/W axes) then left-right flip",
            "train_crop": "random crop to crop_shape",
            "val_test_crop": "center crop to crop_shape",
            "crop_shape": list(crop_shape),
        },
        "supervision": "case-level/study-level",
        "noisy_or_alpha": args.noisy_or_alpha,
        "aggregation": f"NoisyOR: 1 - prod(1 - {args.noisy_or_alpha} * sigmoid(patch_logits)) over D,H,W",
        "official_logits_grid": list(OFFICIAL_LOGITS_GRID),
        "mil_bag_size_per_patient_per_selected_class": OFFICIAL_BAG_SIZE,
        "loss": "official study-level torch.nn.BCELoss on NoisyOR probabilities",
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "encoder_protocol": "trainable backbone optimized end to end",
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "case_label_source": "manifest disease labels",
        "gradient_clipping": (
            None
            if args.max_grad_norm is None
            else {
                "algorithm": "torch.nn.utils.clip_grad_norm_",
                "max_norm": args.max_grad_norm,
                "placement": "after AMP unscale and before optimizer step",
            }
        ),
        "scheduler": "CosineAnnealingLR stepped every optimizer update",
        "checkpoint_selection_metric": args.checkpoint_metric,
        "threshold_selection_rule": f"per-class threshold selected on val by maximizing {args.threshold_objective}; fallback 0.5 when val labels are single-class",
        "test_metric_rule": "AUROC/AUPRC use test probabilities; threshold metrics use fixed val-selected thresholds",
        "mean_std_scope": "over nine abnormalities, not over random seeds",
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
                if args.deterministic and spec.key == "vjepa2_1_b"
                else {}
            ),
        },
        "source_provenance": provenance,
        "device": str(device),
        "cuda_device": cuda_metadata,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": batch_size * args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "amp": use_amp,
        "num_cases": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "smoke_limiters": {
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "max_test_batches": args.max_test_batches,
        },
    }
    if resume_metadata is not None:
        config["resume"] = resume_metadata
    (output_dir / "config.json").write_text(json.dumps(jsonable(config), indent=2), encoding="utf-8")

    log_path = output_dir / "train_log.csv"
    log_rows: list[dict[str, str]] = (
        read_csv_rows(log_path)
        if resume_checkpoint is not None and log_path.exists()
        else []
    )
    best_metric: float | None = None
    if resume_checkpoint is not None and resume_checkpoint.get("best_metric") is not None:
        best_metric = float(resume_checkpoint["best_metric"])
    best_epoch = int(log_rows[-1]["best_epoch"]) if log_rows else 0
    best_higher = args.checkpoint_metric != "val_loss"

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            selected_idx=SELECTED_IDX,
            noisy_or_alpha=args.noisy_or_alpha,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=args.max_train_batches,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        val_bundle = predict(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            selected_idx=SELECTED_IDX,
            noisy_or_alpha=args.noisy_or_alpha,
            use_amp=use_amp,
            desc="val",
            max_batches=args.max_val_batches,
        )
        thresholds_0p5 = {class_name: 0.5 for class_name in CLASSES}
        val_rows_0p5 = per_class_metrics(CLASSES, val_bundle, thresholds_0p5)
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
                    "scaler": scaler.state_dict(),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
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
                "scaler": scaler.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "config": config,
            },
            output_dir / "last.pt",
        )

        log_row = {
            "epoch": str(epoch),
            "train_loss": fmt_float(train_loss),
            "val_loss": fmt_float(val_bundle.loss),
            "val_macro_auroc": fmt_float(mean_or_nan([float(row["auroc"]) for row in val_rows_0p5])),
            "val_macro_auprc_ap": fmt_float(mean_or_nan([float(row["auprc_ap"]) for row in val_rows_0p5])),
            "val_macro_f1_0p5": fmt_float(mean_or_nan([float(row["f1"]) for row in val_rows_0p5])),
            "val_macro_bacc_0p5": fmt_float(mean_or_nan([float(row["balanced_accuracy"]) for row in val_rows_0p5])),
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_metric_value": fmt_float(current_metric),
            "is_best": str(int(is_best)),
            "best_epoch": str(best_epoch),
            "lr": fmt_float(optimizer.param_groups[0]["lr"], 10),
        }
        log_rows.append(log_row)
        write_csv(output_dir / "train_log.csv", log_rows)
        print(json.dumps(log_row, indent=2), flush=True)

    best_path = output_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Training finished without writing best.pt")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    val_bundle = predict(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        selected_idx=SELECTED_IDX,
        noisy_or_alpha=args.noisy_or_alpha,
        use_amp=use_amp,
        desc="val-best",
        max_batches=args.max_val_batches,
    )
    thresholds, threshold_rows = select_thresholds(CLASSES, val_bundle, args.threshold_objective)
    test_bundle = predict(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        selected_idx=SELECTED_IDX,
        noisy_or_alpha=args.noisy_or_alpha,
        use_amp=use_amp,
        desc="test",
        max_batches=args.max_test_batches,
    )
    test_rows = per_class_metrics(CLASSES, test_bundle, thresholds)
    summary_row = summarize_metrics(spec, test_rows)

    save_prediction_csv(output_dir / "val_predictions.csv", CLASSES, val_bundle, thresholds)
    save_prediction_csv(output_dir / "test_predictions.csv", CLASSES, test_bundle, thresholds)
    save_threshold_csv(output_dir / "thresholds.csv", threshold_rows)
    save_per_class_csv(output_dir / "per_class_metrics.csv", spec, test_rows)
    write_csv(output_dir / "summary_metrics.csv", [summary_row])

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
    (output_dir / "config.json").write_text(json.dumps(jsonable(final_config), indent=2), encoding="utf-8")

    print(f"Best epoch: {best_epoch}")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()

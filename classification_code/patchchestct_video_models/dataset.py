"""PatchChestCT dataset that presents CT volumes as videos."""

from __future__ import annotations

import csv
from pathlib import Path
import random
import sys
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification_code.patchchestct_r3d18.preprocessing import (  # noqa: E402
    preprocess_to_display_space,
    resize_display_volume,
)


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

PATCH_GRID_SHAPE = (24, 12, 12)
DISPLAY_SHAPE = (96, 192, 192)
PATCH_GRID_TO_DISPLAY = (4, 16, 16)
PATCH_TARGET_REDUCTIONS = ("adaptive-max", "official-24-to-6")


def _resolve(path: str, base_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return cwd_path
    base_path = base_dir / p
    if base_path.exists():
        return base_path
    return cwd_path


def load_case_labels(row: dict[str, str], classes: Sequence[str]) -> torch.Tensor:
    labels: list[float] = []
    for class_name in classes:
        key = f"{class_name}_label"
        if key not in row:
            raise KeyError(f"Missing case-level label column {key!r} in manifest")
        labels.append(float(row[key]))
    return torch.tensor(labels, dtype=torch.float32)


def load_patch_labels(
    annotation_dir: Path,
    classes: Sequence[str],
    output_shape: Sequence[int] | None = None,
    target_reduction: str = "adaptive-max",
) -> torch.Tensor:
    labels: list[np.ndarray] = []
    for class_name in classes:
        path = annotation_dir / f"{class_name}.npz"
        if path.exists():
            with np.load(path) as data:
                arr = np.asarray(data["arr_0"], dtype=np.float32)
        else:
            arr = np.zeros(PATCH_GRID_SHAPE, dtype=np.float32)
        if arr.shape != PATCH_GRID_SHAPE:
            raise ValueError(f"{path} has shape {arr.shape}; expected {PATCH_GRID_SHAPE}")
        labels.append((arr > 0).astype(np.float32, copy=False))

    target = torch.from_numpy(np.stack(labels, axis=0))
    if output_shape is not None:
        shape = tuple(int(v) for v in output_shape)
        if tuple(target.shape[1:]) != shape:
            if target_reduction == "official-24-to-6":
                if shape != (6, 12, 12):
                    raise ValueError(
                        "official-24-to-6 target reduction requires output_shape=(6, 12, 12), "
                        f"got {shape}"
                    )
                num_classes = target.shape[0]
                target = target.permute(1, 2, 3, 0).contiguous()
                target = (target.reshape(4, 6, 12, 12, num_classes).sum(dim=0) > 0).float()
                target = target.permute(3, 0, 1, 2)
            elif target_reduction == "adaptive-max":
                display_target = target.repeat_interleave(PATCH_GRID_TO_DISPLAY[0], dim=1)
                display_target = display_target.repeat_interleave(PATCH_GRID_TO_DISPLAY[1], dim=2)
                display_target = display_target.repeat_interleave(PATCH_GRID_TO_DISPLAY[2], dim=3)
                if tuple(display_target.shape[1:]) != DISPLAY_SHAPE:
                    raise RuntimeError(
                        f"Expanded patch labels have shape {tuple(display_target.shape[1:])}; expected {DISPLAY_SHAPE}"
                    )
                target = F.adaptive_max_pool3d(display_target[None], output_size=shape)[0]
            else:
                raise ValueError(
                    f"Unsupported patch target reduction {target_reduction!r}; "
                    f"choose one of {PATCH_TARGET_REDUCTIONS}"
                )
    return target.contiguous()


def normalize_hu_to_unit(volume: np.ndarray, clip_hu: tuple[float, float]) -> np.ndarray:
    low, high = clip_hu
    volume = np.nan_to_num(volume, nan=low, posinf=high, neginf=low)
    volume = np.clip(volume, low, high)
    return ((volume - low) / (high - low)).astype(np.float32, copy=False)


def normalize_direct_npz_to_unit(volume: np.ndarray, clip_hu: tuple[float, float]) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=-1.0)
    finite = volume[np.isfinite(volume)]
    if finite.size and float(finite.min()) >= -1.5 and float(finite.max()) <= 1.5:
        return np.clip((volume + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)
    return normalize_hu_to_unit(volume, clip_hu)


def load_video_volume(
    path: Path,
    frames: int,
    image_size: int,
    clip_hu: tuple[float, float],
    allow_unverified_nifti: bool = False,
    input_space: str = "display",
) -> torch.Tensor:
    if input_space == "display":
        display = preprocess_to_display_space(path, allow_unverified_nifti=allow_unverified_nifti)
        video = resize_display_volume(display, (frames, image_size, image_size))
        video = normalize_hu_to_unit(video, clip_hu)
    elif input_space == "direct-npz":
        if path.suffix.lower() != ".npz":
            raise ValueError(f"--input-space direct-npz requires .npz inputs, got {path}")
        with np.load(path) as data:
            arr = np.asarray(data["arr_0"], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"{path} arr_0 has shape {arr.shape}; expected 3D (D,H,W)")
        video = resize_display_volume(arr, (frames, image_size, image_size))
        video = normalize_direct_npz_to_unit(video, clip_hu)
    else:
        raise ValueError(f"Unsupported input_space {input_space!r}")
    return torch.from_numpy(video[None])


class PatchChestCTVideoDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        classes: Sequence[str] = CLASSES,
        frames: int = 16,
        image_size: int = 224,
        clip_hu: tuple[float, float] = (-1000.0, 1000.0),
        mean: Sequence[float] = (0.45, 0.45, 0.45),
        std: Sequence[float] = (0.225, 0.225, 0.225),
        train: bool = False,
        random_horizontal_flip: float = 0.5,
        allow_unverified_nifti: bool = False,
        max_cases: int | None = None,
        input_space: str = "display",
        load_patch_targets: bool = False,
        patch_target_shape: Sequence[int] | None = None,
        patch_target_reduction: str = "adaptive-max",
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.base_dir = self.manifest_csv.parent
        self.classes = list(classes)
        self.frames = int(frames)
        self.image_size = int(image_size)
        self.clip_hu = clip_hu
        self.train = train
        self.random_horizontal_flip = float(random_horizontal_flip)
        self.allow_unverified_nifti = allow_unverified_nifti
        self.input_space = input_space
        self.load_patch_targets = load_patch_targets
        self.patch_target_shape = tuple(int(v) for v in patch_target_shape) if patch_target_shape is not None else None
        if patch_target_reduction not in PATCH_TARGET_REDUCTIONS:
            raise ValueError(
                f"Unsupported patch target reduction {patch_target_reduction!r}; "
                f"choose one of {PATCH_TARGET_REDUCTIONS}"
            )
        self.patch_target_reduction = patch_target_reduction
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1, 1)

        with self.manifest_csv.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        if max_cases is not None:
            self.rows = self.rows[: int(max_cases)]
        if not self.rows:
            raise RuntimeError(f"No rows found in {self.manifest_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def resolve_path(self, path: str) -> Path:
        return _resolve(path, self.base_dir)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_path = self.resolve_path(row["image_path"])
        video = load_video_volume(
            image_path,
            frames=self.frames,
            image_size=self.image_size,
            clip_hu=self.clip_hu,
            allow_unverified_nifti=self.allow_unverified_nifti,
            input_space=self.input_space,
        )
        video = video.repeat(3, 1, 1, 1)
        patch_target = None
        if self.load_patch_targets:
            annotation_dir = self.resolve_path(row["annotation_dir"])
            patch_target = load_patch_labels(
                annotation_dir,
                self.classes,
                self.patch_target_shape,
                self.patch_target_reduction,
            )

        did_flip = self.train and self.random_horizontal_flip > 0.0 and random.random() < self.random_horizontal_flip
        if did_flip:
            video = torch.flip(video, dims=(-1,))
            if patch_target is not None:
                patch_target = torch.flip(patch_target, dims=(-1,))
        video = (video - self.mean) / self.std

        sample: dict[str, object] = {
            "video": video.contiguous(),
            "target": load_case_labels(row, self.classes),
            "volume_id": row["volume_id"],
        }
        if patch_target is not None:
            sample["patch_target"] = patch_target.contiguous()
        return sample


def compute_case_pos_weight(
    dataset: PatchChestCTVideoDataset,
    max_pos_weight: float,
) -> torch.Tensor:
    positives = torch.zeros(len(dataset.classes), dtype=torch.float64)
    for row in dataset.rows:
        for class_index, class_name in enumerate(dataset.classes):
            positives[class_index] += float(row[f"{class_name}_label"])
    total_cases = len(dataset.rows)
    negatives = total_cases - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    return pos_weight.clamp(min=1.0, max=max_pos_weight).float()


def compute_patch_pos_weight(
    dataset: PatchChestCTVideoDataset,
    max_pos_weight: float,
) -> torch.Tensor:
    if dataset.patch_target_shape is None:
        raise ValueError("Patch positive weights require dataset.patch_target_shape")
    positives = torch.zeros(len(dataset.classes), dtype=torch.float64)
    total_cells = 0
    for row in dataset.rows:
        annotation_dir = dataset.resolve_path(row["annotation_dir"])
        labels = load_patch_labels(
            annotation_dir,
            dataset.classes,
            dataset.patch_target_shape,
            dataset.patch_target_reduction,
        )
        positives += labels.sum(dim=(1, 2, 3)).double()
        total_cells += int(labels.shape[1] * labels.shape[2] * labels.shape[3])
    negatives = total_cells - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    return pos_weight.clamp(min=1.0, max=max_pos_weight).float()

"""PatchChestCT CT preprocessing and annotation display-space utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import zoom


DISPLAY_SHAPE = (96, 192, 192)


def _center_crop_or_pad(arr: np.ndarray, target: Sequence[int]) -> np.ndarray:
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


def _raw_dhw_to_display_orientation(arr: np.ndarray) -> np.ndarray:
    """Apply the official axial orientation transform before sizing."""
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.rot90(arr, k=-1, axes=(0, 1))
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.flip(arr, axis=2).copy()
    return arr.astype(np.float32, copy=False)


def _preprocess_npz_to_display_space(path: Path) -> np.ndarray:
    arr = np.load(path)["arr_0"].astype(np.float32) * 1000.0
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.rot90(arr, k=-1, axes=(0, 1))
    arr = _center_crop_or_pad(arr, (192, 192, 96))
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.flip(arr, axis=2).copy()
    return arr.astype(np.float32, copy=False)


def _preprocess_nifti_to_display_space(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Reading NIfTI inputs requires nibabel. Install nibabel or use .npz inputs.") from e

    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    arr = np.transpose(arr, (2, 1, 0))
    arr = _raw_dhw_to_display_orientation(arr)
    return resize_display_volume(arr, DISPLAY_SHAPE)


def preprocess_to_display_space(path: Path, allow_unverified_nifti: bool = False) -> np.ndarray:
    """Transform a CT volume to PatchChestCT display space.

    For official CT-RATE preprocessed `.npz` inputs, this follows
    PatchChestCT's `verify_alignment.py` convention exactly:

    - output shape is `(96, 192, 192)` in `(depth, height, width)`
    - patch annotation `ann[z, y, x]` maps to
      `display[4*z:4*(z+1), 16*y:16*(y+1), 16*x:16*(x+1)]`

    For local `.nii.gz` files, this uses a full-FOV resize fallback after the
    same display orientation transform.  Visual verification is required.
    """
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".npz"):
        return _preprocess_npz_to_display_space(path)
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        if not allow_unverified_nifti:
            raise ValueError(
                "PatchChestCT official alignment expects CT-RATE preprocessed .npz volumes. "
                f"Got NIfTI: {path}. Use the visualization verifier first, then rerun with "
                "--allow-unverified-nifti only for debugging, not for strict experiments."
            )
        return _preprocess_nifti_to_display_space(path)
    raise ValueError(f"Unsupported CT format: {path}")


def resize_display_volume(display: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    target = tuple(int(v) for v in target_shape)
    if tuple(display.shape) == target:
        return display.astype(np.float32, copy=False)
    factors = [target_dim / source_dim for target_dim, source_dim in zip(target, display.shape)]
    return zoom(display, factors, order=1).astype(np.float32, copy=False)


def normalize_hu(volume: np.ndarray, clip_hu: tuple[float, float]) -> np.ndarray:
    volume = np.nan_to_num(volume, nan=clip_hu[0], posinf=clip_hu[1], neginf=clip_hu[0])
    volume = np.clip(volume, clip_hu[0], clip_hu[1])
    volume = (volume - clip_hu[0]) / (clip_hu[1] - clip_hu[0])
    return (volume * 2.0) - 1.0

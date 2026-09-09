"""Pooling operators for mapping dense V-JEPA tokens to PatchChestCT grids."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from classification_code.patchchestct_grid import physical_center_partition


def _logmeanexp_over_partition(
    values: torch.Tensor,
    *,
    dim: int,
    target_size: int,
    temperature: float,
) -> torch.Tensor:
    """Apply stable LogMeanExp to disjoint physical-center bins on one axis."""

    source_size = values.shape[dim]
    intervals = physical_center_partition(source_size, target_size)
    pooled: list[torch.Tensor] = []
    for start, end in intervals:
        region = values.narrow(dim, start, end - start)
        # Centering by the local maximum keeps exp finite even for small tau.
        # ``values`` is float32 before this helper is called, so the complete
        # reduction remains float32 under AMP as well.
        maximum = region.amax(dim=dim, keepdim=True)
        mean_exp = ((region - maximum) / temperature).exp().mean(
            dim=dim,
            keepdim=True,
        )
        pooled.append(maximum + temperature * mean_exp.log())
    return torch.cat(pooled, dim=dim)


def _mean_over_partition(
    values: torch.Tensor,
    *,
    dim: int,
    target_size: int,
) -> torch.Tensor:
    intervals = physical_center_partition(values.shape[dim], target_size)
    return torch.cat(
        [
            values.narrow(dim, start, end - start).mean(dim=dim, keepdim=True)
            for start, end in intervals
        ],
        dim=dim,
    )


def _uniform_interval_size(
    intervals: tuple[tuple[int, int], ...],
) -> int | None:
    sizes = {end - start for start, end in intervals}
    return sizes.pop() if len(sizes) == 1 else None


def _pool_physical_bins_3d(
    logits: torch.Tensor,
    output_shape: tuple[int, int, int],
    temperature: float | None,
) -> torch.Tensor:
    """Shared disjoint-bin implementation for arithmetic mean and LME."""

    values = logits.float()
    depth_bins = physical_center_partition(values.shape[2], output_shape[0])
    height_bins = physical_center_partition(values.shape[3], output_shape[1])
    width_bins = physical_center_partition(values.shape[4], output_shape[2])
    height_group = _uniform_interval_size(height_bins)
    width_group = _uniform_interval_size(width_bins)

    # PatchChestCT's 24x24 -> 12x12 mapping has regular 2x2 planar bins.
    # Reshape those axes and loop only over the six (unequal-sized) depth
    # bins, avoiding a Python loop over all 864 output cells.
    if height_group is not None and width_group is not None:
        batch, channels, _, height, width = values.shape
        if height_group * output_shape[1] != height or width_group * output_shape[2] != width:
            raise AssertionError("Uniform physical-center bins must tile each planar axis")
        outputs: list[torch.Tensor] = []
        for start, end in depth_bins:
            depth_group = end - start
            region = values[:, :, start:end].reshape(
                batch,
                channels,
                depth_group,
                output_shape[1],
                height_group,
                output_shape[2],
                width_group,
            )
            region = region.permute(0, 1, 3, 5, 2, 4, 6).flatten(4)
            if temperature is None:
                pooled = region.mean(dim=-1)
            else:
                maximum = region.amax(dim=-1)
                mean_exp = ((region - maximum.unsqueeze(-1)) / temperature).exp().mean(dim=-1)
                pooled = maximum + temperature * mean_exp.log()
            outputs.append(pooled)
        return torch.stack(outputs, dim=2).contiguous()

    # Generic fallback for other valid source/output grid shapes.  The
    # separable reductions are exactly equivalent within rectangular bins.
    if temperature is None:
        for dim, target_size in ((4, output_shape[2]), (3, output_shape[1]), (2, output_shape[0])):
            values = _mean_over_partition(values, dim=dim, target_size=target_size)
    else:
        for dim, target_size in ((4, output_shape[2]), (3, output_shape[1]), (2, output_shape[0])):
            values = _logmeanexp_over_partition(
                values,
                dim=dim,
                target_size=target_size,
                temperature=temperature,
            )
    return values.contiguous()


def physical_mean_pool3d(
    logits: torch.Tensor,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Average native logits in mutually exclusive physical-center bins."""

    shape = _validate_pool_inputs(logits, output_shape)
    return _pool_physical_bins_3d(logits, shape, temperature=None)


def _validate_pool_inputs(
    logits: torch.Tensor,
    output_shape: Sequence[int],
) -> tuple[int, int, int]:
    if logits.ndim != 5:
        raise ValueError(
            "physical 3-D pooling expects (B,C,D,H,W), "
            f"got {tuple(logits.shape)}"
        )
    shape = tuple(int(value) for value in output_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"output_shape must contain three positive integers, got {output_shape}")
    return shape


def smooth_logmeanexp_pool3d(
    logits: torch.Tensor,
    output_shape: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    """Pool native patch logits into disjoint 3-D bins with LogMeanExp.

    The input and output grids cover the same physical field of view.  Every
    native cell is assigned exactly once via normalized cell centers.  At high
    temperature the operator approaches the per-bin arithmetic mean; at low
    temperature it approaches the per-bin maximum (a smooth OR).

    Computation and output intentionally stay in float32 for numerical
    stability under autocast.  Applying the three separable reductions is
    exactly equivalent to one LogMeanExp over each rectangular 3-D bin.
    """

    shape = _validate_pool_inputs(logits, output_shape)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    return _pool_physical_bins_3d(logits, shape, temperature=temperature)


def _linear_resample_cell_centers(
    values: torch.Tensor,
    *,
    dim: int,
    target_size: int,
) -> torch.Tensor:
    """Linearly sample a uniform source axis at target-cell centers.

    This is the ``align_corners=False`` coordinate rule, implemented as a
    fixed linear interpolation matrix instead of CUDA's generic interpolation
    backward.  Under the strict cuBLAS workspace configuration, the explicit
    matrix form keeps fine-annotation supervision deterministic.
    """

    source_size = int(values.shape[dim])
    if source_size < target_size:
        raise ValueError(
            "cell-center resampling only supports downsampling, got "
            f"{source_size} < {target_size} on dim {dim}"
        )
    positions = (
        (torch.arange(target_size, device="cpu", dtype=torch.float64) + 0.5)
        * (source_size / float(target_size))
        - 0.5
    )
    lower = positions.floor().to(dtype=torch.long).clamp(min=0, max=source_size - 1)
    upper = (lower + 1).clamp(max=source_size - 1)
    upper_weight = positions - lower.to(dtype=positions.dtype)
    rows = torch.arange(target_size, dtype=torch.long)
    matrix = torch.zeros((target_size, source_size), dtype=torch.float64)
    matrix[rows, lower] += 1.0 - upper_weight
    matrix[rows, upper] += upper_weight
    matrix = matrix.to(device=values.device, dtype=values.dtype)
    moved = values.movedim(dim, -1)
    resampled = torch.matmul(moved, matrix.transpose(0, 1))
    return resampled.movedim(-1, dim)


def physical_center_linear_resample3d(
    logits: torch.Tensor,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Resample 3-D logits at normalized physical target-cell centers.

    PatchChestCT's cropped annotation grid samples centers at ``2::4`` in
    depth and ``8::16`` in-plane.  For V-JEPA's native 32x24x24 token grid,
    the ``align_corners=False`` center rule maps exactly to those physical
    locations when producing 24x12x12 logits.
    """

    shape = _validate_pool_inputs(logits, output_shape)
    values = logits.float()
    for dim, target_size in ((4, shape[2]), (3, shape[1]), (2, shape[0])):
        values = _linear_resample_cell_centers(
            values,
            dim=dim,
            target_size=target_size,
        )
    return values.contiguous()

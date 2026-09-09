"""PatchChestCT grid protocols shared by training and dense-token pooling.

The legacy PatchChestCT implementation reduces the 24 annotation planes with
``reshape(B, 4, 6, ...).sum(dim=1)``.  That groups planes by their index modulo
six.  ``anatomical_grid_v2_6x12x12`` instead assigns consecutive annotation
planes to consecutive output cells, matching the physical ordering of the CT
and V-JEPA token grids.
"""

from __future__ import annotations

import torch


LEGACY_OFFICIAL_GRID = "legacy_official_grid"
ANATOMICAL_GRID_V2 = "anatomical_grid_v2_6x12x12"
PATCH_GRID_PROTOCOLS = (LEGACY_OFFICIAL_GRID, ANATOMICAL_GRID_V2)


def physical_center_partition(
    source_size: int,
    target_size: int,
) -> tuple[tuple[int, int], ...]:
    """Partition a uniform source axis by normalized physical cell centers.

    Each source cell is assigned exactly once to the target cell containing
    its normalized center.  Returned intervals are half-open ``(start, end)``
    and therefore can be used directly as slices.  The source and target axes
    are assumed to cover the same physical field of view.
    """

    source_size = int(source_size)
    target_size = int(target_size)
    if source_size <= 0 or target_size <= 0:
        raise ValueError(
            f"source_size and target_size must be positive, got "
            f"{source_size} and {target_size}"
        )
    if source_size < target_size:
        raise ValueError(
            "physical_center_partition requires source_size >= target_size "
            f"to avoid empty target cells, got {source_size} < {target_size}"
        )

    # floor(((index + 0.5) / source_size) * target_size), written with
    # integer arithmetic so boundary behavior is exact and reproducible.
    assignments = tuple(
        min(((2 * index + 1) * target_size) // (2 * source_size), target_size - 1)
        for index in range(source_size)
    )
    intervals: list[tuple[int, int]] = []
    for target_index in range(target_size):
        members = [
            source_index
            for source_index, assigned_target in enumerate(assignments)
            if assigned_target == target_index
        ]
        if not members:
            raise AssertionError(
                f"Target cell {target_index} is empty for {source_size}->{target_size}"
            )
        start, end = members[0], members[-1] + 1
        if members != list(range(start, end)):
            raise AssertionError("Physical-center assignments must form contiguous intervals")
        intervals.append((start, end))

    if intervals[0][0] != 0 or intervals[-1][1] != source_size:
        raise AssertionError("Physical-center partition does not cover the full source axis")
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise AssertionError("Physical-center partition intervals overlap or leave gaps")
    if sum(end - start for start, end in intervals) != source_size:
        raise AssertionError("Physical-center partition must assign every source cell exactly once")
    return tuple(intervals)


def reduce_patch_target_24_to_6(
    target24: torch.Tensor,
    protocol: str = LEGACY_OFFICIAL_GRID,
) -> torch.Tensor:
    """Reduce ``(B,C,24,12,12)`` binary annotations to ``(B,C,6,12,12)``."""

    if target24.ndim != 5:
        raise ValueError(
            f"Expected patch target shape (B,C,24,12,12), got {tuple(target24.shape)}"
        )
    b, c, z, h, w = target24.shape
    if (z, h, w) != (24, 12, 12):
        raise ValueError(
            f"Expected patch target shape (B,C,24,12,12), got {tuple(target24.shape)}"
        )

    if protocol == LEGACY_OFFICIAL_GRID:
        # Exact compatibility with PatchChestCT train_grounding.py:
        # annotations.reshape(B, 4, 6, 12, 12, C).sum(dim=1) > 0
        target = target24.permute(0, 2, 3, 4, 1).contiguous()
        target = target.reshape(b, 4, 6, 12, 12, c).sum(dim=1) > 0
        return target.permute(0, 4, 1, 2, 3).contiguous().float()

    if protocol == ANATOMICAL_GRID_V2:
        intervals = physical_center_partition(source_size=24, target_size=6)
        # For 24->6 these are six consecutive groups of four annotation
        # planes.  ``amax`` implements the annotations' any-positive rule.
        target = torch.stack(
            [target24[:, :, start:end].amax(dim=2) for start, end in intervals],
            dim=2,
        )
        return (target > 0).float()

    valid = ", ".join(PATCH_GRID_PROTOCOLS)
    raise ValueError(f"Unknown patch grid protocol {protocol!r}; choose one of: {valid}")


# These invariants are part of the v2 protocol, not model-specific tuning.
assert physical_center_partition(24, 6) == (
    (0, 4),
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
)
assert physical_center_partition(32, 6) == (
    (0, 5),
    (5, 11),
    (11, 16),
    (16, 21),
    (21, 27),
    (27, 32),
)

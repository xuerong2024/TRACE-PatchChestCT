from __future__ import annotations

import unittest

import torch

from classification_code.patchchestct_grid import (
    ANATOMICAL_GRID_V2,
    LEGACY_OFFICIAL_GRID,
    physical_center_partition,
    reduce_patch_target_24_to_6,
)


class PhysicalCenterPartitionTest(unittest.TestCase):
    def test_native_vjepa_depth_32_to_6_has_expected_disjoint_bins(self) -> None:
        intervals = physical_center_partition(32, 6)
        self.assertEqual(
            intervals,
            ((0, 5), (5, 11), (11, 16), (16, 21), (21, 27), (27, 32)),
        )
        members = [index for start, end in intervals for index in range(start, end)]
        self.assertEqual(members, list(range(32)))
        self.assertEqual(len(members), len(set(members)))

    def test_annotation_depth_24_to_6_is_six_groups_of_four(self) -> None:
        self.assertEqual(
            physical_center_partition(24, 6),
            ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)),
        )

    def test_rejects_empty_target_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_size >= target_size"):
            physical_center_partition(5, 6)


class PatchTargetReductionTest(unittest.TestCase):
    @staticmethod
    def depth_identity_target() -> torch.Tensor:
        target = torch.zeros(1, 24, 24, 12, 12)
        for depth in range(24):
            target[0, depth, depth, 0, 0] = 1.0
        return target

    def test_legacy_protocol_preserves_modulo_six_grouping(self) -> None:
        reduced = reduce_patch_target_24_to_6(
            self.depth_identity_target(),
            protocol=LEGACY_OFFICIAL_GRID,
        )
        self.assertEqual(tuple(reduced.shape), (1, 24, 6, 12, 12))
        for source_depth in range(24):
            positive_depths = torch.nonzero(
                reduced[0, source_depth, :, 0, 0], as_tuple=False
            ).flatten().tolist()
            self.assertEqual(positive_depths, [source_depth % 6])

    def test_anatomical_v2_uses_consecutive_groups(self) -> None:
        reduced = reduce_patch_target_24_to_6(
            self.depth_identity_target(),
            protocol=ANATOMICAL_GRID_V2,
        )
        self.assertEqual(tuple(reduced.shape), (1, 24, 6, 12, 12))
        for source_depth in range(24):
            positive_depths = torch.nonzero(
                reduced[0, source_depth, :, 0, 0], as_tuple=False
            ).flatten().tolist()
            self.assertEqual(positive_depths, [source_depth // 4])

    def test_legacy_matches_historical_reshape_exactly(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        target = torch.randint(0, 2, (2, 9, 24, 12, 12), generator=generator).float()
        expected = target.permute(0, 2, 3, 4, 1).contiguous()
        expected = (expected.reshape(2, 4, 6, 12, 12, 9).sum(dim=1) > 0).float()
        expected = expected.permute(0, 4, 1, 2, 3).contiguous()
        actual = reduce_patch_target_24_to_6(target, protocol=LEGACY_OFFICIAL_GRID)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_unknown_protocol_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown patch grid protocol"):
            reduce_patch_target_24_to_6(
                torch.zeros(1, 1, 24, 12, 12),
                protocol="not-a-grid",
            )


if __name__ == "__main__":
    unittest.main()

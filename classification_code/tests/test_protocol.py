from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from classification_code.patchchestct_vjepa2_1.model import VJEPA21OfficialPatchClassifier
from classification_code.train_patchchestct_official_patch_fold0 import (
    case_target_from_fine_grid,
    dice_at_threshold,
    load_high_res_annotation_mask,
    mct_localization_scores,
)


class CropDerivedCaseTargetTest(unittest.TestCase):
    def test_spatial_or_is_applied_independently_per_class(self) -> None:
        target = np.zeros((3, 24, 12, 12), dtype=np.float32)
        target[0, 2, 3, 4] = 1.0
        target[2, 23, 11, 11] = 1.0
        actual = case_target_from_fine_grid(target)
        np.testing.assert_array_equal(actual, np.asarray([1.0, 0.0, 1.0]))

    def test_invalid_fine_grid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fine target shape"):
            case_target_from_fine_grid(np.zeros((3, 6, 12, 12), dtype=np.float32))


class AnnotationDirectoryValidationTest(unittest.TestCase):
    def test_missing_annotation_directory_is_not_silently_all_negative(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing-volume"
            with self.assertRaisesRegex(FileNotFoundError, "zenodo.org/records/19707049"):
                load_high_res_annotation_mask(missing, ["atelectasis"])

    def test_missing_disease_file_inside_valid_directory_remains_negative(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            mask = load_high_res_annotation_mask(
                Path(temporary_directory),
                ["atelectasis"],
            )
            self.assertEqual(mask.shape, (1, 96, 192, 192))
            self.assertEqual(float(mask.sum()), 0.0)


class EndToEndFineTuningProtocolTest(unittest.TestCase):
    def test_encoder_trains_with_parent_model(self) -> None:
        model = VJEPA21OfficialPatchClassifier.__new__(VJEPA21OfficialPatchClassifier)
        nn.Module.__init__(model)
        model.backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        model.classifier = nn.Linear(4, 2)

        model.train(True)

        self.assertTrue(model.training)
        self.assertTrue(model.classifier.training)
        self.assertTrue(model.backbone.training)
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))


class FineToCoarseLocalizationTest(unittest.TestCase):
    def test_uses_fine_derived_logits_instead_of_direct_coarse_logits(self) -> None:
        direct = torch.full((1, 3, 6, 12, 12), -10.0)
        fine_derived = torch.full((1, 3, 6, 12, 12), 2.0)
        actual = mct_localization_scores(
            direct,
            {"fine_to_coarse_logits": fine_derived},
            selected_idx=[0, 2],
            mode="fine-to-coarse",
        )
        torch.testing.assert_close(actual, fine_derived[:, [0, 2]].sigmoid())


class ValidationThresholdDiceTest(unittest.TestCase):
    def test_fixed_threshold_is_not_reoptimized_on_test(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.4, 0.6, 0.1]
        dsc, counts = dice_at_threshold(labels, scores, threshold=0.5)
        self.assertAlmostEqual(dsc, 0.5)
        self.assertEqual(counts, {"tp": 1.0, "fp": 1.0, "fn": 1.0, "tn": 1.0})


if __name__ == "__main__":
    unittest.main()

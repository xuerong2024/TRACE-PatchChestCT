from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from classification_code.patchchestct_vjepa2_1.model import (
    VJEPA21OfficialPatchClassifier,
    freeze_encoder,
)
from classification_code.train_patchchestct_official_patch_fold0 import (
    case_target_from_fine_grid,
    dice_at_threshold,
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


class FrozenEncoderProtocolTest(unittest.TestCase):
    def test_encoder_stays_eval_when_parent_enters_train_mode(self) -> None:
        model = VJEPA21OfficialPatchClassifier.__new__(VJEPA21OfficialPatchClassifier)
        nn.Module.__init__(model)
        model.backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        model.classifier = nn.Linear(4, 2)
        freeze_encoder(model.backbone)

        model.train(True)

        self.assertTrue(model.training)
        self.assertTrue(model.classifier.training)
        self.assertFalse(model.backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))


class ValidationThresholdDiceTest(unittest.TestCase):
    def test_fixed_threshold_is_not_reoptimized_on_test(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.4, 0.6, 0.1]
        dsc, counts = dice_at_threshold(labels, scores, threshold=0.5)
        self.assertAlmostEqual(dsc, 0.5)
        self.assertEqual(counts, {"tp": 1.0, "fp": 1.0, "fn": 1.0, "tn": 1.0})


if __name__ == "__main__":
    unittest.main()

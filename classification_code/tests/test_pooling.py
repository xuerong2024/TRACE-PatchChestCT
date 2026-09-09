from __future__ import annotations

import unittest

import torch
from torch import nn

from classification_code.patchchestct_grid import physical_center_partition
from classification_code.patchchestct_pooling import (
    physical_mean_pool3d,
    smooth_logmeanexp_pool3d,
)


class PatchChestCTPoolingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.logits = torch.randn(2, 3, 32, 24, 24)
        self.output_shape = (6, 12, 12)

    def _reference(self, temperature: float | None) -> torch.Tensor:
        depth = physical_center_partition(32, 6)
        height = physical_center_partition(24, 12)
        width = physical_center_partition(24, 12)
        result = torch.empty(2, 3, 6, 12, 12)
        for z_out, (z0, z1) in enumerate(depth):
            for y_out, (y0, y1) in enumerate(height):
                for x_out, (x0, x1) in enumerate(width):
                    region = self.logits[:, :, z0:z1, y0:y1, x0:x1].flatten(2)
                    if temperature is None:
                        pooled = region.mean(dim=-1)
                    else:
                        maximum = region.amax(dim=-1)
                        pooled = maximum + temperature * (
                            (region - maximum.unsqueeze(-1)) / temperature
                        ).exp().mean(dim=-1).log()
                    result[:, :, z_out, y_out, x_out] = pooled
        return result

    def test_physical_mean_matches_direct_3d_reference(self) -> None:
        actual = physical_mean_pool3d(self.logits, self.output_shape)
        torch.testing.assert_close(actual, self._reference(None), rtol=0.0, atol=1e-7)

    def test_smooth_or_matches_direct_3d_reference(self) -> None:
        temperature = 0.7
        actual = smooth_logmeanexp_pool3d(
            self.logits,
            self.output_shape,
            temperature,
        )
        torch.testing.assert_close(
            actual,
            self._reference(temperature),
            rtol=0.0,
            atol=1e-7,
        )

    def test_native_linear_then_mean_equals_mean_then_same_linear(self) -> None:
        hidden = torch.randn(2, 7, 32, 24, 24)
        classifier = nn.Linear(7, 3)
        native_logits = classifier(hidden.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        logits_then_pool = physical_mean_pool3d(native_logits, self.output_shape)
        features_then_linear = classifier(
            physical_mean_pool3d(hidden, self.output_shape).permute(0, 2, 3, 4, 1)
        ).permute(0, 4, 1, 2, 3)
        torch.testing.assert_close(logits_then_pool, features_then_linear, rtol=1e-5, atol=2e-7)

    def test_float16_input_is_reduced_in_float32_with_finite_gradients(self) -> None:
        logits = self.logits[:1, :1].half().requires_grad_(True)
        output = smooth_logmeanexp_pool3d(logits, self.output_shape, 0.01)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_constant_logits_are_fixed_points(self) -> None:
        values = torch.full((1, 2, 32, 24, 24), 3.25)
        for temperature in (0.01, 1.0, 100.0):
            actual = smooth_logmeanexp_pool3d(values, self.output_shape, temperature)
            torch.testing.assert_close(actual, torch.full_like(actual, 3.25))

    def test_invalid_temperature_is_rejected(self) -> None:
        for temperature in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                smooth_logmeanexp_pool3d(self.logits, self.output_shape, temperature)


if __name__ == "__main__":
    unittest.main()

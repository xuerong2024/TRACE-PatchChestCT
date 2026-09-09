"""Deterministic replacements for CUDA pooling operations used by case models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _triple(value: int | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    return tuple(int(item) for item in value)


class _DeterministicMaxPool3dFunction(torch.autograd.Function):
    """Use CUDA for the forward pass and deterministic CPU scatter for backward."""

    @staticmethod
    def forward(
        ctx: object,
        x: torch.Tensor,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
        ceil_mode: bool,
        cuda_scatter_backward: bool,
    ) -> torch.Tensor:
        output, indices = F.max_pool3d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
        ctx.save_for_backward(indices)  # type: ignore[attr-defined]
        ctx.input_shape = tuple(x.shape)  # type: ignore[attr-defined]
        ctx.input_device = x.device  # type: ignore[attr-defined]
        ctx.cuda_scatter_backward = cuda_scatter_backward  # type: ignore[attr-defined]
        return output

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        (indices,) = ctx.saved_tensors  # type: ignore[attr-defined]
        input_shape = ctx.input_shape  # type: ignore[attr-defined]
        input_device = ctx.input_device  # type: ignore[attr-defined]
        batch_size, channels = input_shape[:2]

        if ctx.cuda_scatter_backward:  # type: ignore[attr-defined]
            # When pooling windows do not overlap, every saved input index is
            # unique. Assignment therefore needs no atomic accumulation and is
            # deterministic on CUDA.
            grad_input = torch.zeros(
                input_shape,
                dtype=grad_output.dtype,
                device=input_device,
            )
            grad_input.view(batch_size, channels, -1).scatter_(
                2,
                indices.view(batch_size, channels, -1),
                grad_output.contiguous().view(batch_size, channels, -1),
            )
            return grad_input, None, None, None, None, None, None

        grad_output_cpu = grad_output.detach().to(device="cpu").contiguous()
        indices_cpu = indices.detach().to(device="cpu").contiguous()
        grad_input_cpu = torch.zeros(input_shape, dtype=grad_output.dtype, device="cpu")
        grad_input_cpu.view(batch_size, channels, -1).scatter_add_(
            2,
            indices_cpu.view(batch_size, channels, -1),
            grad_output_cpu.view(batch_size, channels, -1),
        )
        return grad_input_cpu.to(device=input_device), None, None, None, None, None, None


class DeterministicMaxPool3d(nn.Module):
    """Drop-in replacement for ``nn.MaxPool3d`` when indices are not returned."""

    def __init__(self, pool: nn.MaxPool3d) -> None:
        super().__init__()
        if pool.return_indices:
            raise ValueError("DeterministicMaxPool3d does not support return_indices=True")
        stride = pool.kernel_size if pool.stride is None else pool.stride
        self.kernel_size = _triple(pool.kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(pool.padding)
        self.dilation = _triple(pool.dilation)
        self.ceil_mode = bool(pool.ceil_mode)
        effective_kernel = tuple(
            dilation * (kernel - 1) + 1
            for kernel, dilation in zip(self.kernel_size, self.dilation)
        )
        self.cuda_scatter_backward = all(
            stride >= kernel
            for stride, kernel in zip(self.stride, effective_kernel)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return F.max_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )
        return _DeterministicMaxPool3dFunction.apply(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.cuda_scatter_backward,
        )


class DeterministicGlobalAvgPool3d(nn.Module):
    """Deterministic equivalent of adaptive average pooling to one output cell."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3, 4), keepdim=True)


class _DeterministicAvgPool3dFunction(torch.autograd.Function):
    """CUDA average-pool forward with a deterministic CPU backward."""

    @staticmethod
    def forward(
        ctx: object,
        x: torch.Tensor,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: int | None,
    ) -> torch.Tensor:
        ctx.input_shape = tuple(x.shape)  # type: ignore[attr-defined]
        ctx.input_device = x.device  # type: ignore[attr-defined]
        ctx.kernel_size = kernel_size  # type: ignore[attr-defined]
        ctx.stride = stride  # type: ignore[attr-defined]
        ctx.padding = padding  # type: ignore[attr-defined]
        ctx.ceil_mode = ceil_mode  # type: ignore[attr-defined]
        ctx.count_include_pad = count_include_pad  # type: ignore[attr-defined]
        ctx.divisor_override = divisor_override  # type: ignore[attr-defined]
        return F.avg_pool3d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        cpu_dtype = (
            torch.float32
            if grad_output.dtype in {torch.float16, torch.bfloat16}
            else grad_output.dtype
        )
        with torch.enable_grad():
            cpu_input = torch.zeros(
                ctx.input_shape,  # type: ignore[attr-defined]
                dtype=cpu_dtype,
                device="cpu",
                requires_grad=True,
            )
            cpu_output = F.avg_pool3d(
                cpu_input,
                kernel_size=ctx.kernel_size,  # type: ignore[attr-defined]
                stride=ctx.stride,  # type: ignore[attr-defined]
                padding=ctx.padding,  # type: ignore[attr-defined]
                ceil_mode=ctx.ceil_mode,  # type: ignore[attr-defined]
                count_include_pad=ctx.count_include_pad,  # type: ignore[attr-defined]
                divisor_override=ctx.divisor_override,  # type: ignore[attr-defined]
            )
            grad_input = torch.autograd.grad(
                cpu_output,
                cpu_input,
                grad_output.detach().to(device="cpu", dtype=cpu_dtype),
            )[0]
        return (
            grad_input.to(device=ctx.input_device, dtype=grad_output.dtype),  # type: ignore[attr-defined]
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DeterministicAvgPool3d(nn.Module):
    """Drop-in deterministic replacement for ``nn.AvgPool3d``."""

    def __init__(self, pool: nn.AvgPool3d) -> None:
        super().__init__()
        stride = pool.kernel_size if pool.stride is None else pool.stride
        self.kernel_size = _triple(pool.kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(pool.padding)
        self.ceil_mode = bool(pool.ceil_mode)
        self.count_include_pad = bool(pool.count_include_pad)
        self.divisor_override = pool.divisor_override

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return F.avg_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
            )
        return _DeterministicAvgPool3dFunction.apply(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )


def replace_max_pool3d(module: nn.Module) -> int:
    """Recursively replace all parameter-free MaxPool3d modules in-place."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MaxPool3d):
            setattr(module, name, DeterministicMaxPool3d(child))
            replaced += 1
        else:
            replaced += replace_max_pool3d(child)
    return replaced


def replace_adaptive_global_avg_pool3d(module: nn.Module) -> int:
    """Replace AdaptiveAvgPool3d(1) modules with a deterministic reduction."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool3d):
            output_size = _triple(child.output_size)
            if output_size != (1, 1, 1):
                raise ValueError(
                    "Only AdaptiveAvgPool3d(output_size=1) has a generic deterministic replacement; "
                    f"got {output_size}"
                )
            setattr(module, name, DeterministicGlobalAvgPool3d())
            replaced += 1
        else:
            replaced += replace_adaptive_global_avg_pool3d(child)
    return replaced


def replace_avg_pool3d(module: nn.Module) -> int:
    """Recursively replace fixed-kernel AvgPool3d modules in-place."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AvgPool3d):
            setattr(module, name, DeterministicAvgPool3d(child))
            replaced += 1
        else:
            replaced += replace_avg_pool3d(child)
    return replaced

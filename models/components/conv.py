# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convolutional layer wrappers with built-in padding and normalization."""

import math
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from .norm import ConvLayerNorm


CONV_NORMALIZATIONS = frozenset([
    'none', 'weight_norm', 'spectral_norm',
    'time_layer_norm', 'layer_norm', 'time_group_norm',
])


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    """Wrap a module with weight_norm or spectral_norm if requested."""
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    return module


def get_norm_module(
    module: nn.Module,
    causal: bool = False,
    norm: str = 'none',
    **norm_kwargs,
) -> nn.Module:
    """Return the appropriate post-conv normalization module.

    Args:
        module: The conv module whose output will be normalized.
        causal: If True, only causal-safe normalizations are allowed.
        norm: One of CONV_NORMALIZATIONS.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm does not support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    return nn.Identity()


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def get_extra_padding_for_conv1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0,
) -> int:
    """Compute extra right-padding so the last window is fully covered."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0,
) -> torch.Tensor:
    """Pad the right side of x so that no time steps are dropped by Conv1d.

    Without this, strided convolutions may silently discard the last frame,
    making it impossible to reconstruct the original length via transposed conv.
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = 'zero',
    value: float = 0.0,
) -> torch.Tensor:
    """F.pad wrapper that handles reflect-padding on very short inputs.

    PyTorch's reflect pad requires the input length to exceed the pad size.
    When the input is shorter, we insert extra zero-padding first, apply
    reflect, then remove the extra zeros.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]) -> torch.Tensor:
    """Remove left and right padding from a 1-D (last-dim) tensor."""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


# ---------------------------------------------------------------------------
# Normalized conv wrappers
# ---------------------------------------------------------------------------

class NormConv1d(nn.Module):
    """Conv1d + optional normalization in a single module."""

    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        return self.norm(self.conv(x))


class NormConvTranspose1d(nn.Module):
    """ConvTranspose1d + optional normalization in a single module."""

    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        return self.norm(self.convtr(x))


# ---------------------------------------------------------------------------
# Streamable conv wrappers (handle asymmetric / causal padding automatically)
# ---------------------------------------------------------------------------

class SConv1d(nn.Module):
    """Conv1d with automatic causal or symmetric padding and normalization.

    Padding is computed so that the output length equals ceil(T / stride),
    which guarantees perfect reconstruction when paired with SConvTranspose1d.

    Args:
        in_channels, out_channels, kernel_size: Standard Conv1d args.
        stride: Convolution stride (downsampling factor).
        dilation: Convolution dilation.
        causal: If True, only left (past) padding is applied.
        norm: Normalization type (see CONV_NORMALIZATIONS).
        pad_mode: Padding mode passed to pad1d ('reflect', 'zero', etc.).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = 'none',
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = 'reflect',
    ):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                f'SConv1d: stride={stride} and dilation={dilation} are both > 1 '
                f'(kernel_size={kernel_size}). This is unusual.'
            )
        self.conv = NormConv1d(
            in_channels, out_channels, kernel_size, stride,
            dilation=dilation, groups=groups, bias=bias,
            causal=causal, norm=norm, norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        # Effective kernel size after dilation
        kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with automatic padding removal and normalization.

    Mirrors SConv1d: removes the padding that SConv1d added, so the
    reconstructed signal matches the original length exactly.

    Args:
        in_channels, out_channels, kernel_size, stride: Standard args.
        causal: Must match the paired SConv1d setting.
        norm: Normalization type.
        trim_right_ratio: In causal mode, fraction of padding trimmed from
            the right (1.0 = trim all from right).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = 'none',
        trim_right_ratio: float = 1.0,
        norm_kwargs: tp.Dict[str, tp.Any] = {},
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels, out_channels, kernel_size, stride,
            causal=causal, norm=norm, norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions."
        assert 0.0 <= self.trim_right_ratio <= 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        y = self.convtr(x)
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
        return unpad1d(y, (padding_left, padding_right))

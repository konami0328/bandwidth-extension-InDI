# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Normalization modules for convolutional layers."""

import typing as tp

import einops
import torch
from torch import nn


class ConvLayerNorm(nn.LayerNorm):
    """Channel-first LayerNorm compatible with convolutional feature maps.

    Standard LayerNorm expects channels last (B, T, C), but conv layers
    produce channels first (B, C, T). This wrapper handles the permutation
    transparently so it can be dropped into any conv stack.
    """

    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move channels to last dim, normalize, then restore original layout
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return x

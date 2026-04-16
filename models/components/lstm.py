# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM wrapper for use inside convolutional encoder-decoder stacks."""

from torch import nn


class SLSTM(nn.Module):
    """Stateless LSTM that operates on channel-first (B, C, T) tensors.

    Handles the (B, C, T) <-> (T, B, C) permutation required by nn.LSTM
    internally, so it can be inserted directly into a Sequential conv stack.

    Args:
        dimension: Number of input and output channels.
        num_layers: Number of stacked LSTM layers.
        skip: If True, adds a residual connection (output = lstm_out + input).
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        # x: (B, C, T) -> lstm expects (T, B, C)
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        # (T, B, C) -> (B, C, T)
        y = y.permute(1, 2, 0)
        return y

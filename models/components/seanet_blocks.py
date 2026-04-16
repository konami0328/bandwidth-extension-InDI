# Copyright (c) Meta Platforms, Inc. and affiliates. (original SEANet architecture)
# Modifications: U-Net skip connections and time conditioning added for this project.

"""SEANet encoder-decoder building blocks.

This module provides the shared backbone used by both model variants:
  - Baseline (single-step regressor): no time conditioning
  - InDI (time-conditioned): each ResBlock receives a time embedding

The key design difference is controlled by the `time_conditioned` flag on
SEANetEncoder and SEANetDecoder. When False, the modules behave exactly like
the original EnCodec SEANet. When True, a sinusoidal TimeEmbedding is injected
into every ResBlock via a small MLP, following the standard diffusion-model
conditioning pattern.

Architecture overview (for a 4-ratio stack with ratios=[8,5,4,2]):

    Input (B, 1, T)
        │
        Conv1d k=7          ← initial projection to n_filters channels
        │
      EncBlock s=2          ← ResBlock + strided Conv1d (downsampling)
      EncBlock s=4
      EncBlock s=5
      EncBlock s=8
        │
        LSTM (2 layers)
        │
        Conv1d k=7          ← project to bottleneck dimension
        │
     [bottleneck latent]
        │
        Conv1d k=7          ← expand from bottleneck
        │
        LSTM (2 layers)
        │
      DecBlock s=8          ← TransposedConv (upsampling) + ResBlock
      DecBlock s=5          ← skip connection from matching EncBlock injected here
      DecBlock s=4
      DecBlock s=2
        │
        Conv1d k=7          ← final projection to 1 channel
        │
    Output (B, 1, T)

Skip connections (U-Net style) are concatenated channel-wise just before each
transposed conv, then projected back to the expected channel count with a 1×1 conv.
"""

import typing as tp

import numpy as np
import torch
import torch.nn as nn

from .conv import SConv1d, SConvTranspose1d
from .lstm import SLSTM


# ---------------------------------------------------------------------------
# Time embedding (used only by the InDI time-conditioned variant)
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding, similar to Transformer positional encoding.

    Maps a scalar degradation level t ∈ [0, 1] to a fixed-size embedding
    vector that can be injected into the network to condition it on the
    current position along the InDI degradation path.

    The raw sinusoidal features are then projected through a small 2-layer MLP
    (with SiLU activation) to give the network capacity to learn a non-linear
    transformation of t.

    Args:
        emb_dim: Dimension of the output embedding vector.
        max_period: Controls the frequency range of the sinusoids.
    """

    def __init__(self, emb_dim: int = 256, max_period: int = 10000):
        super().__init__()
        self.emb_dim = emb_dim

        half_dim = emb_dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / half_dim
        # Precompute inverse frequencies; registered as buffer so they move
        # to the correct device automatically with .to(device).
        self.register_buffer("inv_freq", 1.0 / (max_period ** exponents))

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Degradation level, shape (B,) or (B, 1), values in [0, 1].
        Returns:
            Time embedding, shape (B, emb_dim).
        """
        if t.dim() == 1:
            t = t[:, None]                                   # (B,) -> (B, 1)
        sinusoid_input = t * self.inv_freq[None, :]          # (B, emb_dim/2)
        emb = torch.cat(
            [torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1
        )                                                    # (B, emb_dim)
        return self.proj(emb)                                # (B, emb_dim)


# ---------------------------------------------------------------------------
# Residual block (shared by encoder and decoder)
# ---------------------------------------------------------------------------

class SEANetResnetBlock(nn.Module):
    """SEANet residual block: two dilated convolutions with a skip connection.

    Structure (for kernel_sizes=[3,1], dilations=[d,1]):
        x ──────────────────────────── shortcut ──┐
        │                                         │
        ELU → Conv(k=3, d=d) → ELU → Conv(k=1) ──┤
                                                  +
                                               output

    When `time_conditioned=True`, the time embedding is added to x (as a
    channel-wise bias) before the residual computation, following the
    standard conditioning pattern used in diffusion models.

    Args:
        dim: Number of input and output channels.
        kernel_sizes: Kernel sizes for the two convolutions.
        dilations: Dilation values for the two convolutions.
        activation: Name of the activation function (must be in torch.nn).
        norm: Normalization type for SConv1d.
        causal: Whether to use causal convolutions.
        compress: Channel compression ratio inside the block (from Demucs v3).
        true_skip: If True, shortcut is identity; otherwise a 1×1 conv.
        time_conditioned: If True, accept and inject a time embedding.
        time_emb_dim: Dimensionality of the time embedding (must match
            TimeEmbedding.emb_dim when time_conditioned=True).
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = 'reflect',
        compress: int = 2,
        true_skip: bool = True,
        time_conditioned: bool = False,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), \
            "kernel_sizes and dilations must have the same length."

        act = getattr(nn, activation)
        hidden = dim // compress

        block = []
        for i, (ks, dil) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(in_chs, out_chs, kernel_size=ks, dilation=dil,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)

        self.shortcut: nn.Module = (
            nn.Identity() if true_skip
            else SConv1d(dim, dim, kernel_size=1, norm=norm,
                         norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        )

        # Time conditioning: project time embedding to channel bias
        self.time_conditioned = time_conditioned
        if time_conditioned:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Feature map, shape (B, dim, T).
            time_emb: Time embedding, shape (B, time_emb_dim).
                      Required when time_conditioned=True, ignored otherwise.
        """
        if self.time_conditioned:
            assert time_emb is not None, \
                "time_emb must be provided when time_conditioned=True."
            # Project to channel dim and broadcast over time axis
            bias = self.time_mlp(time_emb).unsqueeze(-1)   # (B, dim, 1)
            x = x + bias
        return self.shortcut(x) + self.block(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SEANetEncoder(nn.Module):
    """SEANet hierarchical downsampling encoder with optional time conditioning.

    Progressively downsamples the input waveform through a series of strided
    convolutions, each preceded by residual blocks. A two-layer LSTM at the
    end captures long-range temporal dependencies before the final bottleneck
    projection.

    Skip feature maps from each downsampling conv are returned alongside the
    bottleneck latent for use by the paired SEANetDecoder (U-Net style).

    Args:
        channels: Number of input audio channels (1 for mono).
        dimension: Bottleneck latent dimension.
        n_filters: Base channel width; doubles at each encoder stage.
        n_residual_layers: Number of ResBlocks per encoder stage.
        ratios: Downsampling stride at each stage (applied in reverse order
                so the list matches the decoder's upsampling order).
        lstm: Number of LSTM layers (0 to disable).
        time_conditioned: Pass True for the InDI model variant.
        time_emb_dim: Must match TimeEmbedding.emb_dim when time_conditioned=True.
        (remaining args forwarded to SEANetResnetBlock / SConv1d)
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = 'reflect',
        true_skip: bool = True,
        compress: int = 2,
        lstm: int = 2,
        time_conditioned: bool = False,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        # Encoder applies ratios in reverse so the architecture matches the decoder
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(ratios))
        self.time_conditioned = time_conditioned
        self.time_emb_dim = time_emb_dim

        act = getattr(nn, activation)
        mult = 1  # channel multiplier, doubles at each stage

        # Initial projection: 1 channel -> n_filters channels
        self.layers = nn.ModuleList([
            SConv1d(channels, mult * n_filters, kernel_size,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ])

        for ratio in self.ratios:
            # Residual processing at current resolution
            for j in range(n_residual_layers):
                self.layers.append(
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base ** j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm, norm_params=norm_params,
                        causal=causal, pad_mode=pad_mode,
                        compress=compress, true_skip=true_skip,
                        time_conditioned=time_conditioned,
                        time_emb_dim=time_emb_dim,
                    )
                )
            # Downsampling conv (stride = ratio)
            self.layers.append(act(**activation_params))
            self.layers.append(
                SConv1d(mult * n_filters, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode)
            )
            mult *= 2

        if lstm:
            self.layers.append(SLSTM(mult * n_filters, num_layers=lstm))

        # Final projection to bottleneck dimension
        self.layers.append(act(**activation_params))
        self.layers.append(
            SConv1d(mult * n_filters, dimension, last_kernel_size,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: tp.Optional[torch.Tensor] = None,
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Args:
            x: Input waveform, shape (B, 1, T).
            time_emb: Time embedding, shape (B, time_emb_dim).
                      Required when time_conditioned=True.
        Returns:
            latent: Bottleneck representation, shape (B, dimension, T').
            skips:  List of feature maps from each downsampling conv,
                    ordered from shallowest to deepest. Used by the decoder
                    for U-Net skip connections.
        """
        skips = []
        out = x
        for layer in self.layers:
            if isinstance(layer, SEANetResnetBlock):
                out = layer(out, time_emb)
            else:
                out = layer(out)
            # Collect skip features after each strided (downsampling) conv
            if isinstance(layer, SConv1d) and layer.conv.conv.stride[0] > 1:
                skips.append(out)
        return out, skips


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SEANetDecoder(nn.Module):
    """SEANet hierarchical upsampling decoder with optional time conditioning.

    Mirrors SEANetEncoder: upsamples the bottleneck latent back to the original
    waveform resolution via transposed convolutions, each followed by residual
    blocks.

    U-Net skip connections from the paired encoder are concatenated (channel-
    wise) just before each transposed conv and projected back to the expected
    channel count via a learned 1×1 conv.

    Args: (symmetric with SEANetEncoder; see that class for full docs)
        final_activation: Optional activation applied to the final output
            (e.g. 'Tanh' to clamp output to [-1, 1]).
        trim_right_ratio: Causal-mode only; fraction of padding removed
            from the right side of each transposed conv.
        time_conditioned: Pass True for the InDI model variant.
        time_emb_dim: Must match TimeEmbedding.emb_dim when time_conditioned=True.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = 'weight_norm',
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = 'reflect',
        true_skip: bool = True,
        compress: int = 2,
        lstm: int = 2,
        trim_right_ratio: float = 1.0,
        time_conditioned: bool = False,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(ratios))
        self.time_conditioned = time_conditioned
        self.time_emb_dim = time_emb_dim

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))  # start at widest channel count

        # Initial projection from bottleneck dimension
        self.layers = nn.ModuleList([
            SConv1d(dimension, mult * n_filters, kernel_size,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ])

        if lstm:
            self.layers.append(SLSTM(mult * n_filters, num_layers=lstm))

        for ratio in self.ratios:
            # Upsampling transposed conv (stride = ratio)
            self.layers.append(act(**activation_params))
            self.layers.append(
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio)
            )
            # Residual processing at current resolution
            for j in range(n_residual_layers):
                self.layers.append(
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base ** j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm, norm_params=norm_params,
                        causal=causal, pad_mode=pad_mode,
                        compress=compress, true_skip=true_skip,
                        time_conditioned=time_conditioned,
                        time_emb_dim=time_emb_dim,
                    )
                )
            mult //= 2

        # Final projection to output channels
        self.layers.append(act(**activation_params))
        self.layers.append(
            SConv1d(n_filters, channels, last_kernel_size,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        )
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            self.layers.append(final_act(**(final_activation_params or {})))

        # 1×1 skip projection convs: one per upsampling stage.
        # Before each transposed conv the encoder skip is concatenated
        # (doubling the channel count), so we need to project back down.
        self.skip_projections = nn.ModuleList()
        in_ch = n_filters * (2 ** len(self.ratios)) * 2  # doubled after concat
        for _ in range(len(self.ratios)):
            self.skip_projections.append(
                SConv1d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=1)
            )
            in_ch //= 2

    def forward(
        self,
        x: torch.Tensor,
        skips: tp.List[torch.Tensor],
        time_emb: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Bottleneck latent, shape (B, dimension, T').
            skips: Skip feature maps from the paired encoder (shallowest first).
            time_emb: Time embedding, shape (B, time_emb_dim).
                      Required when time_conditioned=True.
        Returns:
            Reconstructed waveform, shape (B, 1, T).
        """
        # Decoder consumes skips in reverse order (deepest first)
        used_skips = list(reversed(skips))
        skip_idx = 0
        proj_idx = 0

        for layer in self.layers:
            if isinstance(layer, SEANetResnetBlock):
                x = layer(x, time_emb)
            elif isinstance(layer, SConvTranspose1d):
                # Inject skip connection before upsampling
                if skip_idx < len(used_skips):
                    skip = used_skips[skip_idx]
                    x = torch.cat([x, skip], dim=1)           # double channels
                    x = self.skip_projections[proj_idx](x)    # project back down
                    skip_idx += 1
                    proj_idx += 1
                x = layer(x)
            else:
                x = layer(x)
        return x

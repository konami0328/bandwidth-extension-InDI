"""InDI model: time-conditioned SEANet for iterative bandwidth extension.

The InDI (Inversion by Direct Iteration) model is trained to predict the clean
wideband signal x_0 from an intermediate degraded state x_t, where t ∈ (0, 1]
indicates the degradation level along the path:

    x_t = (1 - t) * x_0 + t * x_1        (x_1 = 8 kHz narrowband input)

At inference, starting from x_1 the model iteratively refines the signal
toward x_0 using the update rule (Eq. 4.5 in the thesis):

    x_{t-δ} = (δ/t) * F_θ(x_t, t) + (1 - δ/t) * x_t

See inference/strategies.py for the step schedule implementations.
"""

import torch
import torch.nn as nn

from .components import SEANetEncoder, SEANetDecoder, TimeEmbedding


class InDIModel(nn.Module):
    """Time-conditioned encoder-decoder for iterative bandwidth extension.

    Wraps SEANetEncoder + SEANetDecoder with a sinusoidal TimeEmbedding.
    The time embedding is injected into every ResBlock so the network can
    adapt its restoration behaviour to the current degradation level t.

    Args:
        n_filters: Base channel width of the SEANet backbone (doubles each
                   encoder stage: 32 → 64 → 128 → 256 → 512).
        dimension: Bottleneck latent dimension.
        ratios: Downsampling / upsampling strides per stage.
        lstm: Number of LSTM layers in the bottleneck (0 to disable).
        time_emb_dim: Dimension of the sinusoidal time embedding.
    """

    def __init__(
        self,
        n_filters: int = 32,
        dimension: int = 128,
        ratios: list = [8, 5, 4, 2],
        lstm: int = 2,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(emb_dim=time_emb_dim)
        self.encoder = SEANetEncoder(
            n_filters=n_filters,
            dimension=dimension,
            ratios=ratios,
            lstm=lstm,
            time_conditioned=True,
            time_emb_dim=time_emb_dim,
        )
        self.decoder = SEANetDecoder(
            n_filters=n_filters,
            dimension=dimension,
            ratios=ratios,
            lstm=lstm,
            time_conditioned=True,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the clean signal x_0 from a degraded state x_t.

        Args:
            x: Degraded waveform x_t, shape (B, 1, T).
            t: Degradation level, shape (B,), values in (0, 1].
        Returns:
            Predicted clean waveform x_0, shape (B, 1, T).
        """
        time_emb = self.time_embedding(t)               # (B, time_emb_dim)
        latent, skips = self.encoder(x, time_emb)
        return self.decoder(latent, skips, time_emb)


if __name__ == '__main__':
    # Quick shape sanity check
    model = InDIModel()
    x = torch.randn(4, 1, 48000)
    t = torch.rand(4)
    y = model(x, t)
    assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"
    print(f"InDIModel OK — input: {x.shape}, output: {y.shape}")

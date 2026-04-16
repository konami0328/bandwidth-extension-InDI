"""Baseline model: single-step SEANet regressor for bandwidth extension.

This is the Multi-Step Regressor Ensemble variant described in Section 4.3
of the thesis. Each instance is a standard encoder-decoder trained to predict
x_0 directly from a fixed degradation level x_t (i.e., t is fixed at training
time, not passed as input).

Unlike InDIModel, this model has no time conditioning — it is specialised for
one degradation level. Multiple instances are trained at different values of t
(e.g., t=1.0, t=0.5) and chained at inference to form a multi-step ensemble:

    x_1  →  Baseline(t=1.0→0.5)  →  x_0.5  →  Baseline(t=0.5→0.0)  →  x_0

This serves as the comparison baseline for the InDI time-conditioned model.
"""

import torch
import torch.nn as nn

from .components import SEANetEncoder, SEANetDecoder


class BaselineModel(nn.Module):
    """Single-step SEANet regressor (no time conditioning).

    Identical architecture to InDIModel but without the TimeEmbedding or
    any conditioning injection. Trained with a fixed degradation level t.

    Args:
        n_filters: Base channel width of the SEANet backbone.
        dimension: Bottleneck latent dimension.
        ratios: Downsampling / upsampling strides per stage.
        lstm: Number of LSTM layers in the bottleneck (0 to disable).
    """

    def __init__(
        self,
        n_filters: int = 32,
        dimension: int = 128,
        ratios: list = [8, 5, 4, 2],
        lstm: int = 2,
    ):
        super().__init__()
        self.encoder = SEANetEncoder(
            n_filters=n_filters,
            dimension=dimension,
            ratios=ratios,
            lstm=lstm,
            time_conditioned=False,
        )
        self.decoder = SEANetDecoder(
            n_filters=n_filters,
            dimension=dimension,
            ratios=ratios,
            lstm=lstm,
            time_conditioned=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the clean signal x_0 from a degraded input x_t.

        Args:
            x: Degraded waveform x_t, shape (B, 1, T).
               The degradation level t is implicit (fixed at training time).
        Returns:
            Predicted clean waveform x_0, shape (B, 1, T).
        """
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)


if __name__ == '__main__':
    # Quick shape sanity check
    model = BaselineModel()
    x = torch.randn(4, 1, 48000)
    y = model(x)
    assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"
    print(f"BaselineModel OK — input: {x.shape}, output: {y.shape}")

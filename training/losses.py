"""Loss functions for bandwidth extension training.

All loss functions operate on raw waveforms (B, 1, T) and return a scalar.

Available losses
----------------
wav_l1_loss       : Time-domain L1 loss.
wav_l2_loss       : Time-domain L2 (MSE) loss.
log_mel_mse_loss  : Log-Mel spectrogram MSE loss (dB scale).
combined_loss     : Weighted sum of wav_l1 + log_mel_mse (recommended).
multiscale_mel_loss: Multi-scale Mel MSE averaged over several FFT sizes.

Recommended training loss
-------------------------
`combined_loss` with wav_weight=100 was the best-performing configuration in
the thesis experiments (Section 5.2). It balances time-domain fidelity with
perceptual spectral accuracy:

    L = wav_weight * L1(wav) + MSE(log_mel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Spectrogram helpers (created once, reused across calls)
# ---------------------------------------------------------------------------

def _make_mel_extractor(
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    device: torch.device = None,
) -> T.MelSpectrogram:
    """Build a MelSpectrogram transform and move it to the target device."""
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        f_min=0,
        f_max=sample_rate // 2,
        power=1.0,
    )
    if device is not None:
        mel = mel.to(device)
    return mel


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def wav_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Time-domain L1 loss.

    Args:
        pred:   Predicted waveform, shape (B, 1, T).
        target: Target waveform,    shape (B, 1, T).
    Returns:
        Scalar loss value.
    """
    return F.l1_loss(pred, target)


def wav_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Time-domain L2 (MSE) loss.

    Args:
        pred:   Predicted waveform, shape (B, 1, T).
        target: Target waveform,    shape (B, 1, T).
    Returns:
        Scalar loss value.
    """
    return F.mse_loss(pred, target)


def log_mel_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mel_extractor: T.MelSpectrogram,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Log-Mel spectrogram MSE loss (amplitude-to-dB scale).

    Computing the loss on log-Mel spectrograms correlates well with the
    Log-Spectral Distortion (LSD) evaluation metric used in the experiments.

    Args:
        pred:           Predicted waveform, shape (B, 1, T).
        target:         Target waveform,    shape (B, 1, T).
        mel_extractor:  A pre-built MelSpectrogram transform on the correct device.
        eps:            Small value added before log to avoid log(0).
    Returns:
        Scalar loss value.
    """
    amp2db = T.AmplitudeToDB(stype="amplitude", top_db=80.0).to(pred.device)

    # Remove channel dim: (B, 1, T) -> (B, T) for MelSpectrogram
    pred_mel   = mel_extractor(pred.squeeze(1))    # (B, n_mels, frames)
    target_mel = mel_extractor(target.squeeze(1))

    pred_log_mel   = amp2db(pred_mel   + eps)
    target_log_mel = amp2db(target_mel + eps)

    return F.mse_loss(pred_log_mel, target_log_mel)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mel_extractor: T.MelSpectrogram,
    wav_weight: float = 100.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted combination of time-domain L1 and log-Mel MSE.

    This was the best-performing loss in the thesis (Section 5.2).
    The wav_weight=100 compensates for the scale difference between the
    two terms so they contribute roughly equally to the total.

    Args:
        pred:          Predicted waveform, shape (B, 1, T).
        target:        Target waveform,    shape (B, 1, T).
        mel_extractor: Pre-built MelSpectrogram transform on the correct device.
        wav_weight:    Scalar weight applied to the L1 term.
    Returns:
        total_loss:  Weighted sum (scalar tensor, used for .backward()).
        components:  Dict with individual loss values (for logging).
    """
    l1   = wav_l1_loss(pred, target)
    lmel = log_mel_mse_loss(pred, target, mel_extractor)
    total = wav_weight * l1 + lmel
    return total, {"wav_l1": l1.item(), "log_mel_mse": lmel.item()}


def multiscale_mel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
    fft_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048),
    n_mels: int = 64,
) -> torch.Tensor:
    """Multi-scale Mel MSE averaged over several FFT window sizes.

    Using multiple scales captures both fine-grained and coarse spectral
    structure. Included for completeness; combined_loss was preferred in
    the final experiments.

    Args:
        pred:        Predicted waveform, shape (B, 1, T).
        target:      Target waveform,    shape (B, 1, T).
        sample_rate: Audio sample rate in Hz.
        fft_sizes:   Tuple of FFT window sizes (powers of 2 recommended).
        n_mels:      Number of Mel filterbank channels per scale.
    Returns:
        Scalar loss averaged across all scales.
    """
    total = torch.tensor(0.0, device=pred.device)
    for win_len in fft_sizes:
        mel_fn = _make_mel_extractor(
            sample_rate=sample_rate,
            n_fft=win_len,
            hop_length=win_len // 4,
            n_mels=n_mels,
            device=pred.device,
        )
        pred_mel   = mel_fn(pred.squeeze(1))
        target_mel = mel_fn(target.squeeze(1))
        total = total + F.mse_loss(pred_mel, target_mel)
    return total / len(fft_sizes)


# ---------------------------------------------------------------------------
# Loss registry — maps CLI argument strings to callable loss functions
# ---------------------------------------------------------------------------

def get_loss_fn(name: str, mel_extractor, wav_weight: float = 100.0):
    """Return a loss function by name.

    All returned callables have the unified signature:
        loss_fn(pred, target) -> (scalar_tensor, component_dict)

    where component_dict is a {str: float} dict for logging (may be empty).

    Args:
        name:          One of: 'combined', 'log_mel', 'wav_l1', 'wav_l2',
                       'multiscale_mel'.
        mel_extractor: Pre-built MelSpectrogram (required for mel-based losses).
        wav_weight:    Weight for the L1 term in 'combined' loss.
    Returns:
        Callable loss function.
    Raises:
        ValueError: If name is not recognised.
    """
    def _wrap(fn):
        """Wrap a loss that returns a plain tensor into (tensor, {}) form."""
        return lambda pred, target: (fn(pred, target), {})

    registry = {
        "combined": lambda pred, target: combined_loss(
            pred, target, mel_extractor, wav_weight
        ),
        "log_mel": _wrap(
            lambda pred, target: log_mel_mse_loss(pred, target, mel_extractor)
        ),
        "wav_l1":  _wrap(wav_l1_loss),
        "wav_l2":  _wrap(wav_l2_loss),
        "multiscale_mel": _wrap(multiscale_mel_loss),
    }

    if name not in registry:
        raise ValueError(
            f"Unknown loss '{name}'. Choose from: {list(registry.keys())}"
        )
    return registry[name]

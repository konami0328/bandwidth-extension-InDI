"""Inference dataset for speech bandwidth extension.

Loads raw .wav files directly (no pre-caching required), resamples to the
target sample rate if needed, and pads each waveform to a multiple of the
SEANet hop length so the encoder-decoder runs without shape errors.

Typical usage:
    dataset = BWEInferenceDataset(input_dir="path/to/8kHz_upsampled_wavs/")
    for filename, waveform, pad_len in DataLoader(dataset, batch_size=1):
        output = model(waveform.squeeze(0))
        output = output[..., : output.shape[-1] - pad_len]  # remove padding
        torchaudio.save(...)
"""

import os
import torch
import torchaudio
from torch.utils.data import Dataset

# SEANet with ratios=[8,5,4,2] has a total downsampling factor of 320.
# All inputs must be a multiple of this to avoid shape mismatches.
SEANET_HOP_LENGTH = 320


class BWEInferenceDataset(Dataset):
    """Dataset for running inference on a directory of .wav files.

    Args:
        input_dir: Directory containing input .wav files (narrowband-upsampled).
        target_sr: Expected sample rate. Files at a different rate are resampled.
    """

    def __init__(self, input_dir: str, target_sr: int = 16000):
        self.input_dir = input_dir
        self.target_sr = target_sr
        self.file_list = sorted(
            f for f in os.listdir(input_dir) if f.endswith('.wav')
        )
        if not self.file_list:
            raise ValueError(f"No .wav files found in: {input_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """Load, optionally resample, and pad one waveform.

        Returns:
            filename:  str — original filename (useful for saving outputs).
            waveform:  Tensor[1, 1, T_padded] — padded input ready for the model.
            pad_len:   int — number of padding samples added (strip from output).
        """
        filename = self.file_list[idx]
        path = os.path.join(self.input_dir, filename)

        waveform, sr = torchaudio.load(path)  # [C, T]

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        # Pad to a multiple of the SEANet hop length
        waveform, pad_len = _pad_to_multiple(waveform, SEANET_HOP_LENGTH)

        # Add batch dimension: [1, T] -> [1, 1, T] so model receives (B, C, T)
        waveform = waveform.unsqueeze(0)

        return filename, waveform, pad_len


def _pad_to_multiple(x: torch.Tensor, multiple: int):
    """Zero-pad the last dimension of x to the next multiple of `multiple`.

    Args:
        x:        Input tensor, any shape.
        multiple: Target alignment (e.g. 320 for SEANet).
    Returns:
        x_padded: Padded tensor.
        pad_len:  Number of zeros added (needed to trim the model output).
    """
    T = x.shape[-1]
    remainder = T % multiple
    pad_len = (multiple - remainder) % multiple
    if pad_len > 0:
        x = torch.nn.functional.pad(x, (0, pad_len))
    return x, pad_len

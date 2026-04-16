"""Training dataset for speech bandwidth extension.

Each sample is a pre-cached .pt file containing a paired waveform dict:
    {
        "original": Tensor[1, T],   # 16 kHz full-bandwidth waveform (target)
        "downup":   Tensor[1, T],   # 8 kHz → 16 kHz upsampled waveform (input)
    }

Pre-caching avoids repeated resampling during training. See
data/prepare/cache_data.py for how to generate these files from raw audio.

If a sample is shorter than segment_size, it is zero-padded on the right.
If it is longer, a random segment is cropped (standard practice for variable-
length audio training).
"""

import os
import torch
from torch.utils.data import Dataset


class BWEDataset(Dataset):
    """Paired dataset of narrowband-upsampled and full-bandwidth waveforms.

    Args:
        data_dir: Directory containing pre-cached .pt files.
        segment_size: Number of samples per training segment (e.g. 48000 for
                      3 seconds at 16 kHz). If None, return full utterances.
    """

    def __init__(self, data_dir: str, segment_size: int = None):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.file_list = sorted(os.listdir(data_dir))
        if not self.file_list:
            raise ValueError(f"No files found in data_dir: {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """Return a (narrowband_upsampled, original) waveform pair.

        Returns:
            wav_input:  Tensor[1, segment_size] — degraded input (x_1 in InDI notation)
            wav_target: Tensor[1, segment_size] — clean target  (x_0 in InDI notation)
        """
        path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(path)

        wav_input = data["downup"]    # [1, T]  narrowband-upsampled (8 kHz → 16 kHz)
        wav_target = data["original"] # [1, T]  full-bandwidth 16 kHz

        if self.segment_size is not None:
            wav_input, wav_target = self._crop_or_pad(wav_input, wav_target)

        return wav_input, wav_target

    def _crop_or_pad(
        self,
        wav_input: torch.Tensor,
        wav_target: torch.Tensor,
    ):
        """Randomly crop or zero-pad both waveforms to segment_size."""
        T = wav_input.shape[-1]

        if T >= self.segment_size:
            # Random crop: same start position for both waveforms
            start = torch.randint(0, T - self.segment_size + 1, (1,)).item()
            wav_input  = wav_input[:, start : start + self.segment_size]
            wav_target = wav_target[:, start : start + self.segment_size]
        else:
            # Zero-pad short utterances on the right
            pad_len = self.segment_size - T
            wav_input  = torch.nn.functional.pad(wav_input,  (0, pad_len))
            wav_target = torch.nn.functional.pad(wav_target, (0, pad_len))

        return wav_input, wav_target

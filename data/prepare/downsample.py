"""Step 1 — Downsample raw audio to create paired training data.

Reads 16 kHz .flac files from a LibriSpeech-style directory tree, and for
each utterance produces three files in a flat output directory:

    <utt_id>_original.wav      — full-bandwidth 16 kHz reference
    <utt_id>_down_8k.wav       — 8 kHz downsampled (bandwidth-limited)
    <utt_id>_downup_8k.wav     — 8 kHz downsampled then upsampled back to 16 kHz
                                  (this is the model input during training)

The downup file simulates what a real narrowband signal looks like after naive
upsampling: correct length, but missing all frequency content above 4 kHz.

Usage
-----
    python downsample.py --data_dir /path/to/LibriSpeech/train-clean-100 \\
                         --output_dir /path/to/output/train \\
                         --orig_sr 16000

Run separately for train and dev splits:
    python downsample.py --data_dir /data/LibriSpeech/train-clean-100 \\
                         --output_dir /data/processed/train
    python downsample.py --data_dir /data/LibriSpeech/dev-clean \\
                         --output_dir /data/processed/dev
"""

import os
import argparse

import numpy as np
import librosa
import soundfile as sf
import torchaudio
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Downsample LibriSpeech flac files to create BWE training pairs."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of raw LibriSpeech .flac files "
                             "(may be nested in speaker/chapter subdirectories).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Flat output directory for all generated .wav files.")
    parser.add_argument("--orig_sr", type=int, default=16000,
                        help="Sample rate of the source audio (default: 16000).")
    parser.add_argument("--target_sr", type=int, default=8000,
                        help="Degraded sample rate to simulate (default: 8000).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------

def _is_complete(utt_id: str, output_dir: str, target_sr: int) -> bool:
    """Return True if all three output files already exist for this utterance."""
    sr_tag = target_sr // 1000
    expected = [
        f"{utt_id}_original.wav",
        f"{utt_id}_down_{sr_tag}k.wav",
        f"{utt_id}_downup_{sr_tag}k.wav",
    ]
    return all(os.path.exists(os.path.join(output_dir, f)) for f in expected)


def _downsample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample wav from orig_sr down to target_sr."""
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)


def _downsample_and_upsample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Downsample to target_sr then upsample back to orig_sr.

    This simulates what a narrowband signal looks like when naively resampled
    to the wideband rate: the length matches, but all high-frequency content
    above target_sr/2 is zeroed out.
    """
    downsampled = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
    return librosa.resample(downsampled, orig_sr=target_sr, target_sr=orig_sr)


def process_all_files(data_dir: str, output_dir: str, orig_sr: int, target_sr: int):
    """Walk data_dir recursively, process every .flac file found."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all .flac files (LibriSpeech stores them in speaker/chapter dirs)
    flac_files = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".flac"):
                flac_files.append(os.path.join(root, fname))

    print(f"Found {len(flac_files)} .flac files in {data_dir}")

    sr_tag = target_sr // 1000
    processed, skipped = 0, 0

    for flac_path in tqdm(flac_files, desc="Downsampling"):
        utt_id = os.path.basename(flac_path).replace(".flac", "")

        # Skip utterances that are already fully processed (allows resuming)
        if _is_complete(utt_id, output_dir, target_sr):
            skipped += 1
            continue

        try:
            waveform, sr = torchaudio.load(flac_path)
            assert sr == orig_sr, \
                f"Expected sample rate {orig_sr}, got {sr} for {flac_path}"

            wav = waveform.squeeze().numpy()  # mono float32

            # Original 16 kHz
            sf.write(
                os.path.join(output_dir, f"{utt_id}_original.wav"),
                wav, orig_sr, subtype="PCM_16",
            )
            # Downsampled to target_sr
            sf.write(
                os.path.join(output_dir, f"{utt_id}_down_{sr_tag}k.wav"),
                _downsample(wav, orig_sr, target_sr), target_sr, subtype="PCM_16",
            )
            # Downsampled then upsampled (model input)
            sf.write(
                os.path.join(output_dir, f"{utt_id}_downup_{sr_tag}k.wav"),
                _downsample_and_upsample(wav, orig_sr, target_sr), orig_sr, subtype="PCM_16",
            )
            processed += 1

        except Exception as e:
            print(f"[WARN] Failed for {utt_id}: {e}")

    print(f"Done. Processed: {processed}, Skipped (already complete): {skipped}")

    # Integrity check
    missing = [
        os.path.basename(p).replace(".flac", "")
        for p in flac_files
        if not _is_complete(
            os.path.basename(p).replace(".flac", ""), output_dir, target_sr
        )
    ]
    if missing:
        print(f"[WARN] {len(missing)} utterances are still incomplete:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more.")
    else:
        print("All utterances processed successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    process_all_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        orig_sr=args.orig_sr,
        target_sr=args.target_sr,
    )

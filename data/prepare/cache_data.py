"""Step 2 — Cache paired waveforms as .pt files for fast training.

Reads the _original.wav and _downup_8k.wav pairs produced by downsample.py
and saves them together as a single .pt file per utterance:

    {
        "original": Tensor[1, T],   # 16 kHz full-bandwidth (training target)
        "downup":   Tensor[1, T],   # 8->16 kHz upsampled  (model input)
    }

Why cache?
----------
Loading two separate .wav files and stacking them on every training iteration
is slow, especially with num_workers > 0. Caching merges them into a single
torch.load() call, which is faster and avoids file-system pressure during
long training runs.

Usage
-----
    python cache_data.py --input_dir /data/processed/train \\
                         --output_dir /data/cached/train

    python cache_data.py --input_dir /data/processed/dev \\
                         --output_dir /data/cached/dev
"""

import os
import argparse

import torch
import torchaudio
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache paired (original, downup) waveforms as .pt files."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing *_original.wav and "
                             "*_downup_8k.wav files (output of downsample.py).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write <utt_id>.pt cache files.")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Expected sample rate for both files (default: 16000).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cache_all(input_dir: str, output_dir: str, sr: int):
    os.makedirs(output_dir, exist_ok=True)

    # Discover utterance IDs from _original.wav files
    all_files = os.listdir(input_dir)
    utt_ids = [
        f.replace("_original.wav", "")
        for f in all_files
        if f.endswith("_original.wav")
    ]
    print(f"Found {len(utt_ids)} utterances in {input_dir}")

    skipped, cached = 0, 0

    for utt_id in tqdm(utt_ids, desc="Caching"):
        out_path = os.path.join(output_dir, f"{utt_id}.pt")

        # Skip if already cached
        if os.path.exists(out_path):
            skipped += 1
            continue

        ori_path    = os.path.join(input_dir, f"{utt_id}_original.wav")
        downup_path = os.path.join(input_dir, f"{utt_id}_downup_8k.wav")

        if not os.path.exists(downup_path):
            print(f"[WARN] Missing downup file for {utt_id}, skipping.")
            continue

        try:
            ori_wav,    sr1 = torchaudio.load(ori_path)    # [1, T]
            downup_wav, sr2 = torchaudio.load(downup_path) # [1, T]

            assert sr1 == sr and sr2 == sr, \
                f"Sample rate mismatch for {utt_id}: got {sr1}, {sr2}, expected {sr}"
            assert ori_wav.shape == downup_wav.shape, \
                f"Shape mismatch for {utt_id}: {ori_wav.shape} vs {downup_wav.shape}"

            torch.save({"original": ori_wav, "downup": downup_wav}, out_path)
            cached += 1

        except Exception as e:
            print(f"[WARN] Failed for {utt_id}: {e}")

    print(f"Done. Cached: {cached}, Skipped (already existed): {skipped}")


if __name__ == "__main__":
    args = parse_args()
    cache_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sr=args.sr,
    )

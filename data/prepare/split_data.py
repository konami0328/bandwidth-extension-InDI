"""Step 3 (optional) — Sample a random subset of cached .pt files.

Useful for creating small subsets for quick experiments or hyperparameter
searches without running on the full dataset.

Usage
-----
Sample 1000 random files from the training cache:
    python split_data.py --source_dir /data/cached/train \\
                         --dest_dir   /data/cached/train_sub1000 \\
                         --n 1000

Sample 100 files from the dev cache (for fast evaluation):
    python split_data.py --source_dir /data/cached/dev \\
                         --dest_dir   /data/cached/dev_sub100 \\
                         --n 100
"""

import os
import random
import shutil
import argparse
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy a random subset of .pt cache files to a new directory."
    )
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Source directory containing .pt files.")
    parser.add_argument("--dest_dir", type=str, required=True,
                        help="Destination directory for the sampled subset.")
    parser.add_argument("--n", type=int, required=True,
                        help="Number of files to sample.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sample_subset(source_dir: str, dest_dir: str, n: int, seed: int):
    all_files = [f for f in os.listdir(source_dir) if f.endswith(".pt")]

    if not all_files:
        raise ValueError(f"No .pt files found in: {source_dir}")
    if n > len(all_files):
        raise ValueError(
            f"Requested {n} files but only {len(all_files)} available in {source_dir}."
        )

    random.seed(seed)
    selected = random.sample(all_files, n)

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Copying {n} files from {source_dir} -> {dest_dir}  (seed={seed})")

    for fname in tqdm(selected, desc="Copying"):
        shutil.copy(
            os.path.join(source_dir, fname),
            os.path.join(dest_dir, fname),
        )

    print(f"Done. {n} files copied to {dest_dir}")


if __name__ == "__main__":
    args = parse_args()
    sample_subset(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        n=args.n,
        seed=args.seed,
    )

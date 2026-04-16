"""Evaluation script for bandwidth extension models.

Compares one or more directories of enhanced .wav files against a ground-truth
directory and reports LSD and ViSQOL for each — matching the evaluation setup
used in the thesis (Chapter 5).

File naming convention
----------------------
Enhanced files are matched to ground-truth files by the utterance ID, which
is extracted as the part of the filename before the first underscore:

    Enhanced:     3000-15664-0017_indi_step2.wav  ->  ID: 3000-15664-0017
    Ground-truth: 3000-15664-0017_original.wav    ->  ID: 3000-15664-0017

Usage examples
--------------
Evaluate LSD only (no ViSQOL binary needed):
    python evaluate.py \\
        --gt_dir   /data/dev/original \\
        --eval_dirs /data/dev/indi_2step

Evaluate LSD + ViSQOL for multiple directories, save CSV:
    python evaluate.py \\
        --gt_dir        /data/dev/original \\
        --eval_dirs     /data/dev/unprocessed /data/dev/indi /data/dev/baseline \\
        --visqol_bin    /path/to/visqol \\
        --visqol_model  /path/to/libsvm_nu_svr_model.txt \\
        --output        results.csv
"""

import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from evaluation.metrics import (
    load_wav, compute_lsd, compute_band_lsd, compute_visqol,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate bandwidth extension model outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground-truth *_original.wav files.")
    parser.add_argument("--eval_dirs", type=str, nargs="+", required=True,
                        help="One or more directories of enhanced .wav files to evaluate.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save a summary CSV (e.g. results.csv).")

    # ViSQOL is optional — omit these flags to skip ViSQOL evaluation
    parser.add_argument("--visqol_bin", type=str, default=None,
                        help="Path to compiled ViSQOL binary (omit to skip ViSQOL).")
    parser.add_argument("--visqol_model", type=str, default=None,
                        help="Path to ViSQOL SVR model file.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ground-truth index
# ---------------------------------------------------------------------------

def build_gt_index(gt_dir: str) -> dict:
    """Build a {utterance_id: filepath} map from the ground-truth directory.

    Expects filenames like: <utt_id>_original.wav
    """
    index = {}
    for fname in os.listdir(gt_dir):
        if fname.endswith("_original.wav"):
            utt_id = fname.replace("_original.wav", "")
            index[utt_id] = os.path.join(gt_dir, fname)
    if not index:
        raise ValueError(f"No *_original.wav files found in gt_dir: {gt_dir}")
    return index


# ---------------------------------------------------------------------------
# Single-directory evaluation
# ---------------------------------------------------------------------------

def evaluate_directory(
    eval_dir: str,
    gt_index: dict,
    use_visqol: bool = False,
    visqol_bin: str = None,
    visqol_model: str = None,
) -> pd.DataFrame:
    """Compute LSD (and optionally ViSQOL) for every matched file in eval_dir.

    Args:
        eval_dir:    Directory of enhanced .wav files.
        gt_index:    {utt_id: gt_path} map from build_gt_index().
        use_visqol:  Whether to run ViSQOL scoring.
        visqol_bin:  ViSQOL binary path.
        visqol_model: ViSQOL model path.
    Returns:
        DataFrame with columns: utt_id, lsd, band_lsd_4_8k, [visqol].
    """
    eval_files = sorted(f for f in os.listdir(eval_dir) if f.endswith(".wav"))
    rows = []

    for fname in tqdm(eval_files, desc=f"  {os.path.basename(eval_dir)}", leave=False):
        utt_id = fname.split("_")[0]
        gt_path = gt_index.get(utt_id)
        if gt_path is None:
            continue  # no matching ground-truth, skip silently

        eval_path = os.path.join(eval_dir, fname)
        try:
            clean, sr   = load_wav(gt_path)
            enhanced, _ = load_wav(eval_path)
        except Exception as e:
            print(f"  [WARN] Could not load {fname}: {e}")
            continue

        row = {
            "utt_id":       utt_id,
            "lsd":          compute_lsd(clean, enhanced, sr=sr),
            "band_lsd_4_8k": compute_band_lsd(
                clean, enhanced, sr=sr, freq_min=4000, freq_max=8000
            ),
        }

        if use_visqol:
            row["visqol"] = compute_visqol(
                gt_path, eval_path, visqol_bin, visqol_model
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    use_visqol = args.visqol_bin is not None and args.visqol_model is not None

    print(f"Ground-truth directory : {args.gt_dir}")
    print(f"ViSQOL                 : {'enabled' if use_visqol else 'disabled (no --visqol_bin provided)'}")
    print()

    gt_index = build_gt_index(args.gt_dir)
    print(f"Ground-truth files found: {len(gt_index)}\n")

    metric_cols = ["lsd", "band_lsd_4_8k"] + (["visqol"] if use_visqol else [])
    summary_rows = []

    for eval_dir in args.eval_dirs:
        label = os.path.basename(eval_dir.rstrip("/"))
        print(f"Evaluating: {label}")

        df = evaluate_directory(
            eval_dir, gt_index,
            use_visqol=use_visqol,
            visqol_bin=args.visqol_bin,
            visqol_model=args.visqol_model,
        )

        if df.empty:
            print(f"  [WARN] No matched files in {eval_dir}\n")
            continue

        averages = {col: df[col].mean() for col in metric_cols if col in df.columns}
        averages["method"]  = label
        averages["n_files"] = len(df)
        summary_rows.append(averages)

        for col, val in averages.items():
            if col not in ("method", "n_files"):
                print(f"  {col:20s}: {val:.4f}")
        print(f"  {'files evaluated':20s}: {len(df)}\n")

    # Summary table
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        cols_order = ["method", "n_files"] + metric_cols
        df_summary = df_summary[[c for c in cols_order if c in df_summary.columns]]

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        if args.output:
            df_summary.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

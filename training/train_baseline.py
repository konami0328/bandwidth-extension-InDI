"""Training script for the Baseline single-step regressor.

The baseline is a standard supervised regressor trained to map a degraded
state x_t (at a fixed degradation level t) directly to the clean signal x_0
in a single forward pass. This is the Multi-Step Regressor Ensemble variant
from Section 4.3 of the thesis.

Training at t=1.0 produces a model that maps narrowband → wideband directly.
Training at t=0.5 produces a model specialised for the midpoint of the
degradation path. Multiple models can be chained at inference:

    x_1 -> model(t=1.0→0.5) -> x_0.5 -> model(t=0.5→0.0) -> x_0

See inference/infer.py for how to use the ensemble at inference time.

Usage
-----
Train the t=1.0 model (full bandwidth extension in one step):
    python train_baseline.py --data_dir /path/to/cached_train \\
                             --save_dir ./checkpoints/baseline_t1.0 \\
                             --t 1.0

Train the t=0.5 model (second half of a 2-step ensemble):
    python train_baseline.py --data_dir /path/to/cached_train \\
                             --save_dir ./checkpoints/baseline_t0.5 \\
                             --t 0.5
"""

import os
import csv
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import BaselineModel
from data.dataset import BWEDataset
from training.losses import get_loss_fn, _make_mel_extractor


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Baseline single-step regressor."
    )

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of pre-cached .pt training files.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save checkpoints and training log.")
    parser.add_argument("--segment_size", type=int, default=48000,
                        help="Training segment length in samples (default: 48000 = 3s).")

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=99)

    # Degradation level — defines which part of the path this model handles
    parser.add_argument("--t", type=float, default=1.0,
                        help="Fixed degradation level. "
                             "t=1.0: map x_1→x_0 directly. "
                             "t=0.5: map x_0.5→x_0 (second step of a 2-step ensemble).")

    # Loss
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["combined", "log_mel", "wav_l1", "wav_l2", "multiscale_mel"])
    parser.add_argument("--wav_weight", type=float, default=100.0)

    # Optimiser
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)

    # LR warm-up (helps avoid early instability)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warm-up epochs. Set 0 to disable.")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-5,
                        help="Initial LR at the start of warm-up.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training at fixed degradation level t={args.t}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Dataset & DataLoader ---
    dataset = BWEDataset(data_dir=args.data_dir, segment_size=args.segment_size)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Training samples: {len(dataset)}")

    # --- Model ---
    model = BaselineModel().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Loss ---
    mel_extractor = _make_mel_extractor(device=device)
    loss_fn = get_loss_fn(args.loss, mel_extractor, args.wav_weight)
    print(f"Loss: {args.loss}  (wav_weight={args.wav_weight})")

    # --- Optimiser & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
        verbose=True,
    )

    # --- CSV logging ---
    log_path = os.path.join(args.save_dir, f"train_log_t{args.t}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "avg_loss", "avg_wav_l1", "avg_log_mel_mse"])

    # --- Epoch loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_components = {"wav_l1": 0.0, "log_mel_mse": 0.0}

        # LR warm-up: linearly ramp from warmup_start_lr to args.lr
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_lr = args.warmup_start_lr + (args.lr - args.warmup_start_lr) * (
                epoch / args.warmup_epochs
            )
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            print(f"  Warm-up LR: {warmup_lr:.2e}")

        pbar = tqdm(loader, desc=f"[Epoch {epoch:03d}/{args.epochs}]")
        for wav_input, wav_target in pbar:
            wav_input  = wav_input.to(device)   # x_1: narrowband-upsampled  [B,1,T]
            wav_target = wav_target.to(device)  # x_0: clean full-bandwidth  [B,1,T]

            # Construct x_t at the fixed degradation level args.t
            # x_t = (1 - t)*x_0 + t*x_1
            t_bc = torch.full(
                (wav_input.shape[0], 1, 1), fill_value=args.t, device=device
            )
            x_t = (1 - t_bc) * wav_target + t_bc * wav_input

            # Forward pass: predict x_0 from x_t (no time input for baseline)
            optimizer.zero_grad()
            pred = model(x_t)

            loss, components = loss_fn(pred, wav_target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for k, v in components.items():
                if k in total_components:
                    total_components[k] += v

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- End of epoch ---
        avg_loss = total_loss / len(loader)
        avg_components = {k: v / len(loader) for k, v in total_components.items()}

        # Only step the scheduler after warm-up is complete
        if epoch > args.warmup_epochs:
            scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}] loss={avg_loss:.4f}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in avg_components.items())
            + f"  lr={current_lr:.2e}"
        )

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"baseline_t{args.t}_epoch{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # Append to CSV log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, current_lr, avg_loss,
                avg_components.get("wav_l1", ""),
                avg_components.get("log_mel_mse", ""),
            ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)

"""Training script for the InDI time-conditioned bandwidth extension model.

The InDI training objective (Eq. 4.1 in the thesis) is:
    - Sample a random degradation level t ~ Uniform(0, 1]
    - Form an intermediate state: x_t = (1-t)*x_0 + t*x_1
    - Train the model to predict x_0 from (x_t, t)

Usage
-----
Basic (uses all defaults from config.yaml):
    python train_indi.py --data_dir /path/to/cached_train --save_dir ./checkpoints/indi

Full example with overrides:
    python train_indi.py \\
        --data_dir  /path/to/cached_train \\
        --save_dir  ./checkpoints/indi \\
        --loss      combined \\
        --wav_weight 100 \\
        --lr        1e-3 \\
        --epochs    100 \\
        --batch_size 32
"""

import os
import csv
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchaudio.transforms as T

# Project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import InDIModel
from data.dataset import BWEDataset
from training.losses import get_loss_fn, _make_mel_extractor


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the InDI time-conditioned bandwidth extension model."
    )

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of pre-cached .pt training files.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save checkpoints and training log.")
    parser.add_argument("--segment_size", type=int, default=48000,
                        help="Training segment length in samples (default: 48000 = 3s).")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=99,
                        help="Random seed (fixes t sampling for reproducibility).")

    # Loss
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["combined", "log_mel", "wav_l1", "wav_l2", "multiscale_mel"],
                        help="Training loss function.")
    parser.add_argument("--wav_weight", type=float, default=100.0,
                        help="Weight for the L1 term in combined loss.")

    # Optimiser
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="ReduceLROnPlateau reduction factor.")
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="ReduceLROnPlateau patience (epochs).")

    # Noise perturbation (optional, from InDI paper Eq. 9)
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Noise perturbation scale added to x_t during training "
                             "(0 = no noise, as used in the thesis experiments).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    model = InDIModel().to(device)
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
    log_path = os.path.join(args.save_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "avg_loss", "avg_wav_l1", "avg_log_mel_mse"])

    # --- Epoch loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_components = {"wav_l1": 0.0, "log_mel_mse": 0.0}

        pbar = tqdm(loader, desc=f"[Epoch {epoch:03d}/{args.epochs}]")
        for wav_input, wav_target in pbar:
            wav_input  = wav_input.to(device)   # x_1: narrowband-upsampled  [B,1,T]
            wav_target = wav_target.to(device)  # x_0: clean full-bandwidth  [B,1,T]

            # Sample degradation level t ~ Uniform(0, 1] for each sample in batch
            t = torch.rand(wav_target.shape[0], device=device).clamp(min=1e-6)
            t_bc = t[:, None, None]  # broadcast shape [B, 1, 1]

            # Form intermediate degraded state x_t (InDI Eq. 4.1)
            # Optional: add small noise perturbation (epsilon=0 in thesis experiments)
            noise = t_bc * args.epsilon * torch.randn_like(wav_target)
            x_t = (1 - t_bc) * wav_target + t_bc * wav_input + noise

            # Forward pass: predict x_0 from (x_t, t)
            optimizer.zero_grad()
            pred = model(x_t, t)

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
        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}] loss={avg_loss:.4f}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in avg_components.items())
            + f"  lr={current_lr:.2e}"
        )

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"indi_epoch{epoch:03d}.pt")
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

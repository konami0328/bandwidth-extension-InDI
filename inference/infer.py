"""Unified inference script for bandwidth extension.

Supports both model variants (InDI and Baseline) and all inference strategies
from a single entry point. Replaces the four original inference scripts.

InDI model (time-conditioned, iterative)
-----------------------------------------
The model iteratively refines the narrowband-upsampled input using the InDI
update rule (Eq. 4.5):

    x_{t-δ} = (δ/t) * F_θ(x_t, t) + (1 - δ/t) * x_t

The step schedule (uniform or non-uniform) controls how t decreases from 1.0
to 0.0 across the inference steps.

Baseline model (single-step regressor ensemble)
------------------------------------------------
Each checkpoint was trained at a fixed degradation level t. Multi-step
inference chains N checkpoints, where each one handles one segment of the
degradation path. Pass multiple checkpoints via --ckpt:

    --ckpt ckpt_t1.0.pt ckpt_t0.5.pt   → 2-step ensemble (t: 1.0→0.5→0.0)
    --ckpt ckpt_t1.0.pt                 → 1-step direct regression

Usage examples
--------------
InDI, 1-step (single forward pass, t=1.0→0.0):
    python infer.py --model indi --ckpt indi_epoch095.pt \\
                    --input_dir /data/test_8k --output_dir /data/out

InDI, 2-step uniform (t: 1.0→0.5→0.0):
    python infer.py --model indi --ckpt indi_epoch095.pt \\
                    --input_dir /data/test_8k --output_dir /data/out \\
                    --steps 2 --strategy uniform

InDI, 2-step non-uniform (t: 1.0→0.9→0.0, thesis best result):
    python infer.py --model indi --ckpt indi_epoch095.pt \\
                    --input_dir /data/test_8k --output_dir /data/out \\
                    --steps 2 --strategy nonuniform --first_step_t 0.9

Baseline, 2-step ensemble:
    python infer.py --model baseline \\
                    --ckpt baseline_t1.0_epoch080.pt baseline_t0.5_epoch080.pt \\
                    --input_dir /data/test_8k --output_dir /data/out
"""

import os
import argparse

import torch
import torchaudio
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import InDIModel, BaselineModel
from data.dataset_inference import BWEInferenceDataset
from inference.strategies import get_schedule


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run bandwidth extension inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--model", type=str, required=True,
                        choices=["indi", "baseline"],
                        help="Model variant to use.")
    parser.add_argument("--ckpt", type=str, nargs="+", required=True,
                        help="Path(s) to checkpoint file(s). "
                             "InDI: one checkpoint. "
                             "Baseline: one per step in the ensemble.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory of narrowband-upsampled .wav files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write enhanced .wav files.")

    # InDI-specific
    parser.add_argument("--steps", type=int, default=1,
                        help="[InDI only] Number of inference steps (default: 1).")
    parser.add_argument("--strategy", type=str, default="uniform",
                        choices=["uniform", "nonuniform", "concave", "convex", "sigmoid"],
                        help="[InDI only] Step schedule strategy (default: uniform).")
    parser.add_argument("--first_step_t", type=float, default=0.9,
                        help="[InDI nonuniform] t value after the first step (default: 0.9).")
    parser.add_argument("--power", type=float, default=2.5,
                        help="[InDI concave/convex] Schedule curvature exponent.")
    parser.add_argument("--sigmoid_k", type=float, default=5.0,
                        help="[InDI sigmoid] Sharpness of the S-curve.")

    # General
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=99)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_indi(ckpt_path: str, device: torch.device) -> InDIModel:
    model = InDIModel()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"  Loaded InDI checkpoint: {ckpt_path}")
    return model


def load_baseline(ckpt_path: str, device: torch.device) -> BaselineModel:
    model = BaselineModel()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"  Loaded Baseline checkpoint: {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_indi(
    model: InDIModel,
    y: torch.Tensor,
    t_schedule: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Run InDI iterative inference on a single waveform.

    Args:
        model:      Loaded InDI model in eval mode.
        y:          Narrowband-upsampled input, shape (1, 1, T).
        t_schedule: Array of t values from 1.0 down to 0.0,
                    length = num_steps + 1.
        device:     Torch device.
    Returns:
        Enhanced waveform, shape (1, 1, T).
    """
    x_hat = y.clone()  # start from x_1 (the degraded input)

    for i in range(len(t_schedule) - 1):
        t_curr = float(t_schedule[i])
        t_next = float(t_schedule[i + 1])
        delta  = t_curr - t_next

        if delta == 0.0:
            continue

        t_tensor = torch.tensor(t_curr, dtype=torch.float32, device=device)
        t_batch  = t_tensor.expand(x_hat.shape[0])  # (B,)

        # Model predicts x_0 from current state x_t
        f_theta = model(x_hat, t_batch)

        # InDI update rule (Eq. 4.5)
        x_hat = (delta / t_curr) * f_theta + (1.0 - delta / t_curr) * x_hat

    return x_hat


@torch.no_grad()
def run_baseline_ensemble(
    models: list,
    y: torch.Tensor,
) -> torch.Tensor:
    """Run multi-step Baseline ensemble inference on a single waveform.

    Each model in the list handles one step of the degradation path.
    The update rule for each step follows the InDI formulation:

        x_{t_next} = (δ/t) * model(x_t) + (1 - δ/t) * x_t

    where the t values are equally spaced across the ensemble steps.
    The final model predicts x_0 directly (last step goes to t=0).

    Args:
        models: List of BaselineModel instances, ordered from t=1.0 toward t=0.
        y:      Narrowband-upsampled input, shape (1, 1, T).
    Returns:
        Enhanced waveform, shape (1, 1, T).
    """
    num_steps = len(models)
    # Uniform t schedule implied by the number of ensemble models
    t_schedule = np.linspace(1.0, 0.0, num_steps + 1)

    x_hat = y.clone()
    for i, model in enumerate(models):
        t_curr = float(t_schedule[i])
        t_next = float(t_schedule[i + 1])
        delta  = t_curr - t_next

        f_theta = model(x_hat)

        if i < num_steps - 1:
            # Intermediate step: blend prediction with current state
            x_hat = (delta / t_curr) * f_theta + (1.0 - delta / t_curr) * x_hat
        else:
            # Final step: the model's output IS the x_0 prediction
            x_hat = f_theta

    return x_hat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model(s) ---
    if args.model == "indi":
        if len(args.ckpt) != 1:
            raise ValueError("InDI inference requires exactly one --ckpt.")
        model = load_indi(args.ckpt[0], device)

        # Build step schedule
        schedule_kwargs = {
            "first_step_t": args.first_step_t,
            "power": args.power,
            "k": args.sigmoid_k,
        }
        t_schedule = get_schedule(args.strategy, args.steps, **schedule_kwargs)
        print(f"  Strategy: {args.strategy}, steps: {args.steps}")
        print(f"  t schedule: {np.round(t_schedule, 3).tolist()}")

    else:  # baseline
        models = [load_baseline(ckpt, device) for ckpt in args.ckpt]
        print(f"  Ensemble size: {len(models)} step(s)")

    # --- Dataset ---
    dataset = BWEInferenceDataset(input_dir=args.input_dir, target_sr=args.sample_rate)
    print(f"  Input files: {len(dataset)}")

    # --- Inference loop ---
    for filename, waveform, pad_len in tqdm(dataset, desc="Inferring"):
        waveform = waveform.to(device)  # (1, 1, T_padded)

        if args.model == "indi":
            output = run_indi(model, waveform, t_schedule, device)
        else:
            output = run_baseline_ensemble(models, waveform)

        # Remove the padding added by BWEInferenceDataset
        if pad_len > 0:
            output = output[..., :-pad_len]

        # Save
        output_wav = output.squeeze(0).cpu()  # (1, T)
        out_path = os.path.join(args.output_dir, filename)
        torchaudio.save(
            out_path, output_wav, args.sample_rate,
            encoding="PCM_S", bits_per_sample=16,
        )

    print(f"Done. Enhanced files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

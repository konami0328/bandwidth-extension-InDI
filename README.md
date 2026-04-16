# Inversion by Direct Iteration for Speech Bandwidth Extension

MSc Thesis Project — University of Edinburgh, 2025

---

## Background

**Speech bandwidth extension (BWE)** recovers missing high-frequency content from a low-quality narrowband signal. Concretely, this project extends speech from 8 kHz to 16 kHz — the difference between a telephone call and near-CD quality audio.

The standard approach trains a neural network to directly map the narrowband input to a wideband output in a single step. This works reasonably well, but suffers from the **regression-to-the-mean effect**: because many different wideband signals could plausibly correspond to the same narrowband input, the model learns to predict their average, producing over-smoothed, unnatural-sounding high frequencies.

**Inversion by Direct Iteration (InDI)** addresses this by decomposing the problem into a sequence of smaller steps. Instead of asking the model to jump straight from the degraded input to the clean output, it asks: *"given a signal that is X% degraded, predict what the fully clean signal looks like"* — and then takes a small step in that direction, repeating until the output is clean. Each individual step is much easier than doing everything at once, and the iterative refinement avoids the averaging problem.

> **Relation to DDPM**: DDPM (denoising diffusion) follows the same iterative philosophy, but its forward process adds *random Gaussian noise* to the signal. This makes the degradation analytically tractable but requires many steps (often 1000) at inference. InDI instead uses a *deterministic linear interpolation* between the clean and degraded signals as its forward process — which maps directly onto the BWE problem and requires as few as 2 steps at inference.

---

## Contributions

This project adapts InDI from image restoration (its original domain) to speech bandwidth extension. Three main contributions:

1. **InDI for BWE** — Successfully adapts the InDI framework to audio, using a combined time-domain L1 + log-Mel spectrogram loss. Reduces LSD from 4.32 (single-step baseline) to 4.26 (4-step InDI).

2. **Non-uniform inference strategy** — Observes that the degradation path is perceptually non-linear: most of the "hard work" happens near the fully-degraded end. Taking a smaller first step (e.g. t: 1.0 → 0.9) followed by a larger second step (0.9 → 0.0) distributes the reconstruction effort more evenly, reducing LSD to **4.04** with just 2 steps.

3. **Time-conditioned model** — Trains a single model that handles all degradation levels simultaneously, by injecting a sinusoidal time embedding. This avoids having to train a separate model for each step of the inference chain.

---

## Project Structure

```
bandwidth-extension-InDI/
│
├── models/                     # Model definitions
│   ├── indi.py                 # InDI time-conditioned model (main contribution)
│   ├── baseline.py             # Single-step regressor (comparison baseline)
│   └── components/
│       ├── seanet_blocks.py    # Shared SEANet encoder-decoder backbone
│       ├── conv.py             # Streaming conv wrappers (from EnCodec)
│       ├── lstm.py             # LSTM wrapper
│       └── norm.py             # Layer normalisation
│
├── data/
│   ├── dataset.py              # Training dataset (loads cached .pt files)
│   ├── dataset_inference.py    # Inference dataset (loads raw .wav files)
│   └── prepare/
│       ├── downsample.py       # Step 1: create paired 16k/8k audio from LibriSpeech
│       ├── cache_data.py       # Step 2: pack pairs into .pt files for fast loading
│       └── split_data.py       # Step 3 (optional): sample a random subset
│
├── training/
│   ├── losses.py               # All loss functions (L1, log-Mel, combined)
│   ├── train_indi.py           # Train the InDI model
│   └── train_baseline.py       # Train the single-step baseline
│
├── inference/
│   ├── infer.py                # Unified inference entry point
│   └── strategies.py           # Step schedule functions (uniform, non-uniform, ...)
│
└── evaluation/
    ├── metrics.py              # LSD, band-LSD, ViSQOL
    └── evaluate.py             # Evaluate one or more output directories
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Download [LibriSpeech](https://www.openslr.org/12) `train-clean-100` and `dev-clean`, then run the three preparation steps:

```bash
# Create paired 16 kHz / 8→16 kHz upsampled audio
python data/prepare/downsample.py \
    --data_dir  /data/LibriSpeech/train-clean-100 \
    --output_dir /data/processed/train

python data/prepare/downsample.py \
    --data_dir  /data/LibriSpeech/dev-clean \
    --output_dir /data/processed/dev

# Pack into .pt cache files for fast training
python data/prepare/cache_data.py \
    --input_dir  /data/processed/train \
    --output_dir /data/cached/train

python data/prepare/cache_data.py \
    --input_dir  /data/processed/dev \
    --output_dir /data/cached/dev
```

### 3. Train

**InDI model** (time-conditioned, main contribution):
```bash
python training/train_indi.py \
    --data_dir /data/cached/train \
    --save_dir ./checkpoints/indi \
    --loss combined \
    --wav_weight 100 \
    --epochs 100
```

**Baseline model** (single-step regressor, for comparison):
```bash
# Train at t=1.0 (direct narrowband → wideband mapping)
python training/train_baseline.py \
    --data_dir /data/cached/train \
    --save_dir ./checkpoints/baseline_t1.0 \
    --t 1.0

# Train at t=0.5 (second step of a 2-step ensemble)
python training/train_baseline.py \
    --data_dir /data/cached/train \
    --save_dir ./checkpoints/baseline_t0.5 \
    --t 0.5
```

### 4. Run inference

**InDI — 1-step** (single forward pass):
```bash
python inference/infer.py \
    --model indi \
    --ckpt  ./checkpoints/indi/indi_epoch095.pt \
    --input_dir  /data/processed/dev/downup \
    --output_dir /data/outputs/indi_1step
```

**InDI — 2-step, non-uniform** (best result in the thesis):
```bash
python inference/infer.py \
    --model indi \
    --ckpt  ./checkpoints/indi/indi_epoch095.pt \
    --input_dir  /data/processed/dev/downup \
    --output_dir /data/outputs/indi_2step_nonuniform \
    --steps 2 --strategy nonuniform --first_step_t 0.9
```

**Baseline — 2-step ensemble**:
```bash
python inference/infer.py \
    --model baseline \
    --ckpt  ./checkpoints/baseline_t1.0/baseline_t1.0_epoch080.pt \
            ./checkpoints/baseline_t0.5/baseline_t0.5_epoch080.pt \
    --input_dir  /data/processed/dev/downup \
    --output_dir /data/outputs/baseline_2step
```

---

## Evaluation

```bash
# LSD only
python evaluation/evaluate.py \
    --gt_dir    /data/processed/dev/original \
    --eval_dirs /data/outputs/indi_2step_nonuniform \
                /data/outputs/baseline_2step \
    --output    results.csv

# LSD + ViSQOL (requires compiled ViSQOL binary)
python evaluation/evaluate.py \
    --gt_dir       /data/processed/dev/original \
    --eval_dirs    /data/outputs/indi_2step_nonuniform \
    --visqol_bin   /path/to/visqol \
    --visqol_model /path/to/libsvm_nu_svr_model.txt \
    --output       results.csv
```

---

## Results

All results on the LibriSpeech `dev-clean` split. LSD (Log-Spectral Distortion) is lower-is-better; ViSQOL MOS-LQO is higher-is-better.

| Method | Steps | LSD ↓ | ViSQOL ↑ |
|---|---|---|---|
| Unprocessed (8→16 kHz naive upsample) | — | 5.93 | — |
| Single-step baseline | 1 | 4.32 | — |
| InDI, uniform | 2 | 4.30 | — |
| InDI, uniform | 4 | 4.26 | — |
| **InDI, non-uniform** (t: 1.0→0.9→0.0) | **2** | **4.04** | — |

The non-uniform 2-step strategy outperforms uniform 4-step inference, confirming that step placement matters more than step count.

ViSQOL improvements over the single-step baseline were minimal across all InDI variants, suggesting that the current loss functions improve spectral accuracy but not perceptual quality — an open direction for future work.

---

## References

- **InDI**: Delbracio & Milanfar, *"Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration"*, TMLR 2023. [[paper]](https://arxiv.org/abs/2303.11435)
- **EnCodec / SEANet**: Défossez et al., *"High Fidelity Neural Audio Compression"*, TMLR 2023. [[paper]](https://arxiv.org/abs/2210.13438) [[code]](https://github.com/facebookresearch/encodec)

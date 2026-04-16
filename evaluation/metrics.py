"""Evaluation metrics for speech bandwidth extension.

All metric functions take numpy arrays (mono, float32/float64) and a sample
rate, and return a scalar float.

Metrics
-------
compute_lsd      : Log-Spectral Distance — primary metric in the thesis.
compute_band_lsd : LSD restricted to a frequency sub-band (e.g. 4–8 kHz).
compute_visqol   : Virtual Speech Quality Objective Listener — perceptual
                   quality metric (requires a compiled ViSQOL binary).

These two metrics match the evaluation setup in the thesis (Chapter 5):
LSD measures spectral accuracy; ViSQOL measures perceptual quality.
"""

import subprocess
import numpy as np
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple:
    """Load a wav file as a float32 numpy array at its native sample rate.

    Returns:
        audio: 1-D numpy array, mono (multi-channel files are averaged).
        sr:    Sample rate in Hz.
    """
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def _align(clean: np.ndarray, enhanced: np.ndarray) -> tuple:
    """Trim both arrays to the same length (shorter one wins)."""
    n = min(len(clean), len(enhanced))
    return clean[:n], enhanced[:n]


# ---------------------------------------------------------------------------
# Log-Spectral Distance (LSD)
# ---------------------------------------------------------------------------

def compute_lsd(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    eps: float = 1e-10,
) -> float:
    """Compute Log-Spectral Distance between clean and enhanced audio.

    LSD measures the average per-frame distortion in the log power spectrum:

        LSD = mean_t { sqrt( mean_f [ (10*log10(S_clean) - 10*log10(S_enh))^2 ] ) }

    Lower is better. The thesis reports LSD values around 4.0-4.5 dB.

    Args:
        clean:      Reference (ground-truth) waveform.
        enhanced:   Enhanced (model output) waveform.
        sr:         Sample rate in Hz.
        n_fft:      STFT window size.
        hop_length: STFT hop size.
        eps:        Floor added before log to avoid log(0).
    Returns:
        Scalar LSD value in dB.
    """
    clean, enhanced = _align(clean, enhanced)
    S_clean = np.abs(librosa.stft(clean,    n_fft=n_fft, hop_length=hop_length)) + eps
    S_enh   = np.abs(librosa.stft(enhanced, n_fft=n_fft, hop_length=hop_length)) + eps
    log_diff = 10.0 * (np.log10(S_clean) - np.log10(S_enh))
    framewise_lsd = np.sqrt(np.mean(log_diff ** 2, axis=0))  # per frame
    return float(np.mean(framewise_lsd))


def compute_band_lsd(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    freq_min: float = 4000.0,
    freq_max: float = 8000.0,
    eps: float = 1e-10,
) -> float:
    """Compute LSD restricted to a specific frequency band.

    Useful for isolating the high-frequency reconstruction quality
    (e.g. 4-8 kHz, the extended band in 8->16 kHz BWE).

    Args:
        freq_min: Lower bound of the frequency band in Hz.
        freq_max: Upper bound of the frequency band in Hz.
        (other args same as compute_lsd)
    Returns:
        Scalar band-limited LSD value in dB, or NaN if no bins fall in band.
    """
    clean, enhanced = _align(clean, enhanced)
    S_clean = np.abs(librosa.stft(clean,    n_fft=n_fft, hop_length=hop_length))
    S_enh   = np.abs(librosa.stft(enhanced, n_fft=n_fft, hop_length=hop_length))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    if len(idx) == 0:
        return float("nan")

    S_clean_band = S_clean[idx, :] + eps
    S_enh_band   = S_enh[idx, :]   + eps
    log_diff = 10.0 * (np.log10(S_clean_band) - np.log10(S_enh_band))
    framewise_lsd = np.sqrt(np.mean(log_diff ** 2, axis=0))
    return float(np.mean(framewise_lsd))


# ---------------------------------------------------------------------------
# ViSQOL perceptual quality metric
# ---------------------------------------------------------------------------

def compute_visqol(
    clean_path: str,
    enhanced_path: str,
    visqol_bin: str,
    visqol_model: str,
) -> float:
    """Compute ViSQOL MOS-LQO score using an external binary.

    ViSQOL must be compiled separately (see https://github.com/google/visqol).
    Pass the paths to the compiled binary and SVR model file.

    Args:
        clean_path:    Path to reference .wav file.
        enhanced_path: Path to degraded .wav file.
        visqol_bin:    Path to compiled ViSQOL binary.
        visqol_model:  Path to ViSQOL SVR model file.
    Returns:
        MOS-LQO score (higher is better), or NaN on failure.
    """
    cmd = [
        visqol_bin,
        f"--reference_file={clean_path}",
        f"--degraded_file={enhanced_path}",
        f"--similarity_to_quality_model={visqol_model}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "MOS-LQO" in line:
                return float(line.split(":")[1].strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"[ViSQOL ERROR] {enhanced_path}: {e}")
    return float("nan")

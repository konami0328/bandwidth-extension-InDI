"""Inference step schedule strategies for InDI iterative reconstruction.

The InDI update rule (Eq. 4.5 in the thesis) moves from a degraded state x_t
toward the clean signal x_0 one step at a time:

    x_{t - δ} = (δ/t) * F_θ(x_t, t) + (1 - δ/t) * x_t

A step schedule defines the sequence of t values visited during inference,
e.g. [1.0, 0.5, 0.0] for uniform 2-step, or [1.0, 0.9, 0.0] for non-uniform
2-step (the thesis Section 4.2 non-uniform strategy).

All schedule functions take `num_steps` and return a numpy array of t values
starting at 1.0 and ending at 0.0, length = num_steps + 1.

Key insight from the thesis (Section 4.2): the degradation between t=1.0 and
t=0.5 is visually much more severe than between t=0.5 and t=0.0.  Taking a
smaller first step (e.g. 1.0 → 0.9) and a larger second step (0.9 → 0.0)
distributes the reconstruction effort more evenly across steps.
"""

import numpy as np
from typing import List


def uniform_schedule(num_steps: int) -> np.ndarray:
    """Equal-sized steps from t=1.0 to t=0.0.

    Example (num_steps=2): [1.0, 0.5, 0.0]
    Example (num_steps=4): [1.0, 0.75, 0.5, 0.25, 0.0]
    """
    return np.linspace(1.0, 0.0, num_steps + 1)


def nonuniform_schedule(num_steps: int, first_step_t: float) -> np.ndarray:
    """Two-region schedule: small first step, large remaining step(s).

    Motivated by the observation that the degradation path is highly
    non-linear: most perceptual change occurs near t=1.0.  By taking a
    smaller first step we avoid asking the model to do too much in one go.

    For num_steps=2: [1.0, first_step_t, 0.0]
    For num_steps=3: [1.0, first_step_t, first_step_t/2, 0.0]  (illustrative)

    The thesis experiments used first_step_t ∈ {0.95, 0.9, 0.75, 0.5, 0.25, 0.1}
    and found first_step_t=0.9 to be optimal for 2-step inference.

    Args:
        num_steps:     Total number of inference steps.
        first_step_t:  t value after the first step (e.g. 0.9 means the first
                       step moves from t=1.0 to t=0.9).
    Returns:
        Array of t values, shape (num_steps + 1,).
    Raises:
        ValueError: If first_step_t is not in (0.0, 1.0).
    """
    if not (0.0 < first_step_t < 1.0):
        raise ValueError(f"first_step_t must be in (0, 1), got {first_step_t}.")

    # First interval: 1.0 → first_step_t  (small step, hard region)
    # Remaining: first_step_t → 0.0  (remaining steps, easier region)
    remaining_steps = num_steps - 1
    tail = np.linspace(first_step_t, 0.0, remaining_steps + 1)
    return np.concatenate([[1.0], tail])


def concave_schedule(num_steps: int, power: float = 2.5) -> np.ndarray:
    """Steps that are denser near t=1.0 (concave curve).

    t_i = 1 - (i/N)^power

    Higher power → more steps concentrated near t=1.0.

    Args:
        num_steps: Total number of inference steps.
        power:     Exponent controlling the curvature (>1 for concave).
    """
    x = np.linspace(0.0, 1.0, num_steps + 1)
    schedule = 1.0 - x ** power
    schedule[-1] = 0.0  # ensure exact 0 at the end
    return schedule


def convex_schedule(num_steps: int, power: float = 2.5) -> np.ndarray:
    """Steps that are denser near t=0.0 (convex curve).

    t_i = (1 - i/N)^power

    Args:
        num_steps: Total number of inference steps.
        power:     Exponent controlling the curvature (>1 for convex).
    """
    x = np.linspace(0.0, 1.0, num_steps + 1)
    schedule = (1.0 - x) ** power
    schedule[-1] = 0.0
    return schedule


def sigmoid_schedule(num_steps: int, k: float = 5.0) -> np.ndarray:
    """S-shaped schedule: denser steps in the middle of the path.

    t_i = 1 - sigmoid(k * (2*i/N - 1))

    Args:
        num_steps: Total number of inference steps.
        k:         Sharpness of the S-curve (larger k → sharper transition).
    """
    x = np.linspace(0.0, 1.0, num_steps + 1)
    schedule = 1.0 - (1.0 / (1.0 + np.exp(-k * (2.0 * x - 1.0))))
    schedule[0]  = 1.0  # ensure exact endpoints
    schedule[-1] = 0.0
    return schedule


# ---------------------------------------------------------------------------
# Registry: maps CLI strategy names to callables
# ---------------------------------------------------------------------------

def get_schedule(strategy: str, num_steps: int, **kwargs) -> np.ndarray:
    """Return a t-value schedule by name.

    Args:
        strategy:  One of 'uniform', 'nonuniform', 'concave', 'convex',
                   'sigmoid'.
        num_steps: Number of inference steps.
        **kwargs:  Strategy-specific keyword arguments:
                   - nonuniform: first_step_t (float, default 0.9)
                   - concave/convex: power (float, default 2.5)
                   - sigmoid: k (float, default 5.0)
    Returns:
        np.ndarray of shape (num_steps + 1,), from 1.0 down to 0.0.
    Raises:
        ValueError: If strategy name is not recognised.
    """
    if strategy == "uniform":
        return uniform_schedule(num_steps)
    elif strategy == "nonuniform":
        return nonuniform_schedule(num_steps, first_step_t=kwargs.get("first_step_t", 0.9))
    elif strategy == "concave":
        return concave_schedule(num_steps, power=kwargs.get("power", 2.5))
    elif strategy == "convex":
        return convex_schedule(num_steps, power=kwargs.get("power", 2.5))
    elif strategy == "sigmoid":
        return sigmoid_schedule(num_steps, k=kwargs.get("k", 5.0))
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: uniform, nonuniform, concave, convex, sigmoid."
        )

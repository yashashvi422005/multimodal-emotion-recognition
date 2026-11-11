"""Temperature scaling for Phase 5 fusion, operating directly on
probability vectors (no raw logits are available from either branch —
the face branch's clip-level "probability" is already an average of
per-frame softmax outputs). This is still exact: for any probability
vector p produced by some softmax(z), z = log(p) + c for an unknown
per-sample constant c, and softmax is shift-invariant, so
softmax(log(p)/T) equals softmax(z/T) regardless of c. Using log(p) as
a pseudo-logit is therefore not an approximation.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _pseudo_logits(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, _EPS, 1.0))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """probs: (n, k). Returns calibrated probabilities; argmax is
    unchanged (temperature scaling is a monotonic rescaling of the
    logits), only the sharpness of the distribution changes."""
    z = _pseudo_logits(probs) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _nll(probs: np.ndarray, y_true_idx: np.ndarray, temperature: float) -> float:
    calibrated = apply_temperature(probs, temperature)
    true_probs = calibrated[np.arange(len(y_true_idx)), y_true_idx]
    return float(-np.mean(np.log(np.clip(true_probs, _EPS, 1.0))))


def fit_temperature(probs: np.ndarray, y_true_idx: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Grid-search the scalar temperature minimizing mean NLL on
    (probs, y_true_idx). Deterministic, no external optimizer dependency."""
    if grid is None:
        grid = np.geomspace(0.1, 10.0, 200)
    nlls = [_nll(probs, y_true_idx, t) for t in grid]
    return float(grid[int(np.argmin(nlls))])

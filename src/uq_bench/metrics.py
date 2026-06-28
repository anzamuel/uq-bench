"""
Metrics for evaluating prediction intervals.

Every metric takes the true targets and the interval endpoints and returns a scalar. The functions are pure numpy and make no assumption about the method that produced the intervals, so any method is scored the same way. See docs/metrics.pdf for the formal definitions.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "aisl",
    "evaluate",
    "mpiw",
    "nciw",
    "niw",
    "picp",
    "pinball",
]


def _as_1d(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64).ravel()


def picp(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Compute the prediction interval coverage probability."""
    y, lo, hi = _as_1d(y_true), _as_1d(lower), _as_1d(upper)
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lower: ArrayLike, upper: ArrayLike) -> float:
    """Compute the mean prediction interval width."""
    return float(np.mean(_as_1d(upper) - _as_1d(lower)))


def niw(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Compute the mean interval width normalized by the target range."""
    y = _as_1d(y_true)
    width = mpiw(lower, upper)
    data_range = float(np.max(y) - np.min(y))
    if data_range == 0.0:
        return width
    return width / data_range


def pinball(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> float:
    """Compute the average pinball loss of the two interval endpoints."""
    y, lo, hi = _as_1d(y_true), _as_1d(lower), _as_1d(upper)
    tau_lo, tau_hi = alpha / 2.0, 1.0 - alpha / 2.0
    loss_lo = np.where(y >= lo, tau_lo * (y - lo), (1.0 - tau_lo) * (lo - y))
    loss_hi = np.where(y >= hi, tau_hi * (y - hi), (1.0 - tau_hi) * (hi - y))
    return float((np.mean(loss_lo) + np.mean(loss_hi)) / 2.0)


def aisl(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float) -> float:
    """Compute the average interval score loss."""
    y, lo, hi = _as_1d(y_true), _as_1d(lower), _as_1d(upper)
    below = np.where(y < lo, (2.0 / alpha) * (lo - y), 0.0)
    above = np.where(y > hi, (2.0 / alpha) * (y - hi), 0.0)
    return float(np.mean((hi - lo) + below + above))


def nciw(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: float,
    center: ArrayLike | None = None,
) -> float:
    """Compute the normalized interval width at the calibrated coverage."""
    y, lo, hi = _as_1d(y_true), _as_1d(lower), _as_1d(upper)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return float("inf")
    f = (lo + hi) / 2.0 if center is None else _as_1d(center)
    left = f - lo
    right = hi - f
    left_safe = np.where(left > 0.0, left, 1.0)  # avoid divide by zero
    right_safe = np.where(right > 0.0, right, 1.0)
    c = np.zeros_like(y)
    c = np.where((y < f) & (left > 0.0), (f - y) / left_safe, c)
    c = np.where((y > f) & (right > 0.0), (y - f) / right_safe, c)
    sorted_c = np.sort(c)
    n = sorted_c.size
    picps = np.arange(1, n + 1) / n
    idx = int(np.searchsorted(picps, 1.0 - alpha))
    c_cal = sorted_c[idx] if idx < n else sorted_c[-1]
    return float(c_cal) * niw(y, lo, hi)


def evaluate(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: float,
    center: ArrayLike | None = None,
) -> dict[str, float]:
    """Compute every interval metric and return them as a mapping."""
    return {
        "picp": picp(y_true, lower, upper),
        "mpiw": mpiw(lower, upper),
        "niw": niw(y_true, lower, upper),
        "pinball": pinball(y_true, lower, upper, alpha),
        "aisl": aisl(y_true, lower, upper, alpha),
        "nciw": nciw(y_true, lower, upper, alpha, center=center),
    }

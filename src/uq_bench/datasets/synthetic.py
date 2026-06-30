"""
Synthetic regression datasets.

Generators build a Dataset from a known data-generating process, so a method's intervals can be checked against controllable noise. The heteroscedastic family grows its noise with the input magnitude, so intervals should widen away from the origin.
"""

import numpy as np

from uq_bench.datasets.core import Dataset

__all__ = ["make_heteroscedastic"]


def make_heteroscedastic(n: int = 1000, seed: int = 0) -> Dataset:
    """Generate a 1-D regression dataset whose noise grows with the input magnitude."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 1))
    noise = rng.normal(0.0, 1.0 + np.abs(x[:, 0]))
    y = np.sin(3.0 * x[:, 0]) + noise
    return Dataset(name="heteroscedastic", X=x, y=y)

"""
Synthetic regression datasets.

Generators build a Dataset from a known data-generating process, so a method's intervals can be checked against controllable noise. Use list_synthetic to list them and load_synthetic to build one. The heteroscedastic family grows its noise with the input magnitude, so intervals should widen away from the origin.
"""

import numpy as np

from uq_bench.datasets.core import Dataset

__all__ = ["list_synthetic", "load_synthetic"]


def _heteroscedastic(n: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 1))
    noise = rng.normal(0.0, 1.0 + np.abs(x[:, 0]))
    y = np.sin(3.0 * x[:, 0]) + noise
    return Dataset(name="heteroscedastic", X=x, y=y)


_GENERATORS = {"heteroscedastic": _heteroscedastic}


def list_synthetic() -> tuple[str, ...]:
    """List the synthetic dataset names."""
    return tuple(_GENERATORS)


def load_synthetic(name: str, n: int = 1000, seed: int = 0) -> Dataset:
    """Generate a synthetic dataset by name."""
    if name not in _GENERATORS:
        raise ValueError(f"unknown dataset {name!r}")
    return _GENERATORS[name](n, seed)

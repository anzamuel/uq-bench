"""
Datasets and their splits.

A Dataset is a named feature matrix and target vector. A Split is one dataset's train, cal, and test parts.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Dataset",
    "Split",
    "partition",
]

type Array = NDArray[np.float64]


@dataclass(frozen=True)
class Dataset:
    """A named feature matrix and target vector."""

    name: str
    X: Array
    y: Array


@dataclass(frozen=True)
class Split:
    """A train, cal, and test split of a dataset, each part a feature-target pair."""

    train: tuple[Array, Array]
    cal: tuple[Array, Array]
    test: tuple[Array, Array]


def partition(
    dataset: Dataset, seed: int, ratio: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> Split:
    """Partition a dataset into a train, cal, and test split deterministically from a seed."""
    if any(r <= 0.0 for r in ratio) or abs(sum(ratio) - 1.0) > 1e-9:
        raise ValueError(f"ratio must be positive and sum to 1, got {ratio}")
    n = dataset.y.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    n_train = int(n * ratio[0])
    n_cal = int(n * ratio[1])
    tr, ca, te = np.split(idx, [n_train, n_train + n_cal])
    return Split(
        train=(dataset.X[tr], dataset.y[tr]),
        cal=(dataset.X[ca], dataset.y[ca]),
        test=(dataset.X[te], dataset.y[te]),
    )

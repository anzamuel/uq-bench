"""
Datasets and their splits.

A Dataset is a named feature matrix and target vector. A Split is one dataset's train and test parts; a method sub-splits the train part for its own calibration.
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
    """A train and test split of a dataset, each part a feature-target pair."""

    train: tuple[Array, Array]
    test: tuple[Array, Array]


def partition(
    dataset: Dataset, seed: int, ratio: tuple[float, float] = (0.8, 0.2)
) -> Split:
    """Partition a dataset into a train and test split deterministically from a seed."""
    if any(r <= 0.0 for r in ratio) or abs(sum(ratio) - 1.0) > 1e-9:
        raise ValueError(f"ratio must be positive and sum to 1, got {ratio}")
    n = dataset.y.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    n_train = int(n * ratio[0])
    tr, te = np.split(idx, [n_train])
    return Split(
        train=(dataset.X[tr], dataset.y[tr]),
        test=(dataset.X[te], dataset.y[te]),
    )

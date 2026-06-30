"""
Benchmark runner.

Runs a method on a dataset by handing it a split over an npz file and scoring the intervals it returns, with the method in its own `uv` environment so the bench shares no dependencies. `run_grid` sweeps that across the method, dataset, seed, and coverage axes.
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from uq_bench import metrics
from uq_bench.datasets import Dataset, partition

__all__ = ["Result", "run", "run_grid"]


@dataclass(frozen=True)
class Result:
    """One grid cell: a method on a dataset at a given seed and coverage."""

    method: str
    dataset: str
    seed: int
    coverage: float
    metrics: dict[str, float]


def run(dataset: Dataset, method: Path, seed: int, coverage: float) -> dict[str, float]:
    """Partition the dataset, run the method over npz, and score its intervals."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv not found on PATH")
    split = partition(dataset, seed)
    with tempfile.TemporaryDirectory() as tmp:
        split_path = Path(tmp) / "split.npz"
        preds_path = Path(tmp) / "preds.npz"
        np.savez(
            split_path,
            X_train=split.train[0],
            y_train=split.train[1],
            X_cal=split.cal[0],
            y_cal=split.cal[1],
            X_test=split.test[0],
        )
        cmd = [uv, "run", str(method), str(split_path), str(preds_path), str(coverage)]
        subprocess.run(cmd, check=True)  # noqa: S603
        preds = np.load(preds_path)
        return metrics.evaluate(
            split.test[1], preds["lower"], preds["upper"], alpha=1.0 - coverage
        )


def run_grid(
    methods: list[Path],
    datasets: list[Dataset],
    seeds: list[int],
    coverages: list[float],
) -> list[Result]:
    """Run every method on every dataset across the seeds and coverages."""
    results: list[Result] = []
    for method, dataset, seed, coverage in product(methods, datasets, seeds, coverages):
        scores = run(dataset, method, seed, coverage)
        results.append(Result(method.stem, dataset.name, seed, coverage, scores))
    return results

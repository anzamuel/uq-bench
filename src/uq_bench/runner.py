"""
Benchmark runner.

Runs a method on a dataset by handing it a train and test split over an npz file, along with the seed so the method's own calibration split is reproducible, and scoring the intervals it returns; the method runs as its own `uv` project so the bench shares no dependencies. A method writes `lower` and `upper` arrays and may add a `center` array with its point predictions, which then anchors the NCIW rescaling instead of the interval midpoint. `run_grid` sweeps that across the method, dataset, seed, and coverage axes.
"""

import os
import shutil
import subprocess
import tempfile
from itertools import product
from pathlib import Path

import numpy as np

from uq_bench import metrics
from uq_bench.datasets import Dataset, partition
from uq_bench.results import Result

__all__ = ["run", "run_grid"]


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
            X_test=split.test[0],
        )
        cmd = [
            uv,
            "run",
            "--directory",
            str(method),
            "main.py",
            str(split_path),
            str(preds_path),
            str(coverage),
            str(seed),
        ]
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)  # let uv use the method's own project env
        subprocess.run(cmd, check=True, env=env)  # noqa: S603
        preds = np.load(preds_path)
        center = preds["center"] if "center" in preds.files else None
        return metrics.evaluate(
            split.test[1],
            preds["lower"],
            preds["upper"],
            alpha=1.0 - coverage,
            center=center,
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
        results.append(Result(method.name, dataset.name, seed, coverage, scores))
    return results

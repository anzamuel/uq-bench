"""
Benchmark runner.

Runs a method on a dataset by handing it a split over an npz file and scoring the intervals it returns. The method runs in its own environment via `uv run`, so the bench shares no dependencies with it.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from uq_bench import metrics
from uq_bench.datasets import Dataset, partition

__all__ = ["run"]


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

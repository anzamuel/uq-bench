"""
Benchmark runner.

Runs a method on a dataset by handing it a train and test split over an npz file, along with the seed so the method's own calibration split is reproducible, and scoring the intervals it returns; the method runs as its own `uv` project so the bench shares no dependencies. A method writes `lower` and `upper` arrays and may add a `center` array with its point predictions, which then anchors the NCIW rescaling instead of the interval midpoint. `run_grid` sweeps that across the method, dataset, seed, and coverage axes over worker threads, one per CPU by default, appending each cell to a results CSV as it finishes and skipping cells already present, so an interrupted sweep resumes where it left off; cells run largest dataset first so the end-of-phase straggler is a small one, and the call is meant to be made once per method with phase-appropriate workers when methods differ in resource shape. See methods/README.md for the full method contract.
"""

import csv
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
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


def _read_results(path: Path) -> list[Result]:
    if not path.exists():
        return []
    keys = ("method", "dataset", "seed", "coverage")
    with path.open(newline="") as f:
        return [
            Result(
                row["method"],
                row["dataset"],
                int(row["seed"]),
                float(row["coverage"]),
                {m: float(v) for m, v in row.items() if m not in keys},
            )
            for row in csv.DictReader(f)
        ]


def run_grid(
    methods: list[Path],
    datasets: list[Dataset],
    seeds: list[int],
    coverages: list[float],
    results_path: Path = Path("results.csv"),
    workers: int | None = None,
) -> list[Result]:
    """Run every method on every dataset across the seeds and coverages."""
    if workers is None:
        workers = os.process_cpu_count() or 4  # scheduling only, never affects results
    results = _read_results(results_path)
    done = {(r.method, r.dataset, r.seed, r.coverage) for r in results}
    cells = [
        cell
        for cell in product(methods, datasets, seeds, coverages)
        if (cell[0].name, cell[1].name, cell[2], cell[3]) not in done
    ]
    cells.sort(
        key=lambda cell: cell[1].X.size, reverse=True
    )  # largest work first, shortest straggler tail
    lock = threading.Lock()
    with results_path.open("a", newline="") as f:
        writer = csv.writer(f)

        def one(cell: tuple[Path, Dataset, int, float]) -> Result:
            method, dataset, seed, coverage = cell
            scores = run(dataset, method, seed, coverage)
            with lock:
                if f.tell() == 0:
                    writer.writerow(["method", "dataset", "seed", "coverage", *scores])
                writer.writerow(
                    [method.name, dataset.name, seed, coverage, *scores.values()]
                )
                f.flush()
            return Result(method.name, dataset.name, seed, coverage, scores)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results += pool.map(one, cells)
    return sorted(results, key=lambda r: (r.method, r.dataset, r.seed, r.coverage))

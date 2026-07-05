"""
Result aggregation.

A Result records one method's metrics on one dataset at a given seed and coverage. A Summary reduces a method, dataset, and coverage group to the mean and standard deviation of each metric, and write_csv writes those as a wide table.
"""

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Result",
    "Summary",
    "aggregate",
    "write_csv",
]


@dataclass(frozen=True)
class Result:
    """One grid cell: a method on a dataset at a given seed and coverage."""

    method: str
    dataset: str
    seed: int
    coverage: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class Summary:
    """The mean and standard deviation of each metric over seeds for one group."""

    method: str
    dataset: str
    coverage: float
    n_seeds: int
    mean: dict[str, float]
    std: dict[str, float]


def aggregate(results: list[Result]) -> list[Summary]:
    """Reduce results to per-metric mean and standard deviation over seeds."""
    groups: dict[tuple[str, str, float], list[dict[str, float]]] = defaultdict(list)
    for r in results:
        groups[(r.method, r.dataset, r.coverage)].append(r.metrics)
    summaries: list[Summary] = []
    for (method, dataset, coverage), runs in groups.items():
        mean = {m: statistics.fmean(run[m] for run in runs) for m in runs[0]}
        std = {
            m: statistics.stdev(run[m] for run in runs) if len(runs) > 1 else 0.0
            for m in runs[0]
        }
        summaries.append(Summary(method, dataset, coverage, len(runs), mean, std))
    return summaries


def write_csv(summaries: list[Summary], path: Path) -> None:
    """Write summaries as a wide CSV with a mean and std column per metric."""
    if not summaries:
        return
    metrics = list(summaries[0].mean)
    header = ["method", "dataset", "coverage", "n_seeds"]
    for m in metrics:
        header += [f"{m}_mean", f"{m}_std"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for s in summaries:
            row = [s.method, s.dataset, s.coverage, s.n_seeds]
            for m in metrics:
                row += [round(s.mean[m], 6), round(s.std[m], 6)]
            writer.writerow(row)

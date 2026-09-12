"""
Benchmark driver.

Runs the paper grid one method phase at a time through `run_grid`, resuming from `results.csv`, with per-phase workers sized to each method's resource shape per the `threading` line of its README, threads times workers close to cores. Phases go cheapest first so an interrupted run keeps its quick wins, and ctabpfn-a runs before its siblings so they ride the shared inference cache; within a phase the runner already orders cells largest first. Datasets are the six paper picks, seeds the ten replication seeds, coverage 0.9 only.
"""

import os
from pathlib import Path

from uq_bench.datasets import load_real
from uq_bench.runner import run_grid

ROOT = Path(__file__).parent
DATASETS = (
    "energy_efficiency",
    "concrete",
    "insurance",
    "airfoil",
    "debutanizer",
    "parkinsons",
    "computer",
    "kin8nm",
    "powerplant",
    "sulfur",
    "naval_propulsion",
    "miami_housing",
)
SEEDS = tuple(range(777, 787))  # the replication seeds
COVERAGES = (0.9,)
CORES = os.process_cpu_count() or 4
CLEAR_JOBS = int(os.environ.get("CLEAR_JOBS", "4"))
TORCH_WORKERS = min(4, max(2, CORES // 4))
PHASES = (
    ("baseline", CORES),
    ("tabpfn", TORCH_WORKERS),
    ("ctabpfn-a", TORCH_WORKERS),  # fills the shared inference cache for m and q
    ("ctabpfn-m", TORCH_WORKERS),
    ("ctabpfn-q", TORCH_WORKERS),
    ("uacqr", max(1, CORES // 2)),
    ("clear", max(1, CORES // CLEAR_JOBS)),
    ("pcs", CORES),
)


def main() -> None:
    """Run every phase of the benchmark grid, resuming where results.csv left off."""
    datasets = [load_real(name) for name in DATASETS]
    for method, workers in PHASES:
        print(f"phase {method}: {workers} workers", flush=True)
        run_grid(
            [ROOT / "methods" / method],
            datasets,
            seeds=list(SEEDS),
            coverages=list(COVERAGES),
            results_path=ROOT / "results.csv",
            workers=workers,
        )


if __name__ == "__main__":
    main()

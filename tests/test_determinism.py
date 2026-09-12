"""
Determinism tests over every method.

Every project under methods/ is discovered and run through the harness: the same seed twice must give identical metrics and a different seed must change them. UQ_BENCH_FAST asks each method for its own cheap settings so the suite stays method agnostic and fast.
"""

from pathlib import Path

import numpy as np
import pytest

from uq_bench import runner
from uq_bench.datasets import Dataset

METHODS = Path(__file__).parent.parent / "methods"


def _methods() -> list[str]:
    return sorted(p.parent.name for p in METHODS.glob("*/main.py"))


@pytest.fixture(scope="module")
def dataset():
    # low noise: heavy noise legitimately drives uacqr to all-infinite
    # intervals whose metrics cannot distinguish seeds
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=400)
    return Dataset(name="determinism", X=X, y=y)


@pytest.mark.parametrize("method", _methods())
def test_run_is_deterministic_in_the_seed(dataset, method, monkeypatch):
    monkeypatch.setenv("UQ_BENCH_FAST", "1")
    first = runner.run(dataset, METHODS / method, seed=0, coverage=0.9)
    second = runner.run(dataset, METHODS / method, seed=0, coverage=0.9)
    other = runner.run(dataset, METHODS / method, seed=1, coverage=0.9)
    assert first == second
    assert first != other


def test_run_grid_persists_and_resumes(dataset, tmp_path):
    csv_path = tmp_path / "results.csv"

    def grid():
        return runner.run_grid(
            [METHODS / "baseline"],
            [dataset],
            seeds=[0, 1],
            coverages=[0.9],
            results_path=csv_path,
        )

    first = grid()
    content = csv_path.read_text()
    assert len(first) == 2
    assert len(content.splitlines()) == 3  # header plus one row per cell
    second = grid()
    assert second == first  # cells read back from the csv, not rerun
    assert csv_path.read_text() == content

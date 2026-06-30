from uq_bench.datasets.core import Dataset, Split, partition
from uq_bench.datasets.real import list_real, load_real
from uq_bench.datasets.synthetic import list_synthetic, load_synthetic

__all__ = [
    "Dataset",
    "Split",
    "list_real",
    "list_synthetic",
    "load_real",
    "load_synthetic",
    "partition",
]

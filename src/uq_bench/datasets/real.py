"""
Real-world datasets.

Each named dataset is one Parquet table in the dataset hub repository, fetched into the local cache and returned as a Dataset. Use list_real to list them and load_real to load one. Features are cast to float64 and the target is the __target__ column.
"""

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from uq_bench.datasets.core import Dataset

__all__ = ["list_real", "load_real"]

_HF_REPO_ID = "anzamuel/uq-bench"


def list_real() -> tuple[str, ...]:
    """List the dataset names in the hub repository."""
    files = list_repo_files(_HF_REPO_ID, repo_type="dataset")
    return tuple(
        sorted(f.removesuffix(".parquet") for f in files if f.endswith(".parquet"))
    )


def load_real(name: str) -> Dataset:
    """Fetch a real-world dataset from the hub cache and return it as a Dataset."""
    path = hf_hub_download(_HF_REPO_ID, f"{name}.parquet", repo_type="dataset")
    table = pq.read_table(path)
    y = table.column("__target__").to_numpy().astype(np.float64)
    feat = [c for c in table.column_names if c != "__target__"]
    X = np.column_stack([table.column(c).to_numpy() for c in feat]).astype(np.float64)
    return Dataset(name=name, X=X, y=y)

"""
CTABPFN-M adapter over the ctabpfn package.

Splits the train part 75/25 at the given seed, reads the model's native quantile grid through the shared content-addressed cache, and scales the two half widths around the native median by the exact conformal order statistic of the ratio score. Reads the split from argv[1], writes lower, upper, and center test bounds to argv[2], and takes the target coverage as argv[3] and the seed as argv[4].
"""

import sys

import numpy as np
from ctabpfn import multiplicative


def main() -> None:
    """Run the multiplicative conformal TabPFN variant over the npz contract."""
    split_path, preds_path = sys.argv[1], sys.argv[2]
    coverage, seed = float(sys.argv[3]), int(sys.argv[4])
    data = np.load(split_path)
    lower, upper, center = multiplicative(
        data["X_train"], data["y_train"], data["X_test"], coverage, seed
    )
    np.savez(preds_path, lower=lower, upper=upper, center=center)


if __name__ == "__main__":
    main()

"""
Metric correctness tests.

Hand-computed values for every interval metric, plus the degenerate cases the CLEAR reference implementation defines: zero-width intervals fall back to a zero covering scale and infinite bounds give an infinite NCIW.
"""

import numpy as np
import pytest

from uq_bench import metrics

Y = np.array([0.0, 1.0, 2.0, 3.0])
LO = np.array([0.0, 2.0, 0.0, 0.0])
HI = np.array([1.0, 3.0, 1.0, 4.0])


def test_picp():
    assert metrics.picp(Y, LO, HI) == 0.5


def test_mpiw():
    assert metrics.mpiw(LO, HI) == 1.75


def test_niw():
    assert metrics.niw(Y, LO, HI) == pytest.approx(1.75 / 3.0)


def test_niw_zero_range_falls_back_to_width():
    y = np.array([1.0, 1.0])
    assert metrics.niw(y, y - 1.0, y + 1.0) == 2.0


def test_pinball():
    got = metrics.pinball([1.0], [0.0], [2.0], alpha=0.2)
    assert got == pytest.approx(0.1)  # 0.1*(1-0) below, 0.1*(2-1) above, halved


def test_aisl():
    got = metrics.aisl([1.0, 5.0], [0.0, 0.0], [2.0, 2.0], alpha=0.2)
    assert got == pytest.approx(17.0)  # covered width 2, miss 2 + 10*(5-2)


def test_nciw_symmetric():
    y = np.array([0.5, -0.5, 1.5, -2.0])
    lo, hi = np.full(4, -1.0), np.full(4, 1.0)
    got = metrics.nciw(y, lo, hi, alpha=0.5)
    assert got == pytest.approx(0.5 * 2.0 / 3.5)  # c*=0.5 at 50% target


def test_nciw_center_shifts_scale():
    y = np.array([0.0, 0.0, 4.0, 4.0])
    lo, hi = np.full(4, 0.0), np.full(4, 4.0)
    center = np.full(4, 1.0)  # asymmetric: left width 1, right width 3
    got = metrics.nciw(y, lo, hi, alpha=0.0, center=center)
    assert got == pytest.approx(1.0)  # c*=max(1/1, 3/3)=1, niw=1


def test_nciw_zero_width_counts_as_covered():
    y = np.array([1.0, 2.0, 3.0])
    q = np.array([1.0, 1.0, 1.0])
    assert metrics.nciw(y, q, q, alpha=0.5) == 0.0  # degenerate c=0 fallback


def test_nciw_infinite_bounds():
    y = np.array([0.0, 1.0])
    assert metrics.nciw(y, [-1.0, -1.0], [1.0, np.inf], alpha=0.1) == np.inf


def test_evaluate_keys():
    got = metrics.evaluate(Y, LO, HI, alpha=0.1)
    assert set(got) == {"picp", "mpiw", "niw", "pinball", "aisl", "nciw"}

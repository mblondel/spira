import numpy as np
import scipy.sparse as sp

from spira.impl.dataset import mean

from testing import assert_array_almost_equal

X = [[3, 0, 0, 1],
     [2, 0, 5, 0],
     [0, 4, 3, 0],
     [0, 0, 2, 0],
     [1, 0, 0, 0]]


def test_mean_axis1():
    expected = [2, 3.5, 3.5, 2, 1]
    for func in (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix):
        means = mean(func(X, dtype=np.float64))
        assert_array_almost_equal(means, expected)


def test_mean_axis0():
    expected = [2, 4, 10./3, 1]
    for func in (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix):
        means = mean(func(X, dtype=np.float64), axis=0)
    assert_array_almost_equal(means, expected)

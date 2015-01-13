import numpy as np
import scipy.sparse as sp


from .dataset_fast import _mean_axis0_csr
from .dataset_fast import _mean_axis1_csr


def _mean_csr(X, axis, means):
    if axis == 0:
        _mean_axis0_csr(X.data, X.indices, X.indptr, means)
    else:
        _mean_axis1_csr(X.data, X.indices, X.indptr, means)


def _mean_csc(X, axis, means):
    # CSC is simply the transposed of CSR.
    if axis == 0:
        _mean_axis1_csr(X.data, X.indices, X.indptr, means)
    else:
        _mean_axis0_csr(X.data, X.indices, X.indptr, means)


def _mean_coo(X, axis, means):
    if axis == 0:
        indices = X.col
    else:
        indices = X.row

    sums = np.array(X.sum(axis=axis)).ravel()
    counts = np.bincount(indices, minlength=means.shape[0])
    counts[counts == 0] = 1  # To avoid division by zero.
    means[:] = sums / counts


def mean(X, axis=1):

    if not axis in (0, 1):
        raise ValueError("Invalid axis.")

    if axis == 0:
        means = np.zeros(X.shape[1], dtype=np.float64)
    elif axis == 1:
        means = np.zeros(X.shape[0], dtype=np.float64)

    if sp.isspmatrix_coo(X):
        _mean_coo(X, axis, means)
    elif sp.isspmatrix_csc(X):
        _mean_csc(X, axis, means)
    else:
        X = sp.csr_matrix(X)
        _mean_csr(X, axis, means)

    return means

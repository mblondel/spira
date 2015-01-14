# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np


def _transform_csr(X,
                   np.ndarray[double, ndim=1] mean,
                   np.ndarray[double, ndim=1] std,
                   int with_mean,
                   int with_std,
                   int axis):
    cdef int n_rows = X.shape[0]

    cdef np.ndarray[double, ndim=1] X_data = X.data
    cdef np.ndarray[int, ndim=1] X_indices = X.indices
    cdef np.ndarray[int, ndim=1] X_indptr = X.indptr

    cdef int n_nz
    cdef double* data
    cdef int* indices

    cdef int u, ii, i, n

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]
        indices = <int*>X_indices.data + X_indptr[u]

        for ii in xrange(n_nz):
            i = indices[ii]
            n = u if axis == 1 else i
            if with_mean:
                data[ii] -= mean[n]
            if with_std:
                if std[n] > 0:
                    data[ii] /= std[n]


def _inverse_transform_csr(X,
                           np.ndarray[double, ndim=1] mean,
                           np.ndarray[double, ndim=1] std,
                           int with_mean,
                           int with_std,
                           int axis):
    cdef int n_rows = X.shape[0]

    cdef np.ndarray[double, ndim=1] X_data = X.data
    cdef np.ndarray[int, ndim=1] X_indices = X.indices
    cdef np.ndarray[int, ndim=1] X_indptr = X.indptr

    cdef int n_nz
    cdef double* data
    cdef int* indices

    cdef int u, ii, i, n

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]
        indices = <int*>X_indices.data + X_indptr[u]

        for ii in xrange(n_nz):
            i = indices[ii]
            n = u if axis == 1 else i
            if with_std:
                if std[n] > 0:
                    data[ii] *= std[n]
            if with_mean:
                data[ii] += mean[n]

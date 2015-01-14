# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np

from libc.math cimport sqrt


def _mean_axis1_csr(X, np.ndarray[double, ndim=1] mean):
    cdef int n_rows = X.shape[0]

    cdef np.ndarray[double, ndim=1] X_data = X.data
    cdef np.ndarray[int, ndim=1] X_indices = X.indices
    cdef np.ndarray[int, ndim=1] X_indptr = X.indptr

    cdef int n_nz
    cdef double* data

    cdef int u, ii

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]

        if n_nz > 0:
            for ii in xrange(n_nz):
                mean[u] += data[ii]

            mean[u] /= n_nz


def _mean_axis0_csr(X, np.ndarray[double, ndim=1] mean):
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]

    cdef np.ndarray[double, ndim=1] X_data = X.data
    cdef np.ndarray[int, ndim=1] X_indices = X.indices
    cdef np.ndarray[int, ndim=1] X_indptr = X.indptr

    cdef int n_nz
    cdef double* data
    cdef int* indices

    cdef int u, ii, i
    cdef np.ndarray[int, ndim=1] count = np.zeros(n_cols, dtype=np.int32)

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]
        indices = <int*>X_indices.data + X_indptr[u]

        if n_nz > 0:
            for ii in xrange(n_nz):
                i = indices[ii]
                mean[i] += data[ii]
                count[i] += 1

    for i in xrange(n_cols):
        if count[i] > 0:
            mean[i] /= count[i]


def _std_axis1_csr(X,
                   np.ndarray[double, ndim=1] mean,
                   np.ndarray[double, ndim=1] std):
    cdef int n_rows = X.shape[0]

    cdef np.ndarray[double, ndim=1] X_data = X.data
    cdef np.ndarray[int, ndim=1] X_indices = X.indices
    cdef np.ndarray[int, ndim=1] X_indptr = X.indptr

    cdef int n_nz
    cdef double* data

    cdef int u, ii
    cdef double diff

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]

        if n_nz > 0:
            for ii in xrange(n_nz):
                diff = data[ii] - mean[u]
                std[u] += diff * diff

            std[u] /= n_nz
            std[u] = sqrt(std[u])


def _std_axis0_csr(X,
                   np.ndarray[double, ndim=1] mean,
                   np.ndarray[double, ndim=1] std):

    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]

    cdef np.ndarray[double, ndim=1] X_data = X.data
    cdef np.ndarray[int, ndim=1] X_indices = X.indices
    cdef np.ndarray[int, ndim=1] X_indptr = X.indptr

    cdef int n_nz
    cdef double* data
    cdef int* indices

    cdef int u, ii, i
    cdef double diff
    cdef np.ndarray[int, ndim=1] count = np.zeros(n_cols, dtype=np.int32)

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]
        indices = <int*>X_indices.data + X_indptr[u]

        for ii in xrange(n_nz):
            i = indices[ii]
            diff = data[ii] - mean[i]
            std[i] += diff * diff
            count[i] += 1

    for i in xrange(n_cols):
        if count[i] > 0:
            std[i] = sqrt(std[i] / count[i])

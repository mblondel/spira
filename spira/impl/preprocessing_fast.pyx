# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np


def _transform_csr(np.ndarray[double, ndim=1] X_data,
                   np.ndarray[int, ndim=1] X_indices,
                   np.ndarray[int, ndim=1] X_indptr,
                   np.ndarray[double, ndim=1] mean,
                   np.ndarray[double, ndim=1] std,
                   int with_mean,
                   int with_std):

    cdef int n_rows = mean.shape[0]

    cdef int n_nz
    cdef double* data
    cdef int* indices

    cdef int u, ii

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]
        indices = <int*>X_indices.data + X_indptr[u]

        for ii in xrange(n_nz):
            if with_mean:
                data[ii] -= mean[u]
            if with_std:
                if std[u] > 0:
                    data[ii] /= std[u]


def _inverse_transform_csr(np.ndarray[double, ndim=1] X_data,
                           np.ndarray[int, ndim=1] X_indices,
                           np.ndarray[int, ndim=1] X_indptr,
                           np.ndarray[double, ndim=1] mean,
                           np.ndarray[double, ndim=1] std,
                           int with_mean,
                           int with_std):

    cdef int n_rows = mean.shape[0]

    cdef int n_nz
    cdef double* data
    cdef int* indices

    cdef int u, ii

    for u in xrange(n_rows):
        n_nz = X_indptr[u+1] - X_indptr[u]
        data = <double*>X_data.data + X_indptr[u]
        indices = <int*>X_indices.data + X_indptr[u]

        for ii in xrange(n_nz):
            if with_std:
                if std[u] > 0:
                    data[ii] *= std[u]
            if with_mean:
                data[ii] += mean[u]

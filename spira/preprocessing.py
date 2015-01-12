# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from .impl.preprocessing_fast import _mean_csr
from .impl.preprocessing_fast import _std_csr
from .impl.preprocessing_fast import _transform_csr
from .impl.preprocessing_fast import _inverse_transform_csr


class StandardScaler(object):

    def __init__(self, with_mean=True, with_std=False, copy=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X):
        X = sp.csr_matrix(X)
        self.mean_ = np.zeros(X.shape[0])
        self.std_ = np.zeros(X.shape[0])
        _mean_csr(X.data, X.indices, X.indptr, self.mean_)
        _std_csr(X.data, X.indices, X.indptr, self.mean_, self.std_)
        return self

    def _check_data(self, X):
        if sp.isspmatrix_csr(X):
            if self.copy:
                X = X.copy()
        else:
            X = sp.csr_matrix(X)

        return X

    def transform(self, X):
        X = self._check_data(X)

        _transform_csr(X.data, X.indices, X.indptr, self.mean_, self.std_,
                       self.with_mean, self.with_std)

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = self._check_data(X)

        _inverse_transform_csr(X.data, X.indices, X.indptr, self.mean_,
                               self.std_, self.with_mean, self.with_std)

        return X

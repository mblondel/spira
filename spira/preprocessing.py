# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from .impl.dataset_fast import _mean_axis1_csr
from .impl.dataset_fast import _std_axis1_csr
from .impl.dataset_fast import _mean_axis0_csr
from .impl.dataset_fast import _std_axis0_csr
from .impl.preprocessing_fast import _transform_csr
from .impl.preprocessing_fast import _inverse_transform_csr


class StandardScaler(object):

    def __init__(self, axis=1, with_mean=True, with_std=False, copy=True):
        self.axis = axis
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X):
        X = sp.csr_matrix(X)

        if self.axis == 1:
            self.mean_ = np.zeros(X.shape[0])
            self.std_ = np.zeros(X.shape[0])
            _mean_axis1_csr(X, self.mean_)
            _std_axis1_csr(X, self.mean_, self.std_)

        elif self.axis == 0:
            self.mean_ = np.zeros(X.shape[1])
            self.std_ = np.zeros(X.shape[1])
            _mean_axis0_csr(X, self.mean_)
            _std_axis0_csr(X, self.mean_, self.std_)

        else:
            raise ValueError("Incorrect axis.")

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

        _transform_csr(X, self.mean_, self.std_, self.with_mean, self.with_std,
                       self.axis)

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = self._check_data(X)

        _inverse_transform_csr(X, self.mean_, self.std_, self.with_mean,
                               self.with_std, self.axis)

        return X

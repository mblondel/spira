# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from .dataset import mean


class Dummy(object):
    """
    An estimator which always predict the row-wise (axis=1)
    or column-wise (axis=0) mean.
    """

    def __init__(self, axis=1):
        self.axis = axis

    def fit(self, X):
        X = sp.csr_matrix(X)
        self.mean_ = mean(X, axis=self.axis)
        return self

    def predict(self, X):
        if sp.isspmatrix_csr(X):
            X = X.copy()
        else:
            X = sp.csr_matrix(X)

        if self.axis == 0:
            X.data = self.mean_[X.indices]

        elif self.axis == 1:
            n_observed = np.diff(X.indptr)
            X.data = np.repeat(self.mean_, n_observed)

        else:
            raise ValueError("Invalid axis.")

        return X

    def score(self, X):
        X = sp.csr_matrix(X)
        X_predicted = self.predict(X)
        mse = np.mean((X.data - X_predicted.data) ** 2)
        return np.sqrt(mse)

# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from .preprocessing_fast import _mean_csr


class Dummy(object):

    def fit(self, X):
        X = sp.csr_matrix(X)
        self.mean_ = np.zeros(X.shape[0])
        _mean_csr(X.data, X.indices, X.indptr, self.mean_)
        return self

    def predict(self, X):
        if sp.isspmatrix_csr(X):
            X = X.copy()
        else:
            X = sp.csr_matrix(X)

        n_observed = np.diff(X.indptr)
        X.data = np.repeat(self.mean_, n_observed)
        return X

    def score(self, X):
        X = sp.csr_matrix(X)
        X_predicted = self.predict(X)
        mse = np.mean((X.data - X_predicted.data) ** 2)
        return np.sqrt(mse)

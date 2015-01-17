# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

# FIXME: don't depend on scikit-learn.
from sklearn.base import BaseEstimator

from .matrix_fact_fast import _cd_fit, _predict
from ..metrics import rmse


class MatrixFactorization(BaseEstimator):

    def __init__(self, alpha=1.0, n_components=30, max_iter=10, tol=1e-3,
                 callback=None, random_state=None, verbose=0):
        self.alpha = alpha
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.random_state = random_state
        self.verbose = verbose

    def _init(self, X, rng):
        n_rows, n_cols = X.shape
        P = np.zeros((n_rows, self.n_components), order="C")
        Q = rng.rand(self.n_components, n_cols)
        Q = np.asfortranarray(Q)
        return P, Q

    def fit(self, X):
        X = sp.csr_matrix(X, dtype=np.float64)
        n_rows, n_cols = X.shape
        n_data = len(X.data)

        # Initialization.
        rng = np.random.RandomState(self.random_state)
        self.P_, self.Q_ = self._init(X, rng)

        residuals = np.empty(n_data, dtype=np.float64)
        n_max = max(n_rows, n_cols)
        g = np.empty(n_max, dtype=np.float64)
        h = np.empty(n_max, dtype=np.float64)
        delta = np.empty(n_max, dtype=np.float64)

        # Model estimation.
        _cd_fit(self, X.data, X.indices, X.indptr, self.P_, self.Q_, residuals,
                g, h, delta, self.n_components, self.alpha, self.max_iter,
                self.tol, self.callback, self.verbose)

        return self

    def predict(self, X):
        X = sp.csr_matrix(X)
        out = np.zeros_like(X.data)
        _predict(out, X.indices, X.indptr, self.P_, self.Q_)
        return sp.csr_matrix((out, X.indices, X.indptr), shape=X.shape)

    def score(self, X):
        X = sp.csr_matrix(X)
        X_pred = self.predict(X)
        return rmse(X, X_pred)

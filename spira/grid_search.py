from functools import partial, reduce
from itertools import product
import operator
from collections import Mapping

import numpy as np

# FIXME: don't depend on scikit-learn.
from sklearn.base import clone

from .cross_validation import cross_val_score


class ParameterGrid(object):

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)


class GridSearchCV(object):

    def __init__(self, estimator, param_grid, cv, refit=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.refit = refit

    def fit(self, X):
        param_grid = list(ParameterGrid(self.param_grid))
        n_folds = len(self.cv)
        n_grid = len(param_grid)

        scores = np.zeros((n_folds, n_grid), dtype=np.float64)

        for i, (X_tr, X_te) in enumerate(self.cv.split(X)):
            for j, params in enumerate(param_grid):
                estimator = clone(self.estimator)
                estimator.set_params(**params)
                estimator.fit(X_tr)

                scores[i, j] = estimator.score(X_te)

        # FIXME: handle higher is better as well.
        best = scores.mean(axis=0).argmin()
        self.best_params_ = param_grid[best]

        # Refit
        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X)

        return self

    @property
    def predict(self):
        return self.best_estimator_.predict

    @property
    def score(self):
        return self.best_estimator_.score

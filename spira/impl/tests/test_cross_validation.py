import numpy as np
import scipy.sparse as sp

from spira.cross_validation import cross_val_score
from spira.cross_validation import ShuffleSplit
from spira.completion import ExplicitMF

from testing import assert_equal


def test_shuffle_split():
    X = [[3, 0, 0, 1],
         [2, 0, 5, 0],
         [0, 4, 3, 0],
         [0, 0, 2, 0]]
    X = sp.coo_matrix(X)

    cv = ShuffleSplit(n_iter=10)
    for X_tr, X_te in cv.split(X):
        assert_equal(X.shape, X_tr.shape)
        assert_equal(X.shape, X_te.shape)
        assert_equal(X.data.shape[0],
                     X_tr.data.shape[0] + X_te.data.shape[0])

def test_cross_val_score():
    # Generate some toy data.
    rng = np.random.RandomState(0)
    U = rng.rand(50, 3)
    V = rng.rand(3, 20)
    X = np.dot(U, V)

    cv = ShuffleSplit(n_iter=10)
    mf = ExplicitMF(n_components=3, max_iter=10, alpha=1e-3, random_state=0,
                    verbose=0)
    scores = cross_val_score(mf, X, cv)
    assert_equal(len(scores), cv.n_iter)

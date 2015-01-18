import numpy as np

from spira.completion import ExplicitMF
from spira.grid_search import GridSearchCV
from spira.cross_validation import ShuffleSplit

from testing import assert_equal
from testing import assert_almost_equal


# Generate some toy data.
rng = np.random.RandomState(0)
U = rng.rand(50, 3)
V = rng.rand(3, 20)
X = np.dot(U, V)


def test_grid_search():
    cv = ShuffleSplit(n_iter=5, random_state=0)
    mf = ExplicitMF(n_components=3, max_iter=10, random_state=0)
    param_grid = {"alpha": [0.1, 1.0, 10]}
    gcv = GridSearchCV(mf, param_grid, cv)
    gcv.fit(X)

    assert_equal(gcv.best_estimator_.alpha, 0.1)
    assert_equal(gcv.best_params_, {"alpha": 0.1})

    mf = ExplicitMF(alpha=0.1, n_components=3, max_iter=10, random_state=0)
    mf.fit(X)

    assert_almost_equal(mf.score(X), gcv.score(X))

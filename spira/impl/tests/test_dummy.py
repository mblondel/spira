import numpy as np

from spira.completion import Dummy

from testing import assert_array_almost_equal

X = [[3, 0, 0, 1],
     [2, 0, 5, 0],
     [0, 4, 3, 0],
     [0, 0, 2, 0],
     [1, 0, 0, 0]]

X = np.array(X, dtype=np.float64)


def test_dummy():
    est = Dummy()
    est.fit(X)
    X_predicted = est.predict(X).toarray()
    X_expected = [[2, 0, 0, 2],
                  [3.5, 0, 3.5, 0],
                  [0, 3.5, 3.5, 0],
                  [0, 0, 2, 0],
                  [1, 0, 0, 0]]
    assert_array_almost_equal(X_predicted, X_expected)

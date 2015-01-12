# Author: Mathieu Blondel
# License: BSD

import sys

import numpy as np
import matplotlib.pyplot as plt

from spira.datasets import load_movielens
from spira.cross_validation import ShuffleSplit
from spira.cross_validation import cross_val_score
from spira.completion import MatrixFactorization
from spira.completion import Dummy

try:
    version = sys.argv[1]
except:
    version = "100k"

X = load_movielens(version)
print X.shape

alphas = np.logspace(-3, 0, 10)
mf_scores = []

for alpha in alphas:
    cv = ShuffleSplit(n_iter=3, train_size=0.75, random_state=0)
    mf = MatrixFactorization(n_components=30, max_iter=10, alpha=alpha)
    mf_scores.append(cross_val_score(mf, X, cv))

# Array of size n_alphas x n_folds.
mf_scores = np.array(mf_scores)

cv = ShuffleSplit(n_iter=3, train_size=0.75, random_state=0)
dummy = Dummy()
# Array of size n_folds.
dummy_scores = cross_val_score(mf, X, cv)

plt.figure()
plt.plot(alphas, mf_scores.mean(axis=1), label="Matrix Factorization")
plt.plot(alphas, [dummy_scores.mean()] * len(alphas), label="Dummy")
plt.xlabel("alpha")
plt.xscale("log")
plt.ylabel("RMSE")
plt.legend()
plt.show()

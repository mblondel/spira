# Author: Mathieu Blondel
# License: BSD

import sys

import numpy as np
import matplotlib.pyplot as plt

from spira.datasets import load_movielens
from spira.cross_validation import ShuffleSplit
from spira.cross_validation import cross_val_score
from spira.completion import ExplicitMF
from spira.completion import Dummy

try:
    version = sys.argv[1]
except:
    version = "100k"

X = load_movielens(version)
print X.shape

alphas = np.logspace(-3, 0, 10)
mf_scores = []

cv = ShuffleSplit(n_iter=3, train_size=0.75, random_state=0)

for alpha in alphas:
    mf = ExplicitMF(n_components=30, max_iter=10, alpha=alpha)
    mf_scores.append(cross_val_score(mf, X, cv))

# Array of size n_alphas x n_folds.
mf_scores = np.array(mf_scores)

dummy = Dummy()
dummy_scores = cross_val_score(dummy, X, cv)

dummy = Dummy(axis=0)
dummy_scores2 = cross_val_score(dummy, X, cv)

plt.figure()
plt.plot(alphas, mf_scores.mean(axis=1), label="Matrix Factorization")
plt.plot(alphas, [dummy_scores.mean()] * len(alphas), label="User mean")
plt.plot(alphas, [dummy_scores2.mean()] * len(alphas), label="Movie mean")
plt.xlabel("alpha")
plt.xscale("log")
plt.ylabel("RMSE")
plt.legend()
plt.show()

# Author: Mathieu Blondel
# License: BSD

import sys

import numpy as np
import matplotlib.pyplot as plt

from spira.datasets import load_movielens
from spira.cross_validation import ShuffleSplit
from spira.cross_validation import cross_val_score
from spira.completion import MatrixFactorization

try:
    version = sys.argv[1]
except:
    version = "100k"

X = load_movielens(version)
print X.shape

alphas = np.logspace(-3, 0, 10)
scores = []

for alpha in alphas:
    cv = ShuffleSplit(n_iter=5, train_size=0.75, random_state=0)
    mf = MatrixFactorization(n_components=30, max_iter=10, alpha=alpha)
    scores.append(cross_val_score(mf, X, cv))

# n_alphas x n_folds
scores = np.array(scores)

plt.figure()
plt.plot(alphas, scores.mean(axis=1))
plt.xlabel("alpha")
plt.xscale("log")
plt.ylabel("RMSE")
plt.show()

# Author: Mathieu Blondel
# License: BSD

import time
import sys

import numpy as np
import matplotlib.pyplot as plt

from spira.datasets import load_movielens
from spira.cross_validation import train_test_split
from spira.completion import ExplicitMF


def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


class Callback(object):

    def __init__(self, X_tr, X_te):
        self.X_tr = X_tr
        self.X_te = X_te
        self.obj = []
        self.rmse = []
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()

        X_pred = mf.predict(self.X_tr)
        loss = 0.5 * sqnorm(X_pred.data - self.X_tr.data)
        regul = 0.5 * mf.alpha * (sqnorm(mf.P_) + sqnorm(mf.Q_))
        self.obj.append(loss + regul)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        self.rmse.append(rmse)

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() -  self.start_time - self.test_time)

try:
    version = sys.argv[1]
except:
    version = "100k"

X = load_movielens(version)
print X.shape

X_tr, X_te = train_test_split(X, train_size=0.75, random_state=0)
X_tr = X_tr.tocsr()
X_te = X_te.tocsr()

cb = Callback(X_tr, X_te)
mf = ExplicitMF(n_components=30, max_iter=50, alpha=0.1, verbose=1, callback=cb)
mf.fit(X_tr)

plt.figure()
plt.plot(cb.times, cb.obj)
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("Objective value")

plt.figure()
plt.plot(cb.times, cb.rmse)
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("RMSE")

plt.show()

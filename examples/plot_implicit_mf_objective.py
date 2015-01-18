# Author: Mathieu Blondel
# License: BSD

import time
import sys

import numpy as np
import matplotlib.pyplot as plt

from spira.datasets import load_movielens
from spira.cross_validation import train_test_split
from spira.completion import ImplicitMF


def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


def error(X, P, Q):
    # Compute ||X - PQ||^2_F (taking into account implicitly stored zeros)
    # in an efficient manner.
    error = np.dot(X.data, X.data)
    XP = X.T * P  # sparse dot
    PP = np.dot(P.T, P)
    error -= 2 * np.trace(np.dot(XP, Q))
    error += np.trace(Q.T.dot(PP).dot(Q))
    return error


class Callback(object):

    def __init__(self, X_tr, X_te):
        self.X_tr = X_tr
        self.X_te = X_te
        self.obj = []
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()

        loss = 0.5 * error(self.X_tr, mf.P_, mf.Q_)
        regul = 0.5 * mf.alpha * (sqnorm(mf.P_) + sqnorm(mf.Q_))
        self.obj.append(loss + regul)

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() -  self.start_time - self.test_time)

try:
    version = sys.argv[1]
except:
    version = "100k"

X = load_movielens(version)
print X.shape

# Binarize and pretend this is implicit feedback.
cond = X.data > X.data.mean()
X.data[cond] = 1
X.data[~cond] = 0

X_tr, X_te = train_test_split(X, train_size=0.75, random_state=0)
X_tr = X_tr.tocsr()
X_te = X_te.tocsr()

cb = Callback(X_tr, X_te)
mf = ImplicitMF(n_components=30, max_iter=50, alpha=0.1, callback=cb)
mf.fit(X_tr)

plt.figure()
plt.plot(cb.times, cb.obj)
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("Objective value")

plt.show()

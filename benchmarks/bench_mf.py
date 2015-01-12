import sys
import time

from sklearn.externals import joblib

from spira.datasets import load_movielens
from spira.completion import MatrixFactorization

try:
    version = sys.argv[1]
except:
    version = "100k"

X = load_movielens(version)
print X.shape

start = time.time()
mf = MatrixFactorization(n_components=30, max_iter=10,
                         alpha=1e-1, random_state=0, verbose=1)
mf.fit(X)
print "Time", time.time() - start
print "RMSE", mf.score(X)

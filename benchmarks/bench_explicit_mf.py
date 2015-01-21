import sys
import time

from spira.datasets import load_movielens
from spira.cross_validation import train_test_split
from spira.completion import ExplicitMF

try:
    version = sys.argv[1]
except:
    version = "100k"

X = load_movielens(version)
print X.shape

X_tr, X_te = train_test_split(X, train_size=0.75, random_state=0)

start = time.time()
mf = ExplicitMF(n_components=30, max_iter=10, alpha=1e-1, random_state=0,
                verbose=1)
mf.fit(X_tr)
print "Time", time.time() - start
print "RMSE", mf.score(X_te)

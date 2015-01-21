import sys
import time

from spira.datasets import load_movielens
from spira.cross_validation import train_test_split
from spira.completion import ImplicitMF
from spira.metrics import average_precision

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

start = time.time()
mf = ImplicitMF(n_components=30, max_iter=10, alpha=1e-1, random_state=0,
                verbose=1)
mf.fit(X_tr)
print "Time", time.time() - start
X_score = mf.decision_function(X_te)
print "Average Precision", average_precision(X_te, X_score)

# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp


def _check(X_true, X_pred):
    if X_true.shape != X_pred.shape:
        raise ValueError("X_true and X_pred should have the same shape.")

    # FIXME: we just need to check that both matrix
    # have the same sparse format.
    X_true = sp.csr_matrix(X_true)
    X_pred = sp.csr_matrix(X_pred)

    return X_true, X_pred


def rmse(X_true, X_pred):
    X_true, X_pred = _check(X_true, X_pred)
    mse = np.mean((X_true.data - X_pred.data) ** 2)
    return np.sqrt(mse)


def precision(X_true, X_pred):
    X_true, X_pred = _check(X_true, X_pred)
    tp = np.logical_and(X_true.data == 1, X_pred.data == 1).sum()
    n_pos_pred = np.sum(X_pred.data == 1)
    # Proportion of truly positives in predicted positives.
    return float(tp) / n_pos_pred


def recall(X_true, X_pred):
    X_true, X_pred = _check(X_true, X_pred)
    tp = np.logical_and(X_true.data == 1, X_pred.data == 1).sum()
    n_pos = np.sum(X_true.data == 1)
    # Proportion of positives correctly predicted as positive.
    return float(tp) / n_pos


def f1_score(X_true, X_pred):
    X_true, X_pred = _check(X_true, X_pred)
    p = precision(X_true, X_pred)
    r = recall(X_true, X_pred)
    # Harmonic mean of precision and recall.
    return 2 * p * r / (p + r)


def average_precision(X_true, X_score):
    y_true, y_score = X_true.data, X_score.data
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    n = len(y_true)
    n_pos = np.sum(y_true == pos_label)
    i = np.arange(n)[::-1]
    s = np.cumsum(y_sorted[i] / (i + 1.0))

    return np.sum(s * y_sorted[i]) / n_pos
